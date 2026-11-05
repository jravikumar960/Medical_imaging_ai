"""Loopback FastAPI for pretrained, task-specific BraTS brain-tumor segmentation."""
from __future__ import annotations

import base64
import gzip
import hashlib
import logging
import threading
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import UploadFile
from starlette.concurrency import run_in_threadpool

import mri_segmentation


MAX_MRI_REQUEST_BYTES = 512 * 1024 * 1024
MAX_MRI_FILE_BYTES = 128 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
logger = logging.getLogger(__name__)
_MODEL = None
_MODEL_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()

app = FastAPI(title="BraTS Brain MRI Tumor Segmentation API", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _model():
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = mri_segmentation.load_model()
        return _MODEL


def _run_inference(paths: dict[str, Path], output_path: Path):
    with _INFERENCE_LOCK:
        return mri_segmentation.run_segmentation(paths, output_path, model_loader=_model)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "workflow": "BraTS 2018 multimodal brain tumor segmentation",
        "model": "MONAI/brats_mri_segmentation",
        "checkpoint_present": mri_segmentation.MODEL_PATH.is_file(),
        "model_loaded": _MODEL is not None,
        "required_sequences": list(mri_segmentation.MRI_CHANNELS),
        "intended_use": "Research demonstration only; not for diagnosis.",
    }


@app.post("/mri/segment")
async def segment_mri(request: Request):
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "multipart/form-data":
        raise HTTPException(status_code=415, detail="Upload T1c, T1, T2, and FLAIR as four multipart NIfTI files.")
    content_length = request.headers.get("content-length")
    if not content_length:
        raise HTTPException(status_code=411, detail="A bounded Content-Length is required for MRI uploads.")
    try:
        request_bytes = int(content_length)
        if request_bytes < 0:
            raise HTTPException(status_code=400, detail="Invalid request size.")
        if request_bytes > MAX_MRI_REQUEST_BYTES:
            raise HTTPException(status_code=413, detail="Combined MRI upload exceeds the 512 MiB request limit.")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid request size.") from None

    with tempfile.TemporaryDirectory(prefix="medical-imaging-mri-") as temp_dir:
        directory = Path(temp_dir)
        paths: dict[str, Path] = {}
        total_bytes = 0
        study_digest = hashlib.sha256()
        try:
            async with request.form(
                max_files=4,
                max_fields=0,
                max_part_size=MAX_MRI_FILE_BYTES,
            ) as form:
                if set(form.keys()) != set(mri_segmentation.MRI_CHANNELS):
                    raise HTTPException(
                        status_code=422,
                        detail="Exactly four file fields are required: t1c, t1, t2, and flair.",
                    )
                for channel in mri_segmentation.MRI_CHANNELS:
                    upload = form.get(channel)
                    if not isinstance(upload, UploadFile):
                        raise HTTPException(status_code=422, detail=f"{channel.upper()} must be a file upload.")
                    filename = (upload.filename or "").lower()
                    if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
                        raise HTTPException(
                            status_code=415,
                            detail=f"{channel.upper()} must be a .nii or .nii.gz NIfTI volume.",
                        )
                    suffix = ".nii.gz" if filename.endswith(".nii.gz") else ".nii"
                    path = directory / f"{channel}{suffix}"
                    file_bytes = 0
                    study_digest.update(channel.encode("ascii") + b"\0")
                    with path.open("wb") as target:
                        while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
                            file_bytes += len(chunk)
                            total_bytes += len(chunk)
                            if file_bytes > MAX_MRI_FILE_BYTES:
                                raise HTTPException(
                                    status_code=413,
                                    detail=f"{channel.upper()} exceeds the 128 MiB per-volume limit.",
                                )
                            if total_bytes > MAX_MRI_REQUEST_BYTES:
                                raise HTTPException(
                                    status_code=413,
                                    detail="Combined MRI upload exceeds the 512 MiB request limit.",
                                )
                            target.write(chunk)
                            study_digest.update(chunk)
                    if file_bytes == 0:
                        raise HTTPException(status_code=400, detail=f"{channel.upper()} upload is empty.")
                    paths[channel] = path
        except HTTPException:
            raise
        except Exception as exc:
            logger.info("MRI multipart request could not be parsed (%s)", type(exc).__name__)
            raise HTTPException(status_code=400, detail="MRI multipart request is malformed or unreadable.") from None

        output_path = directory / "brats_tumor_segmentation.nii.gz"
        try:
            result = await run_in_threadpool(
                _run_inference,
                paths,
                output_path,
            )
        except (mri_segmentation.MRISegmentationError, mri_segmentation.MRIDependencyError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
        except Exception as exc:
            logger.exception("BraTS MRI segmentation failed (%s)", type(exc).__name__)
            raise HTTPException(
                status_code=503,
                detail="BraTS MRI segmentation failed. Check the local MRI API log.",
            ) from None

        mask_data = gzip.compress(
            np.asfortranarray(result["mask"]).tobytes(order="F"),
            mtime=0,
        )
        preview_data = gzip.compress(
            np.asfortranarray(result["preview"]).tobytes(order="F"),
            mtime=0,
        )
        return {
            "status": "completed",
            "model": "MONAI/brats_mri_segmentation (BraTS MRI segmentation v0.5.4)",
            "task": "3D segmentation of primary glioma subregions from aligned T1c, T1, T2, and FLAIR MRI.",
            "intended_use": "Research demonstration only; not for diagnosis.",
            "input_sequence_order": list(mri_segmentation.MRI_CHANNELS),
            "study_id": study_digest.hexdigest(),
            "output_classes": {
                "1": "Tumor core (TC)",
                "2": "Whole-tumor remainder (WT excluding TC)",
                "4": "Enhancing tumor (ET)",
            },
            "threshold": 0.5,
            "geometry": {
                "shape": list(result["shape"]),
                "affine": result["affine"].tolist(),
                "spacing_mm": list(result["spacing_mm"]),
            },
            "inference_roi_size": list(result["roi_size"]),
            "label_counts": result["label_counts"],
            "mask_array_gzip_base64": base64.b64encode(mask_data).decode("ascii"),
            "flair_preview_gzip_base64": base64.b64encode(preview_data).decode("ascii"),
            "mask_nifti_gzip_base64": base64.b64encode(output_path.read_bytes()).decode("ascii"),
            "warning": (
                "This model is trained for BraTS 2018-style, skull-stripped, co-registered, 1 mm "
                "T1c/T1/T2/FLAIR primary-glioma inputs. File metadata cannot verify protocol, "
                "registration, skull stripping, or tumor type. The mask is not a diagnosis and "
                "the model is not clinically validated."
            ),
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("mri_api:app", host="127.0.0.1", port=8003, reload=False)
