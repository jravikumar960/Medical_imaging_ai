"""Local HTTP API for the same pretrained TorchXRayVision inference used by Streamlit."""
from __future__ import annotations

import logging
import threading
from io import BytesIO
import base64

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import numpy as np
import torch
import torchxrayvision as xrv

from dicom_io import DicomInputError, dicom_bytes_to_cxr_candidate
from imaging import gradcam, prepare_image, overlay_heatmap, open_uploaded_image
from input_suitability import InputSuitabilityError, require_verified_cxr_modality_image
from doctor_reviews import router as doctor_reviews_router
from reviews_api import router as reviews_api_router
import matplotlib.pyplot as plt


MAX_REQUEST_BYTES = 25 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "application/dicom"}
MODEL_WEIGHTS = "densenet121-res224-all"
logger = logging.getLogger(__name__)
_MODEL_LOCK = threading.Lock()
_MODEL_STATE_LOCK = threading.Lock()
_MODEL = None
_MODEL_STATUS = "not_loaded"
_MODEL_ERROR = None

api = FastAPI(title="Local Chest X-ray Research Inference API", docs_url=None, redoc_url=None)
api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Authorization", "Content-Type"],
)
api.include_router(doctor_reviews_router)
api.include_router(reviews_api_router)


def load_model():
    global _MODEL, _MODEL_STATUS, _MODEL_ERROR
    with _MODEL_LOCK:
        with _MODEL_STATE_LOCK:
            if _MODEL is not None:
                return _MODEL
            _MODEL_STATUS = "loading"
            _MODEL_ERROR = None
        try:
            model = xrv.models.DenseNet(weights=MODEL_WEIGHTS)
            model.eval()
        except Exception as exc:
            with _MODEL_STATE_LOCK:
                _MODEL_STATUS = "failed"
                _MODEL_ERROR = type(exc).__name__
            logger.exception("TorchXRayVision model initialization failed")
            raise ModelInitializationError(
                "The pretrained TorchXRayVision model could not be initialized. "
                "Check the API server log and verify the checkpoint cache or network access."
            ) from exc
        with _MODEL_STATE_LOCK:
            _MODEL = model
            _MODEL_STATUS = "loaded"
            return _MODEL


class ModelInitializationError(RuntimeError):
    """Raised when the configured pretrained model cannot be initialized."""


def model_status():
    with _MODEL_STATE_LOCK:
        return {
            "model_loaded": _MODEL is not None,
            "model_status": _MODEL_STATUS,
            "model_error": _MODEL_ERROR,
        }


def infer_bytes(data: bytes, content_type: str, model=None, target_finding: str | None = None):
    if content_type == "application/dicom":
        image = dicom_bytes_to_cxr_candidate(data)
    elif content_type in {"image/png", "image/jpeg"}:
        expected_format = {"image/png": "PNG", "image/jpeg": "JPEG"}[content_type]
        try:
            image = open_uploaded_image(BytesIO(data), expected_format=expected_format)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("Request did not contain a readable PNG or JPEG image.") from exc
    else:
        raise ValueError("Use PNG, JPEG, or single-frame DICOM content.")

    verification = require_verified_cxr_modality_image(image, content_type)
    input_warning = (
        f"Modality verified as chest X-ray "
        f"(score {verification.score:.6f}, threshold {verification.threshold:.6f}; "
        "verifier scores are not calibrated probabilities)."
    )

    input_tensor, gray = prepare_image(image)
    if model is None:
        model = load_model()
    with torch.inference_mode():
        output = model(input_tensor)
    targets = list(getattr(model, "pathologies", []))
    if output.ndim != 2 or output.shape[0] != 1 or output.shape[1] != len(targets):
        raise RuntimeError("The model returned an invalid output shape.")
    scores = output[0].detach().cpu().numpy()
    if not np.isfinite(scores).all() or any(not isinstance(item, str) or not item for item in targets):
        raise RuntimeError("The model returned invalid labels or non-finite scores.")
    if getattr(model, "op_threshs", None) is None:
        score_type = "model_output_score"
    else:
        score_type = "operating_point_normalized_score"
        if np.any(scores < 0) or np.any(scores > 1):
            raise RuntimeError("The model returned an out-of-range operating-point-normalized score.")
    sorted_order = np.argsort(-scores, kind="stable").tolist()
    if target_finding is not None:
        if not isinstance(target_finding, str) or not target_finding.strip():
            raise ValueError("Target finding must be a non-empty string.")
        clean_target = target_finding.strip()
        if clean_target not in targets:
            raise ValueError(
                f"Unknown target finding '{clean_target}'. Must be one of: {', '.join(targets)}."
            )
        selected_index = targets.index(clean_target)
    else:
        selected_index = int(np.argmax(scores))

    selected_finding = targets[selected_index]
    selected_score = float(scores[selected_index])
    selected_rank = sorted_order.index(selected_index) + 1

    cam, _ = gradcam(model, input_tensor, selected_index)
    figure = overlay_heatmap(gray, cam)
    try:
        rendered = BytesIO()
        figure.savefig(rendered, format="png", bbox_inches="tight", pad_inches=0)
        rendered.seek(0)
        image_buffer = BytesIO()
        with Image.open(rendered) as overlay_image:
            overlay_image.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS).save(
                image_buffer, format="PNG"
            )
        overlay_png = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    finally:
        plt.close(figure)
    preview_buffer = BytesIO()
    gray.save(preview_buffer, format="PNG")
    input_preview_png = base64.b64encode(preview_buffer.getvalue()).decode("ascii")
    return {
        "model": MODEL_WEIGHTS,
        "task": "Multi-label chest X-ray finding scoring",
        "score_type": score_type,
        "input_warning": input_warning,
        "input_preview_png_base64": input_preview_png,
        "target_finding": selected_finding,
        "target_index": selected_index,
        "target_score": selected_score,
        "target_rank": selected_rank,
        "predictions": [
            {"finding": target, "score": float(score)}
            for target, score in zip(targets, scores)
        ],
        "gradcam": {
            "finding": selected_finding,
            "target_finding": selected_finding,
            "index": selected_index,
            "score": selected_score,
            "rank": selected_rank,
            "overlay_png_base64": overlay_png,
        },
    }


@api.get("/health")
async def health():
    return {"status": "ok", **model_status()}


@api.post("/infer")
async def infer(request: Request, target_finding: str | None = None):
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Use PNG, JPEG, or single-frame DICOM content.")
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BYTES:
                raise HTTPException(status_code=413, detail="Image exceeds the request size limit.")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid request size.") from None
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_REQUEST_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds the request size limit.")
        chunks.append(chunk)

    finding_param = target_finding
    if finding_param is None:
        finding_param = request.query_params.get("target_finding")
    if finding_param is None:
        finding_param = request.headers.get("x-target-finding")
    if finding_param is not None and not finding_param.strip():
        finding_param = None

    try:
        return infer_bytes(b"".join(chunks), content_type, target_finding=finding_param)
    except InputSuitabilityError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except (ValueError, DicomInputError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except ModelInitializationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except Exception:
        logger.exception("X-ray inference or Grad-CAM failed")
        raise HTTPException(
            status_code=503,
            detail="X-ray inference or Grad-CAM failed. Check the API server log for diagnostic details.",
        ) from None
