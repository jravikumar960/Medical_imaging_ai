"""Legitimate DICOM CT to NIfTI conversion and provenance-record generation.

This utility extracts source DICOM metadata and a geometry-preserving NIfTI
volume from a real CT series.  It does not fabricate data: every field is read
from the supplied DICOM files, and malformed or inconsistent series are rejected.

The generated files follow the schemas validated by ``ct_provenance.py``:

* ``medical-imaging-source-dicom-series/v1``
* ``medical-imaging-nifti-conversion/v1``

The converter always writes NIfTI voxels in ascending physical slice order and
records an identity voxel-to-source index map.  It never creates reviewer
attestations; reviewer approval must be supplied separately through
``orientation_review.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import pydicom
except ImportError as exc:  # pragma: no cover - environment guard
    raise RuntimeError("ct_dicom_conversion requires pydicom in the active environment.") from exc

try:
    import nibabel as nib
except ImportError as exc:  # pragma: no cover - environment guard
    raise RuntimeError("ct_dicom_conversion requires nibabel in the active environment.") from exc


SOURCE_SCHEMA = "medical-imaging-source-dicom-series/v1"
CONVERSION_SCHEMA = "medical-imaging-nifti-conversion/v1"
CONVERSION_SOFTWARE = "medical-imaging-ai-ct-dicom-conversion"
CONVERSION_VERSION = "1.0.0"


class DicomConversionError(ValueError):
    """A DICOM series is missing, unreadable, inconsistent, or unsupported."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_floats(value: Any, length: int, field: str) -> list[float]:
    try:
        if len(value) != length:
            raise DicomConversionError(f"{field} must contain {length} values.")
    except TypeError as exc:
        raise DicomConversionError(f"{field} must contain {length} values.") from exc
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise DicomConversionError(f"{field} must contain finite numbers.") from exc
    if not all(math.isfinite(item) for item in result):
        raise DicomConversionError(f"{field} must contain finite numbers.")
    return result

def _dataset_uid(dataset: Any, tag: str) -> str:
    value = getattr(dataset, tag, None)
    if not isinstance(value, str) or not value.strip():
        raise DicomConversionError(f"DICOM {tag} is required.")
    return value.strip()


def load_dicom_series(input_dir: str | Path) -> list[Any]:
    """Read all CT slices from one DICOM series directory."""
    root = Path(input_dir)
    if not root.is_dir():
        raise DicomConversionError(f"DICOM input directory does not exist: {root}")
    datasets: list[Any] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            dataset = pydicom.dcmread(str(path), force=False)
        except Exception:
            continue
        if not hasattr(dataset, "PixelData"):
            continue
        if getattr(dataset, "Modality", None) != "CT":
            raise DicomConversionError(f"{path} is not a CT instance (Modality={getattr(dataset, 'Modality', None)!r}).")
        datasets.append(dataset)
    if not datasets:
        raise DicomConversionError("No readable CT DICOM instances with pixel data were found.")
    series_uids = {_dataset_uid(ds, "SeriesInstanceUID") for ds in datasets}
    if len(series_uids) != 1:
        raise DicomConversionError("Input directory contains more than one DICOM series.")
    frame_uids = {_dataset_uid(ds, "FrameOfReferenceUID") for ds in datasets}
    if len(frame_uids) != 1:
        raise DicomConversionError("Input directory contains more than one FrameOfReferenceUID.")
    return datasets


def _series_geometry(datasets: Sequence[Any]) -> dict[str, Any]:
    """Validate a single-frame CT series and return sorted slice geometry."""
    records: list[dict[str, Any]] = []
    for index, dataset in enumerate(datasets):
        try:
            pixel_array = np.asarray(dataset.pixel_array)
        except Exception as exc:
            raise DicomConversionError(f"Cannot read pixel data from DICOM instance {index}.") from exc
        if pixel_array.ndim != 2:
            raise DicomConversionError(
                f"DICOM instance {index} is not a single-frame 2D CT slice (shape={pixel_array.shape!r})."
            )
        rows = int(getattr(dataset, "Rows"))
        columns = int(getattr(dataset, "Columns"))
        if rows <= 0 or columns <= 0 or pixel_array.shape != (rows, columns):
            raise DicomConversionError(f"DICOM instance {index} has inconsistent Rows/Columns/pixel shape.")
        orientation = _as_floats(getattr(dataset, "ImageOrientationPatient", None), 6, "ImageOrientationPatient")
        row_cosine = orientation[:3]
        column_cosine = orientation[3:]
        row_norm = math.sqrt(sum(value * value for value in row_cosine))
        column_norm = math.sqrt(sum(value * value for value in column_cosine))
        dot = sum(a * b for a, b in zip(row_cosine, column_cosine))
        if abs(row_norm - 1.0) > 0.002 or abs(column_norm - 1.0) > 0.002 or abs(dot) > 0.002:
            raise DicomConversionError(f"DICOM instance {index} has non-orthonormal ImageOrientationPatient.")
        normal = [
            row_cosine[1] * column_cosine[2] - row_cosine[2] * column_cosine[1],
            row_cosine[2] * column_cosine[0] - row_cosine[0] * column_cosine[2],
            row_cosine[0] * column_cosine[1] - row_cosine[1] * column_cosine[0],
        ]
        position = _as_floats(getattr(dataset, "ImagePositionPatient", None), 3, "ImagePositionPatient")
        spacing = _as_floats(getattr(dataset, "PixelSpacing", None), 2, "PixelSpacing")
        if min(spacing) <= 0:
            raise DicomConversionError(f"DICOM instance {index} has non-positive PixelSpacing.")
        sop_uid = _dataset_uid(dataset, "SOPInstanceUID")
        series_uid = _dataset_uid(dataset, "SeriesInstanceUID")
        frame_uid = _dataset_uid(dataset, "FrameOfReferenceUID")
        patient_position = getattr(dataset, "PatientPosition", None)
        if not isinstance(patient_position, str) or not patient_position.strip():
            raise DicomConversionError(f"DICOM instance {index} is missing PatientPosition.")
        spacing_between = getattr(dataset, "SpacingBetweenSlices", None)
        if spacing_between is None:
            spacing_between_value = None
        else:
            try:
                spacing_between_value = float(spacing_between)
            except (TypeError, ValueError) as exc:
                raise DicomConversionError(f"DICOM instance {index} has invalid SpacingBetweenSlices.") from exc
            if not math.isfinite(spacing_between_value) or spacing_between_value <= 0:
                raise DicomConversionError(f"DICOM instance {index} has invalid SpacingBetweenSlices.")
        slope = getattr(dataset, "RescaleSlope", None)
        intercept = getattr(dataset, "RescaleIntercept", None)
        rescale = None
        if slope is not None or intercept is not None:
            try:
                rescale = (float(slope if slope is not None else 1.0),
                           float(intercept if intercept is not None else 0.0))
            except (TypeError, ValueError) as exc:
                raise DicomConversionError(f"DICOM instance {index} has invalid rescale metadata.") from exc
            if not all(math.isfinite(value) for value in rescale):
                raise DicomConversionError(f"DICOM instance {index} has non-finite rescale metadata.")
        records.append({
            "dataset": dataset,
            "sop_uid": sop_uid,
            "series_uid": series_uid,
            "frame_uid": frame_uid,
            "patient_position": patient_position.strip(),
            "orientation": orientation,
            "normal": normal,
            "position": position,
            "spacing": spacing,
            "rows": rows,
            "columns": columns,
            "spacing_between": spacing_between_value,
            "rescale": rescale,
            "pixel_array": pixel_array,
        })

    if len({record["sop_uid"] for record in records}) != len(records):
        raise DicomConversionError("DICOM series contains duplicate SOPInstanceUID values.")
    reference = records[0]
    for record in records[1:]:
        if any(not math.isclose(a, b, abs_tol=1e-4, rel_tol=0)
               for a, b in zip(record["orientation"], reference["orientation"])):
            raise DicomConversionError("DICOM ImageOrientationPatient is inconsistent across slices.")
        if record["spacing"] != reference["spacing"] or record["rows"] != reference["rows"] or \
                record["columns"] != reference["columns"] or record["patient_position"] != reference["patient_position"]:
            raise DicomConversionError("DICOM slice dimensions, PixelSpacing, or PatientPosition are inconsistent.")
        if record["rescale"] != reference["rescale"]:
            raise DicomConversionError("DICOM rescale slope/intercept are inconsistent across slices.")

    orientation = reference["orientation"]
    row_cosine = orientation[:3]
    column_cosine = orientation[3:]
    normal = [
        row_cosine[1] * column_cosine[2] - row_cosine[2] * column_cosine[1],
        row_cosine[2] * column_cosine[0] - row_cosine[0] * column_cosine[2],
        row_cosine[0] * column_cosine[1] - row_cosine[1] * column_cosine[0],
    ]
    ordered = sorted(records, key=lambda record: sum(record["position"][axis] * normal[axis] for axis in range(3)))
    projections = [sum(record["position"][axis] * normal[axis] for axis in range(3)) for record in ordered]
    steps = [projections[i + 1] - projections[i] for i in range(len(projections) - 1)]
    if len(ordered) < 2:
        raise DicomConversionError("A CT series must contain at least two slices for orientation verification.")
    if any(step <= 1e-3 for step in steps):
        raise DicomConversionError("DICOM slices have duplicate or non-increasing physical positions.")
    nominal_values = [record["spacing_between"] for record in ordered]
    if all(value is None for value in nominal_values):
        nominal_value = sum(steps) / len(steps)
        for record in ordered:
            record["spacing_between"] = nominal_value
    else:
        if any(value is None for value in nominal_values):
            raise DicomConversionError("DICOM SpacingBetweenSlices is inconsistently present.")
        if any(not math.isclose(value, nominal_values[0], abs_tol=1e-3, rel_tol=1e-4) for value in nominal_values):
            raise DicomConversionError("DICOM SpacingBetweenSlices is inconsistent.")
        nominal_value = nominal_values[0]
    if any(not math.isclose(step, nominal_value, abs_tol=1e-3, rel_tol=1e-4) for step in steps):
        raise DicomConversionError("DICOM slice ordering has a gap or inconsistent spacing.")
    step_vectors = [[ordered[i + 1]["position"][axis] - ordered[i]["position"][axis]
                     for axis in range(3)] for i in range(len(ordered) - 1)]
    mean_step = [sum(vector[axis] for vector in step_vectors) / len(step_vectors) for axis in range(3)]
    if any(any(not math.isclose(vector[axis], mean_step[axis], abs_tol=1e-3, rel_tol=1e-4)
               for axis in range(3)) for vector in step_vectors):
        raise DicomConversionError("DICOM slice positions do not form a consistent affine stack.")

    return {
        "ordered": ordered,
        "orientation": orientation,
        "normal": normal,
        "mean_step": mean_step,
        "series_uid": reference["series_uid"],
        "frame_uid": reference["frame_uid"],
        "patient_position": reference["patient_position"],
        "spacing": reference["spacing"],
        "rows": reference["rows"],
        "columns": reference["columns"],
        "spacing_between": ordered[0]["spacing_between"],
        "rescale": reference["rescale"],
    }

def build_source_metadata(series: dict[str, Any]) -> dict[str, Any]:
    """Build the exact source DICOM metadata object from validated series geometry."""
    ordered = series["ordered"]
    slices = []
    for record in ordered:
        dataset = record["dataset"]
        slices.append({
            "sop_instance_uid": record["sop_uid"],
            "series_instance_uid": record["series_uid"],
            "frame_of_reference_uid": record["frame_uid"],
            "modality": "CT",
            "patient_position": record["patient_position"],
            "image_orientation_patient": [float(value) for value in record["orientation"]],
            "image_position_patient": [float(value) for value in record["position"]],
            "rows": record["rows"],
            "columns": record["columns"],
            "pixel_spacing_mm": [float(value) for value in record["spacing"]],
            "spacing_between_slices_mm": float(record["spacing_between"]),
        })
    return {
        "schema": SOURCE_SCHEMA,
        "patient_coordinate_system": "LPS",
        "series_instance_uid": series["series_uid"],
        "frame_of_reference_uid": series["frame_uid"],
        "slices": slices,
    }


def _ras_affine(series: dict[str, Any]) -> np.ndarray:
    orientation = series["orientation"]
    row_cosine = orientation[:3]
    column_cosine = orientation[3:]
    spacing = series["spacing"]
    first_position = series["ordered"][0]["position"]
    last_position = series["ordered"][-1]["position"]
    count = len(series["ordered"])
    step_vector = [(last_position[axis] - first_position[axis]) / (count - 1) for axis in range(3)]
    dicom_to_lps = [
        [row_cosine[axis] * spacing[1], column_cosine[axis] * spacing[0], step_vector[axis], first_position[axis]]
        for axis in range(3)
    ] + [[0.0, 0.0, 0.0, 1.0]]
    expected_ras = [
        [-dicom_to_lps[0][column] for column in range(4)],
        [-dicom_to_lps[1][column] for column in range(4)],
        [dicom_to_lps[2][column] for column in range(4)],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return np.asarray(expected_ras, dtype=np.float64)


def convert_series(
    datasets: Sequence[Any],
    output_nifti: str | Path,
    source_metadata_path: str | Path,
    conversion_record_path: str | Path,
) -> dict[str, Any]:
    """Convert one validated CT series and write source metadata plus conversion record."""
    series = _series_geometry(datasets)
    ordered = series["ordered"]
    rows = series["rows"]
    columns = series["columns"]
    count = len(ordered)
    slices = [record["pixel_array"] for record in ordered]
    volume = np.stack(slices, axis=0)  # (slice, row, column)
    volume = np.transpose(volume, (2, 1, 0))  # (column, row, slice) == (i, j, k)
    if series["rescale"] is not None:
        slope, intercept = series["rescale"]
        volume = volume.astype(np.float32) * np.float32(slope) + np.float32(intercept)
    else:
        volume = volume.astype(np.float32)

    affine = _ras_affine(series)
    image = nib.Nifti1Image(volume, affine)
    image.header["qform_code"] = 1
    image.header["sform_code"] = 1
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    output_path = Path(output_nifti)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(output_path))

    source_metadata = build_source_metadata(series)
    metadata_path = Path(source_metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_bytes = (json.dumps(source_metadata, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    metadata_path.write_bytes(metadata_bytes)
    # Re-read canonical bytes written to disk so hashes bind to exact file bytes.
    metadata_bytes = metadata_path.read_bytes()

    volume_sha256 = _sha256_file(output_path)
    metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
    conversion_record = {
        "schema": CONVERSION_SCHEMA,
        "nifti_affine_coordinate_system": "RAS",
        "nifti_sha256": volume_sha256,
        "source_metadata_sha256": metadata_sha256,
        "nifti_shape": [int(value) for value in volume.shape],
        "source_sop_instance_uids_by_slice_index": [record["sop_uid"] for record in ordered],
        "conversion_software": CONVERSION_SOFTWARE,
        "conversion_version": CONVERSION_VERSION,
        "nifti_voxel_to_source_indices": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    }
    conversion_path = Path(conversion_record_path)
    conversion_path.parent.mkdir(parents=True, exist_ok=True)
    conversion_bytes = (json.dumps(conversion_record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    conversion_path.write_bytes(conversion_bytes)
    conversion_bytes = conversion_path.read_bytes()
    # Keep the in-memory hash consistent with the exact bytes on disk.
    conversion_record["conversion_record_sha256_not_stored"] = None
    del conversion_record["conversion_record_sha256_not_stored"]
    return {
        "nifti_path": str(output_path),
        "source_metadata_path": str(metadata_path),
        "conversion_record_path": str(conversion_path),
        "series_instance_uid": series["series_uid"],
        "frame_of_reference_uid": series["frame_uid"],
        "nifti_sha256": volume_sha256,
        "source_metadata_sha256": metadata_sha256,
        "conversion_record_sha256": hashlib.sha256(conversion_bytes).hexdigest(),
        "ordered_sop_instance_uids": [record["sop_uid"] for record in ordered],
    }


def convert_dicom_directory(
    input_dir: str | Path,
    output_nifti: str | Path,
    source_metadata_path: str | Path,
    conversion_record_path: str | Path,
) -> dict[str, Any]:
    """Read, validate, convert, and return a summary for one CT series directory."""
    datasets = load_dicom_series(input_dir)
    return convert_series(datasets, output_nifti, source_metadata_path, conversion_record_path)

def _build_review_from_conversion(
    result: dict[str, Any],
    geometry: dict[str, Any],
    reviewer_id: str,
    laterality_verified: bool,
    slice_direction_verified: bool,
    review_notes: str,
    sign: bool,
) -> dict[str, Any] | None:
    try:
        import orientation_review
    except ImportError:
        return None
    metadata_path = Path(result["source_metadata_path"])
    conversion_path = Path(result["conversion_record_path"])
    review = orientation_review.prepare_orientation_review(
        nifti_path=result["nifti_path"],
        geometry=geometry,
        source_metadata_path=metadata_path,
        conversion_record_path=conversion_path,
        reviewer_id=reviewer_id,
        laterality_verified=laterality_verified,
        slice_direction_verified=slice_direction_verified,
        review_notes=review_notes,
        sign=sign,
    )
    return review


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert a CT DICOM series to NIfTI and generate provenance records.")
    parser.add_argument("--input-dir", required=True, help="Directory containing one CT DICOM series.")
    parser.add_argument("--output-nifti", required=True, help="Output NIfTI path (.nii or .nii.gz).")
    parser.add_argument("--source-metadata", required=True, help="Output source DICOM metadata JSON path.")
    parser.add_argument("--conversion-record", required=True, help="Output NIfTI conversion record JSON path.")
    parser.add_argument("--review-output", help="Optional reviewer orientation evidence JSON path.")
    parser.add_argument("--reviewer-id", default="", help="Reviewer identity for an explicit orientation review.")
    parser.add_argument("--verify-laterality", action="store_true", help="Reviewer explicitly verified anatomy and laterality.")
    parser.add_argument("--verify-slice-direction", action="store_true", help="Reviewer explicitly verified physical slice direction.")
    parser.add_argument("--review-notes", default="", help="Human reviewer notes (required when generating review evidence).")
    parser.add_argument("--sign-review", action="store_true", help="Sign the review using CT_ORIENTATION_REVIEW_HMAC_KEY.")
    args = parser.parse_args(argv)

    result = convert_dicom_directory(
        args.input_dir,
        args.output_nifti,
        args.source_metadata,
        args.conversion_record,
    )
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.review_output:
        if not args.reviewer_id.strip():
            raise DicomConversionError("--reviewer-id is required when writing review evidence.")
        if not args.review_notes.strip():
            raise DicomConversionError("--review-notes is required when writing review evidence.")
        import segmentation
        geometry = segmentation.inspect_nifti_geometry(result["nifti_path"])
        review = _build_review_from_conversion(
            result,
            geometry,
            args.reviewer_id,
            args.verify_laterality,
            args.verify_slice_direction,
            args.review_notes,
            args.sign_review,
        )
        if review is None:
            raise DicomConversionError("orientation_review module is unavailable.")
        review_path = Path(args.review_output)
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote orientation review to {review_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
