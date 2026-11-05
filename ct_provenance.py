"""Verify reviewable source DICOM metadata and a geometry-preserving conversion record.

This module validates consistency and exact-file bindings. It cannot prove that
metadata was honestly extracted or that a reviewer is independent; those facts
still require external source records and human review.

The JSON input schemas are intentionally strict and documented in
``reports/ct_provenance_verifier.md``. No patient images or DICOM files are
loaded by this verifier.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence


SOURCE_SCHEMA = "medical-imaging-source-dicom-series/v1"
CONVERSION_SCHEMA = "medical-imaging-nifti-conversion/v1"
NIFTI_ORIGIN_SCHEMA = "medical-imaging-ct-nifti-origin/v1"
NIFTI_ORIGIN_TYPE = "nifti_dataset_distribution"
TASK09_DATASET_IMAGE_PATH = "./imagesTs/spleen_1.nii.gz"


class ProvenanceError(ValueError):
    """Source metadata or conversion record is missing, altered, or inconsistent."""


def _read_json(path: str | Path, description: str) -> tuple[dict[str, Any], str]:
    source = Path(path)
    if not source.is_file():
        raise ProvenanceError(f"Required {description} file is missing.")
    try:
        raw = source.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ProvenanceError(f"{description} file is unreadable JSON.") from None
    if not isinstance(value, dict):
        raise ProvenanceError(f"{description} must be a JSON object.")
    return value, hashlib.sha256(raw).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_nifti_origin_record(
    dataset_metadata_bytes: bytes,
    image_path: str,
    volume_sha256: str,
    geometry: dict[str, Any],
) -> dict[str, Any]:
    """Build the Task09 NIfTI-origin record from its distributed manifest and inspection."""
    try:
        metadata = json.loads(dataset_metadata_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise ProvenanceError("Task09 dataset.json must be readable UTF-8 JSON.") from None
    if not isinstance(metadata, dict):
        raise ProvenanceError("Task09 dataset.json must contain a JSON object.")
    if image_path != TASK09_DATASET_IMAGE_PATH:
        raise ProvenanceError("NIfTI-origin workflow accepts only the listed Task09 test image spleen_1.")
    if metadata.get("name") != "Spleen" or metadata.get("release") != "1.0 06/08/2018":
        raise ProvenanceError("dataset.json does not identify the expected Task09 Spleen release.")
    if metadata.get("description") != "Spleen Segmentation" or \
            metadata.get("reference") != "Memorial Sloan Kettering Cancer Center" or \
            metadata.get("licence") != "CC-BY-SA 4.0":
        raise ProvenanceError("dataset.json description, reference, or license is unsupported.")
    if metadata.get("modality") != {"0": "CT"}:
        raise ProvenanceError("Task09 dataset.json does not identify CT modality.")
    if metadata.get("labels") != {"0": "background", "1": "spleen"} or \
            metadata.get("numTraining") != 41 or metadata.get("numTest") != 20:
        raise ProvenanceError("Task09 dataset.json labels or declared split counts are unsupported.")
    training = metadata.get("training")
    test = metadata.get("test")
    if metadata.get("tensorImageSize") != "3D" or not isinstance(training, list) or not isinstance(test, list):
        raise ProvenanceError("Task09 dataset.json 3D image or split inventory is invalid.")
    if image_path not in test:
        raise ProvenanceError("dataset.json does not list spleen_1 as a test image.")
    if len(training) != 41 or len(test) != 20:
        raise ProvenanceError("Task09 dataset.json 3D image or split inventory is invalid.")
    if not isinstance(volume_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", volume_sha256):
        raise ProvenanceError("Inspected volume SHA-256 is missing or invalid.")
    if not isinstance(geometry, dict) or not isinstance(geometry.get("shape"), list) or \
            not isinstance(geometry.get("affine"), list):
        raise ProvenanceError("Server-inspected NIfTI geometry is missing or invalid.")
    geometry_bytes = json.dumps(geometry, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    geometry_sha256 = hashlib.sha256(geometry_bytes).hexdigest()
    return {
        "schema": NIFTI_ORIGIN_SCHEMA,
        "origin_type": NIFTI_ORIGIN_TYPE,
        "dataset": {
            "name": "Medical Segmentation Decathlon",
            "task": "Task09_Spleen",
            "dataset_name": metadata["name"],
            "description": metadata["description"],
            "reference": metadata["reference"],
            "release": metadata["release"],
            "license": metadata.get("licence"),
            "modality": "CT",
            "labels": metadata["labels"],
            "tensor_image_size": metadata["tensorImageSize"],
            "declared_num_training": metadata["numTraining"],
            "declared_num_test": metadata["numTest"],
            "split": "test",
            "image_path": image_path,
            "label_path": None,
            "dataset_json_sha256": hashlib.sha256(dataset_metadata_bytes).hexdigest(),
        },
        "distributed_representation": "NIfTI",
        "volume_source_authenticity": "not_independently_authenticated",
        "original_dicom_metadata": "not_supplied_in_the_dataset_distribution",
        "conversion_history": "unavailable",
        "volume": {
            "sha256": volume_sha256,
            "geometry_sha256": geometry_sha256,
            "geometry": geometry,
        },
    }


def verify_nifti_origin_provenance(
    record: dict[str, Any],
    geometry: dict[str, Any],
    review: dict[str, Any],
    *,
    expected_volume_sha256: str,
    expected_record_sha256: str,
) -> None:
    """Validate a generated NIfTI-origin record and its exact review/file bindings."""
    if not isinstance(record, dict) or record.get("schema") != NIFTI_ORIGIN_SCHEMA or \
            record.get("origin_type") != NIFTI_ORIGIN_TYPE:
        raise ProvenanceError("NIfTI-origin provenance schema or origin type is unsupported.")
    if set(record) != {
        "schema", "origin_type", "dataset", "distributed_representation",
        "volume_source_authenticity", "original_dicom_metadata",
        "conversion_history", "volume",
    }:
        raise ProvenanceError("NIfTI-origin record contains unsupported or DICOM-origin fields.")
    canonical_record = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    if hashlib.sha256(canonical_record).hexdigest() != expected_record_sha256:
        raise ProvenanceError("NIfTI-origin record bytes differ from the signed review record hash.")
    dataset = record.get("dataset")
    if not isinstance(dataset, dict) or set(dataset) != {
        "name", "task", "dataset_name", "description", "reference", "release",
        "license", "modality", "labels", "tensor_image_size",
        "declared_num_training", "declared_num_test",
        "split", "image_path", "label_path", "dataset_json_sha256",
    }:
        raise ProvenanceError("NIfTI-origin dataset metadata contains unsupported fields.")
    if not isinstance(dataset, dict) or dataset.get("name") != "Medical Segmentation Decathlon" or \
            dataset.get("task") != "Task09_Spleen" or dataset.get("dataset_name") != "Spleen" or \
            dataset.get("description") != "Spleen Segmentation" or \
            dataset.get("reference") != "Memorial Sloan Kettering Cancer Center" or \
            dataset.get("release") != "1.0 06/08/2018" or dataset.get("split") != "test" or \
            dataset.get("license") != "CC-BY-SA 4.0" or dataset.get("modality") != "CT" or \
            dataset.get("labels") != {"0": "background", "1": "spleen"} or \
            dataset.get("tensor_image_size") != "3D" or \
            dataset.get("declared_num_training") != 41 or dataset.get("declared_num_test") != 20 or \
            dataset.get("image_path") != TASK09_DATASET_IMAGE_PATH or dataset.get("label_path") is not None:
        raise ProvenanceError("NIfTI-origin record does not identify the selected Task09 Spleen test image.")
    if not isinstance(dataset.get("dataset_json_sha256"), str) or \
            not re.fullmatch(r"[0-9a-f]{64}", dataset["dataset_json_sha256"]):
        raise ProvenanceError("NIfTI-origin dataset.json digest is missing or invalid.")
    if record.get("distributed_representation") != "NIfTI" or \
            record.get("volume_source_authenticity") != "not_independently_authenticated" or \
            record.get("original_dicom_metadata") != "not_supplied_in_the_dataset_distribution" or \
            record.get("conversion_history") != "unavailable":
        raise ProvenanceError("NIfTI-origin record must accurately state unavailable DICOM and conversion history.")
    volume = record.get("volume")
    if not isinstance(volume, dict) or set(volume) != {"sha256", "geometry_sha256", "geometry"}:
        raise ProvenanceError("NIfTI-origin volume record contains unsupported fields.")
    if not isinstance(volume, dict) or volume.get("sha256") != expected_volume_sha256 or \
            review.get("volume_sha256") != expected_volume_sha256:
        raise ProvenanceError("NIfTI-origin provenance is not bound to these exact NIfTI bytes.")
    if volume.get("geometry") != geometry:
        raise ProvenanceError("NIfTI-origin record geometry differs from the server-inspected NIfTI geometry.")
    geometry_bytes = json.dumps(geometry, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    geometry_sha256 = hashlib.sha256(geometry_bytes).hexdigest()
    if volume.get("geometry_sha256") != geometry_sha256 or review.get("geometry_sha256") != geometry_sha256:
        raise ProvenanceError("NIfTI-origin provenance is not bound to the inspected geometry.")
    if review.get("origin_record_sha256") != expected_record_sha256:
        raise ProvenanceError("Orientation review is not bound to this exact NIfTI-origin record.")


def _numbers(value: Any, length: int, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ProvenanceError(f"{field} must contain {length} numbers.")
    try:
        numbers = [float(item) for item in value]
    except (TypeError, ValueError):
        raise ProvenanceError(f"{field} must contain finite numbers.") from None
    if not all(math.isfinite(item) for item in numbers):
        raise ProvenanceError(f"{field} must contain finite numbers.")
    return numbers


def _uid(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ProvenanceError(f"{field} is missing or invalid.")
    parts = value.split(".")
    if len(parts) < 2 or any(not part.isdigit() for part in parts):
        raise ProvenanceError(f"{field} is malformed.")
    return value


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _close(left: float, right: float, *, abs_tol: float = 1e-3, rel_tol: float = 1e-4) -> bool:
    return math.isclose(left, right, abs_tol=abs_tol, rel_tol=rel_tol)


def _validate_index_map(matrix: Any, shape: Sequence[int], source_shape: Sequence[int]) -> list[list[float]]:
    if not isinstance(matrix, list) or len(matrix) != 4:
        raise ProvenanceError("Conversion voxel-to-source index map must be a 4x4 matrix.")
    rows = [_numbers(row, 4, "Conversion voxel-to-source index map row") for row in matrix]
    if any(not _close(value, round(value), abs_tol=1e-8, rel_tol=0) for row in rows for value in row):
        raise ProvenanceError("Conversion voxel-to-source index map must use integer indices.")
    if rows[3] != [0.0, 0.0, 0.0, 1.0]:
        raise ProvenanceError("Conversion voxel-to-source index map has an invalid homogeneous row.")
    linear = [row[:3] for row in rows[:3]]
    if any(sum(abs(linear[r][c]) for c in range(3)) != 1 for r in range(3)) or \
            any(sum(abs(linear[r][c]) for r in range(3)) != 1 for c in range(3)):
        raise ProvenanceError("Conversion index map must be a signed axis permutation without resampling.")
    offsets = [int(rows[row][3]) for row in range(3)]
    for source_axis, source_extent in enumerate(source_shape):
        voxel_axis = next(column for column in range(3) if linear[source_axis][column] != 0)
        sign = int(linear[source_axis][voxel_axis])
        if int(shape[voxel_axis]) != int(source_extent):
            raise ProvenanceError(
                "NIfTI grid does not cover the entire source DICOM grid; axis extents differ."
            )
        expected_offset = 0 if sign > 0 else int(source_extent) - 1
        if offsets[source_axis] != expected_offset:
            raise ProvenanceError(
                "Conversion index map is shifted or cropped and does not cover the entire source DICOM grid."
            )
    return rows


def verify_source_provenance(
    nifti_path: str | Path | None,
    geometry: dict[str, Any],
    review: dict[str, Any],
    source_metadata_path: str | Path | None,
    conversion_record_path: str | Path | None,
    *,
    expected_nifti_sha256: str | None = None,
) -> None:
    """Verify source-record hashes, series identity, slice inventory, and affine mapping.

    ``source_metadata`` JSON contains a series identity and one metadata record
    per DICOM instance. ``conversion_record`` binds those records to the exact
    NIfTI byte hash and gives the integer signed-permutation mapping from NIfTI
    voxel indices ``[i,j,k]`` to DICOM ``[column,row,slice]`` indices.
    DICOM patient coordinates are LPS; NIfTI affine coordinates are compared
    in RAS using the standard LPS-to-RAS sign conversion.
    """
    if source_metadata_path is None or conversion_record_path is None:
        raise ProvenanceError("Actual source DICOM metadata and conversion-record files are required.")
    metadata, metadata_hash = _read_json(source_metadata_path, "source metadata")
    conversion, conversion_hash = _read_json(conversion_record_path, "conversion record")
    if metadata_hash != review.get("source_metadata_sha256"):
        raise ProvenanceError("Source metadata file hash differs from the signed review.")
    if conversion_hash != review.get("conversion_record_sha256"):
        raise ProvenanceError("Conversion record file hash differs from the signed review.")
    if metadata.get("schema") != SOURCE_SCHEMA or conversion.get("schema") != CONVERSION_SCHEMA:
        raise ProvenanceError("Source metadata or conversion record schema is unsupported.")
    if metadata.get("patient_coordinate_system") != "LPS" or conversion.get("nifti_affine_coordinate_system") != "RAS":
        raise ProvenanceError("Source/conversion coordinate systems must explicitly be DICOM LPS and NIfTI RAS.")
    if not isinstance(conversion.get("conversion_software"), str) or not conversion["conversion_software"].strip() or \
            not isinstance(conversion.get("conversion_version"), str) or not conversion["conversion_version"].strip():
        raise ProvenanceError("Conversion software name and version are required.")
    if conversion.get("conversion_software") != review.get("conversion_software") or \
            conversion.get("conversion_version") != review.get("conversion_version"):
        raise ProvenanceError("Conversion software/version differ from the signed review.")

    if nifti_path is None:
        if not isinstance(expected_nifti_sha256, str) or len(expected_nifti_sha256) != 64:
            raise ProvenanceError("A server-verified NIfTI SHA-256 is required when the volume is not present.")
        nifti_digest = expected_nifti_sha256
    else:
        nifti_digest = _file_sha256(nifti_path)
    if conversion.get("nifti_sha256") != nifti_digest or review.get("volume_sha256") != nifti_digest:
        raise ProvenanceError("Conversion record is not bound to these exact NIfTI bytes.")
    if conversion.get("source_metadata_sha256") != metadata_hash:
        raise ProvenanceError("Conversion record is not bound to the supplied source metadata.")

    series_uid = _uid(metadata.get("series_instance_uid"), "Source SeriesInstanceUID")
    frame_uid = _uid(metadata.get("frame_of_reference_uid"), "Source FrameOfReferenceUID")
    if series_uid != review.get("series_instance_uid"):
        raise ProvenanceError("Source DICOM series UID differs from the signed review.")
    if frame_uid != review.get("frame_of_reference_uid"):
        raise ProvenanceError("Source DICOM FrameOfReferenceUID differs from the signed review.")
    slices = metadata.get("slices")
    if not isinstance(slices, list) or len(slices) < 2:
        raise ProvenanceError("Source metadata must contain at least two DICOM slices.")

    seen_uids: set[str] = set()
    records: list[dict[str, Any]] = []
    for index, item in enumerate(slices):
        if not isinstance(item, dict):
            raise ProvenanceError("Source DICOM slice metadata contains an invalid record.")
        uid = _uid(item.get("sop_instance_uid"), "Source SOPInstanceUID")
        if uid in seen_uids:
            raise ProvenanceError("Source DICOM metadata contains duplicate slice instances.")
        seen_uids.add(uid)
        if item.get("series_instance_uid") != series_uid or item.get("frame_of_reference_uid") != frame_uid:
            raise ProvenanceError("Source slices contain unrelated series or FrameOfReference UIDs.")
        if item.get("modality") != "CT":
            raise ProvenanceError("Source series contains a non-CT DICOM instance.")
        position = _numbers(item.get("image_position_patient"), 3, f"Slice {index} ImagePositionPatient")
        orientation = _numbers(item.get("image_orientation_patient"), 6, f"Slice {index} ImageOrientationPatient")
        spacing = _numbers(item.get("pixel_spacing_mm"), 2, f"Slice {index} PixelSpacing")
        rows, columns = item.get("rows"), item.get("columns")
        if not isinstance(rows, int) or rows <= 0 or not isinstance(columns, int) or columns <= 0:
            raise ProvenanceError("Source DICOM rows and columns must be positive integers.")
        if min(spacing) <= 0:
            raise ProvenanceError("Source DICOM PixelSpacing values must be positive.")
        if not isinstance(item.get("patient_position"), str) or not item["patient_position"].strip():
            raise ProvenanceError("Source DICOM PatientPosition is required for review.")
        records.append({"uid": uid, "position": position, "orientation": orientation,
                        "spacing": spacing, "rows": rows, "columns": columns,
                        "spacing_between_slices_mm": item.get("spacing_between_slices_mm")})

    reference = records[0]
    orientation = reference["orientation"]
    spacing = reference["spacing"]
    patient_position = slices[0]["patient_position"]
    for record in records[1:]:
        if any(not _close(a, b, abs_tol=1e-4, rel_tol=0) for a, b in zip(record["orientation"], orientation)):
            raise ProvenanceError("DICOM ImageOrientationPatient is inconsistent across slices.")
        if record["spacing"] != spacing or record["rows"] != reference["rows"] or record["columns"] != reference["columns"]:
            raise ProvenanceError("DICOM slice dimensions or PixelSpacing are inconsistent.")
    if any(item["patient_position"] != patient_position for item in slices):
        raise ProvenanceError("DICOM PatientPosition is inconsistent across source slices.")

    row_cosine, column_cosine = orientation[:3], orientation[3:]
    row_norm = math.sqrt(_dot(row_cosine, row_cosine))
    column_norm = math.sqrt(_dot(column_cosine, column_cosine))
    normal = [row_cosine[1] * column_cosine[2] - row_cosine[2] * column_cosine[1],
              row_cosine[2] * column_cosine[0] - row_cosine[0] * column_cosine[2],
              row_cosine[0] * column_cosine[1] - row_cosine[1] * column_cosine[0]]
    if abs(row_norm - 1) > 0.002 or abs(column_norm - 1) > 0.002 or abs(_dot(row_cosine, column_cosine)) > 0.002:
        raise ProvenanceError("DICOM direction cosines are not orthonormal.")
    ordered = sorted(records, key=lambda record: _dot(record["position"], normal))
    projections = [_dot(record["position"], normal) for record in ordered]
    steps = [projections[i + 1] - projections[i] for i in range(len(projections) - 1)]
    if any(step <= 1e-3 for step in steps):
        raise ProvenanceError("DICOM slices have duplicate or non-increasing physical positions.")
    nominal_values = [record["spacing_between_slices_mm"] for record in ordered
                      if record["spacing_between_slices_mm"] is not None]
    if not nominal_values:
        raise ProvenanceError("DICOM source does not establish nominal slice spacing; missing slices cannot be checked.")
    try:
        nominal = [float(value) for value in nominal_values]
    except (TypeError, ValueError):
        raise ProvenanceError("DICOM SpacingBetweenSlices metadata is invalid.") from None
    if not all(math.isfinite(value) and value > 0 for value in nominal) or \
            any(not _close(value, nominal[0], abs_tol=1e-3, rel_tol=1e-4) for value in nominal):
        raise ProvenanceError("DICOM SpacingBetweenSlices is inconsistent.")
    if any(not _close(step, nominal[0], abs_tol=1e-3, rel_tol=1e-4) for step in steps):
        raise ProvenanceError("DICOM slice ordering has a gap or inconsistent spacing.")
    step_vectors = [[ordered[i + 1]["position"][axis] - ordered[i]["position"][axis]
                     for axis in range(3)] for i in range(len(ordered) - 1)]
    mean_step = [sum(vector[axis] for vector in step_vectors) / len(step_vectors) for axis in range(3)]
    if any(any(not _close(vector[axis], mean_step[axis], abs_tol=1e-3, rel_tol=1e-4)
               for axis in range(3)) for vector in step_vectors):
        raise ProvenanceError("DICOM slice positions do not form a consistent affine stack.")

    expected_uids = [record["uid"] for record in ordered]
    if conversion.get("source_sop_instance_uids_by_slice_index") != expected_uids:
        raise ProvenanceError("Conversion record slice inventory/order does not match the source DICOM series.")
    review_orientation = _numbers(review.get("image_orientation_patient"), 6, "Review ImageOrientationPatient")
    if any(not _close(a, b, abs_tol=1e-4, rel_tol=0) for a, b in zip(orientation, review_orientation)):
        raise ProvenanceError("Signed review ImageOrientationPatient differs from source DICOM metadata.")
    first_position = _numbers(review.get("first_image_position_patient"), 3, "Review first ImagePositionPatient")
    last_position = _numbers(review.get("last_image_position_patient"), 3, "Review last ImagePositionPatient")
    low, high = ordered[0]["position"], ordered[-1]["position"]
    direction = "ascending" if _distance(first_position, low) <= 1e-3 and _distance(last_position, high) <= 1e-3 else \
        "descending" if _distance(first_position, high) <= 1e-3 and _distance(last_position, low) <= 1e-3 else None
    if direction is None or review.get("slice_order") != direction:
        raise ProvenanceError("Review first/last slice positions or direction do not match source DICOM.")

    shape = geometry.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 3 or any(not isinstance(d, int) or d <= 0 for d in shape):
        raise ProvenanceError("Inspected NIfTI geometry has invalid dimensions.")
    if conversion.get("nifti_shape") != list(shape):
        raise ProvenanceError("Conversion record dimensions differ from the inspected NIfTI.")
    source_shape = [reference["columns"], reference["rows"], len(ordered)]
    index_map = _validate_index_map(conversion.get("nifti_voxel_to_source_indices"), shape, source_shape)

    # DICOM [column,row,slice] index -> patient LPS coordinates.
    step_vector = [(ordered[-1]["position"][axis] - ordered[0]["position"][axis]) / (len(ordered) - 1)
                   for axis in range(3)]
    dicom_to_lps = [
        [row_cosine[r] * spacing[1], column_cosine[r] * spacing[0], step_vector[r], ordered[0]["position"][r]]
        for r in range(3)
    ] + [[0.0, 0.0, 0.0, 1.0]]
    expected_ras = [[-dicom_to_lps[0][c] for c in range(4)],
                    [-dicom_to_lps[1][c] for c in range(4)],
                    dicom_to_lps[2][:], [0.0, 0.0, 0.0, 1.0]]
    expected_affine = [[sum(expected_ras[r][k] * index_map[k][c] for k in range(4)) for c in range(4)]
                       for r in range(4)]
    affine = geometry.get("affine")
    if not isinstance(affine, list) or len(affine) != 4:
        raise ProvenanceError("Inspected NIfTI affine is malformed.")
    actual_affine = [_numbers(row, 4, "Inspected NIfTI affine row") for row in affine]
    if any(not _close(actual_affine[r][c], expected_affine[r][c], abs_tol=1e-3, rel_tol=1e-5)
           for r in range(4) for c in range(4)):
        raise ProvenanceError("DICOM patient-coordinate geometry disagrees with the NIfTI affine/conversion mapping.")
