#!/usr/bin/env python
"""Convert a TCIA DICOM series into staged PNG images and manifest rows."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from PIL import Image
try:
    import pydicom
except ImportError as exc:
    raise RuntimeError("pydicom is required for TCIA DICOM conversion.") from exc
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MEDICAL_MANIFEST_COLUMNS = ["image_id","local_path","group_id","category","source","source_title","license","attribution","license_url","source_url","landing_url","split","grouping_verified"]
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "unknown"
def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None

def _load_series(input_dir: Path) -> list[tuple[Path, Any]]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"DICOM input directory does not exist: {input_dir}")
    datasets = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            dataset = pydicom.dcmread(str(path), force=False)
        except Exception:
            continue
        if not getattr(dataset, "PixelData", None):
            continue
        if getattr(dataset, "Modality", None) not in {"CT", "MR"}:
            raise ValueError(f"{path} is not a CT or MR DICOM instance (Modality={getattr(dataset, 'Modality', None)!r}).")
        datasets.append((path, dataset))
    if not datasets:
        raise ValueError("No readable CT/MR DICOM instances with pixel data were found.")
    series_uids = {str(getattr(ds, "SeriesInstanceUID", "")).strip() for _, ds in datasets}
    if len(series_uids) != 1 or not next(iter(series_uids)):
        raise ValueError("Input directory must contain exactly one DICOM series.")
    return datasets

def _slice_sort_key(item: tuple[Path, Any]) -> tuple[float, str]:
    _path, dataset = item
    instance_number = getattr(dataset, "InstanceNumber", None)
    if instance_number is not None:
        try:
            return (float(instance_number), str(getattr(dataset, "SOPInstanceUID", "")))
        except (TypeError, ValueError):
            pass
    slice_location = getattr(dataset, "SliceLocation", None)
    if slice_location is not None:
        value = _as_float(slice_location)
        if value is not None:
            return (value, str(getattr(dataset, "SOPInstanceUID", "")))
    position = getattr(dataset, "ImagePositionPatient", None)
    if position is not None:
        try:
            return (float(position[-1]), str(getattr(dataset, "SOPInstanceUID", "")))
        except (TypeError, ValueError, IndexError):
            pass
    return (0.0, str(getattr(dataset, "SOPInstanceUID", "")))
def _convert_pixel_array(dataset: Any) -> np.ndarray:
    array = np.asarray(dataset.pixel_array)
    if array.ndim != 2:
        raise ValueError(f"DICOM instance is not a single 2D slice (shape={array.shape!r}).")
    slope = getattr(dataset, "RescaleSlope", None)
    intercept = getattr(dataset, "RescaleIntercept", None)
    if slope is not None or intercept is not None:
        try:
            slope_value = float(slope if slope is not None else 1.0)
            intercept_value = float(intercept if intercept is not None else 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid RescaleSlope/RescaleIntercept metadata.") from exc
        if not (math.isfinite(slope_value) and math.isfinite(intercept_value)):
            raise ValueError("Non-finite RescaleSlope/RescaleIntercept metadata.")
        array = array.astype(np.float32) * slope_value + intercept_value
    else:
        array = array.astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError("DICOM pixel data contains non-finite values.")
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum <= minimum:
        scaled = np.zeros_like(array, dtype=np.uint8)
    else:
        scaled = np.rint((array - minimum) * (255.0 / (maximum - minimum))).clip(0, 255)
    return scaled.astype(np.uint8)

def convert_series(input_dir: Path, staging_dir: Path, category: str, source: str, source_title: str, license_name: str, attribution: str, license_url: str, source_url: str, landing_url: str, split: str | None, grouping_verified_override: bool | None, metadata_output: Path | None) -> pd.DataFrame:
    datasets = _load_series(input_dir)
    datasets.sort(key=_slice_sort_key)
    first = datasets[0][1]
    patient_id = str(getattr(first, "PatientID", "")).strip()
    study_uid = str(getattr(first, "StudyInstanceUID", "")).strip()
    series_uid = str(getattr(first, "SeriesInstanceUID", "")).strip()
    modality = str(getattr(first, "Modality", "")).strip()
    if grouping_verified_override is None:
        grouping_verified = bool(patient_id)
    else:
        grouping_verified = grouping_verified_override
    if patient_id:
        group_id = f"tcia-subject:{patient_id}"
    elif study_uid:
        group_id = f"tcia-study:{study_uid}"
    else:
        raise ValueError("DICOM series has neither PatientID nor StudyInstanceUID.")
    staging_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seen_hashes = set()
    metadata_records = []
    for index, (path, dataset) in enumerate(datasets, start=1):
        instance_uid = str(getattr(dataset, "SOPInstanceUID", "")).strip()
        if not instance_uid:
            raise ValueError(f"DICOM instance {index} is missing SOPInstanceUID.")
        image_id = f"{source}-{_safe_component(patient_id or study_uid)}-{_safe_component(series_uid)}-{index:04d}"
        filename = f"{image_id}.png"
        target = staging_dir / filename
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing staged file: {target}")
        pixels = _convert_pixel_array(dataset)
        image = Image.fromarray(pixels, mode="L")
        image.save(target, format="PNG")
        image.verify()
        content_hash = _sha256_file(target)
        if content_hash in seen_hashes:
            target.unlink(missing_ok=True)
            continue
        seen_hashes.add(content_hash)
        image.close()
        row = {"image_id": image_id, "local_path": str(target.resolve()), "group_id": group_id, "category": category, "source": source, "source_title": source_title, "license": license_name, "attribution": attribution, "license_url": license_url, "source_url": source_url, "landing_url": landing_url}
        if split:
            row["split"] = split
    row["grouping_verified"] = "true" if grouping_verified else "false"
    rows.append(row)
    metadata_records.append({"image_id": image_id})
    if metadata_output is not None:
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.write_text(json.dumps(metadata_records, indent=2), encoding="utf-8")
    frame = pd.DataFrame(rows, columns=MEDICAL_MANIFEST_COLUMNS)
    return frame

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing one DICOM series.")
    parser.add_argument("--staging-dir", type=Path, required=True, help="Medical staging directory.")
    parser.add_argument("--output-manifest", type=Path, required=True, help="CSV file to append/create with manifest rows.")
    parser.add_argument("--category", choices=("ct", "mri", "ultrasound", "non_chest_xray"), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-title", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--attribution", required=True)
    parser.add_argument("--license-url", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--landing-url", required=True)
    parser.add_argument("--split", choices=("train", "validation", "calibration", "test"))
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--grouping-verified", action="store_true", help="Mark grouping as verified when source metadata supports it.")
    parser.add_argument("--append", action="store_true", help="Append rows to an existing manifest instead of failing if it exists.")
    args = parser.parse_args()
    frame = convert_series(input_dir=args.input_dir, staging_dir=args.staging_dir, category=args.category, source=args.source, source_title=args.source_title, license_name=args.license, attribution=args.attribution, license_url=args.license_url, source_url=args.source_url, landing_url=args.landing_url, split=args.split, grouping_verified_override=True if args.grouping_verified else None, metadata_output=args.metadata_output)
    if args.output_manifest.exists() and not args.append:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {args.output_manifest}")
    if args.append and args.output_manifest.exists():
        existing = pd.read_csv(args.output_manifest, dtype=str, keep_default_na=False)
        frame = pd.concat([existing, frame], ignore_index=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_manifest, index=False, quoting=1)
    print(f"Staged {len(frame)} images and wrote {args.output_manifest}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
