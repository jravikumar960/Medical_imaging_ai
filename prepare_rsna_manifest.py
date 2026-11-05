"""Prepare an evaluation manifest from an already authorized RSNA data copy.

This script never downloads, edits, moves, or deletes source data. It maps the
RSNA Target field only to the two outputs used by TorchXRayVision's own
RSNA_Pneumonia_Dataset adapter: Pneumonia and Lung Opacity. It requires an
external split map with a genuine patient-level identifier; RSNA patientId is
an examination/image ID and must not be treated as a patient identifier.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError


MODEL_LABELS = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening",
    "Cardiomegaly", "Nodule", "Mass", "Hernia", "Lung Lesion", "Fracture",
    "Lung Opacity", "Enlarged Cardiomediastinum",
]
RSNA_TARGET_LABELS = {"Pneumonia", "Lung Opacity"}
VALID_SPLITS = {"train", "validation", "test"}
SPLIT_MAP_COLUMNS = {"patientId", "patient_id", "split"}


def prepare_manifest(
    labels_csv: Path,
    split_map_csv: Path,
    image_root: Path,
    output_csv: Path,
    image_extension: str = ".jpg",
    force: bool = False,
) -> pd.DataFrame:
    labels_csv = Path(labels_csv).resolve()
    split_map_csv = Path(split_map_csv).resolve()
    image_root = Path(image_root).resolve()
    output_csv = Path(output_csv).resolve()
    extension = image_extension.lower()
    if extension not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("image_extension must be one of .jpg, .jpeg, or .png")
    if output_csv in {labels_csv, split_map_csv}:
        raise ValueError("Output must not overwrite either input CSV.")
    if output_csv == image_root or image_root in output_csv.parents:
        raise ValueError("Output must be outside the source image directory.")
    if output_csv.exists() and not force:
        raise FileExistsError(f"Output already exists; use --force to replace it: {output_csv}")

    labels = pd.read_csv(labels_csv)
    missing_label_fields = {"patientId", "Target"} - set(labels.columns)
    if missing_label_fields:
        raise ValueError(f"RSNA labels CSV is missing columns: {', '.join(sorted(missing_label_fields))}")
    if labels.empty:
        raise ValueError("RSNA labels CSV has no records.")
    if labels["patientId"].isna().any() or labels["patientId"].astype(str).str.strip().eq("").any():
        raise ValueError("RSNA labels CSV contains an empty patientId/examination ID.")
    labels["patientId"] = labels["patientId"].astype(str).str.strip()
    if labels.duplicated().any():
        raise ValueError("RSNA labels CSV contains duplicate records.")

    targets = pd.to_numeric(labels["Target"], errors="coerce")
    invalid_targets = labels["Target"].isna() | targets.isna() | ~targets.isin([0, 1])
    if invalid_targets.any():
        bad_rows = labels.index[invalid_targets].tolist()[:5]
        raise ValueError(f"RSNA Target must contain only explicit 0 or 1 values; invalid row index(es): {bad_rows}")
    labels["Target"] = targets.astype("int8")

    # RSNA annotation CSVs repeat patientId for multiple bounding boxes. Those
    # rows become one image-level label, but conflicting image labels are errors.
    target_counts = labels.groupby("patientId")["Target"].nunique()
    conflicts = target_counts[target_counts > 1]
    if not conflicts.empty:
        raise ValueError(f"Conflicting Target labels for {len(conflicts)} RSNA examination ID(s).")
    image_labels = labels.groupby("patientId", as_index=False, sort=False)["Target"].first()

    split_map = pd.read_csv(split_map_csv)
    missing_split_fields = SPLIT_MAP_COLUMNS - set(split_map.columns)
    if missing_split_fields:
        raise ValueError(f"Split map is missing columns: {', '.join(sorted(missing_split_fields))}")
    if split_map[list(SPLIT_MAP_COLUMNS)].isna().any().any():
        raise ValueError("Split map patientId, patient_id, and split fields must be explicit.")
    for column in ("patientId", "patient_id"):
        split_map[column] = split_map[column].astype(str).str.strip()
        if split_map[column].eq("").any():
            raise ValueError(f"Split map contains an empty {column} value.")
    if split_map["patientId"].equals(split_map["patient_id"]):
        raise ValueError(
            "patient_id cannot simply copy RSNA patientId; supply an authorized crosswalk to actual patient groups."
        )
    split_map["split"] = split_map["split"].astype(str).str.strip().str.lower()
    invalid_splits = sorted(set(split_map["split"]) - VALID_SPLITS)
    if invalid_splits:
        raise ValueError(f"Unsupported split name(s): {', '.join(invalid_splits)}")
    if split_map["patientId"].duplicated().any():
        raise ValueError("Split map contains duplicate patientId/examination IDs.")
    if set(split_map["split"]) != VALID_SPLITS:
        missing_splits = sorted(VALID_SPLITS - set(split_map["split"]))
        raise ValueError(f"Split map must include train, validation, and test; missing: {', '.join(missing_splits)}")

    patient_split_counts = split_map.groupby("patient_id")["split"].nunique()
    overlap = patient_split_counts[patient_split_counts > 1]
    if not overlap.empty:
        raise ValueError(f"Patient-level leakage: {len(overlap)} patient(s) occur in multiple splits.")

    label_ids = set(image_labels["patientId"])
    split_ids = set(split_map["patientId"])
    if label_ids != split_ids:
        missing = len(label_ids - split_ids)
        extra = len(split_ids - label_ids)
        raise ValueError(f"Split map must cover labels exactly (unmapped IDs={missing}, unknown IDs={extra}).")

    records = image_labels.merge(split_map, on="patientId", how="inner", validate="one_to_one")
    image_paths = []
    for exam_id in records["patientId"]:
        path = image_root / f"{exam_id}{extension}"
        if not path.is_file():
            raise FileNotFoundError(f"Image file does not exist for RSNA examination {exam_id}: {path}")
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Unreadable image for RSNA examination {exam_id}: {path}: {exc}") from exc
        image_paths.append(path.name)

    records["image_path"] = image_paths
    records["Pneumonia"] = records["Target"].astype("int8")
    records["Lung Opacity"] = records["Target"].astype("int8")
    for label in MODEL_LABELS:
        if label not in RSNA_TARGET_LABELS:
            records[label] = pd.NA

    manifest_columns = ["image_path", "split", "patient_id", *MODEL_LABELS]
    manifest = records[manifest_columns].copy()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_csv, index=False)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path, help="Authorized RSNA stage_2_train_labels.csv")
    parser.add_argument("--split-map", required=True, type=Path, help="CSV with patientId, patient_id, split")
    parser.add_argument("--image-root", required=True, type=Path, help="Directory of already obtained JPEG/PNG images")
    parser.add_argument("--output", required=True, type=Path, help="New manifest path outside the source image directory")
    parser.add_argument("--image-extension", default=".jpg", choices=[".jpg", ".jpeg", ".png"])
    parser.add_argument("--force", action="store_true", help="Allow replacing an existing output manifest")
    args = parser.parse_args(argv)
    manifest = prepare_manifest(
        args.labels, args.split_map, args.image_root, args.output, args.image_extension, args.force
    )
    print(f"Prepared {len(manifest)} image-level manifest rows at {args.output}")
    print("Mapped RSNA Target to Pneumonia and Lung Opacity; all other pathology labels are blank.")
    print("This is label preparation only; it does not establish independence from model pretraining.")


if __name__ == "__main__":
    main()
