"""Compare saved binary spleen masks with held-out reference NIfTI masks.

This offline evaluator does not run the model or establish dataset authorization,
orientation truth, or clinical validity. It requires a pre-separated manifest
with train, validation, calibration, and held-out test rows.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


REQUIRED_COLUMNS = {
    "case_id",
    "patient_id",
    "split",
    "prediction_path",
    "reference_path",
    "dataset",
    "reference_source",
    "label_definition",
}
VALID_SPLITS = {"train", "validation", "calibration", "test"}


def _checked_manifest(manifest: Iterable[Mapping[str, object]]) -> list[dict[str, str]]:
    rows = [dict(row) for row in manifest]
    if not rows:
        raise ValueError("Manifest has no rows.")
    missing = REQUIRED_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"Manifest is missing required columns: {', '.join(sorted(missing))}")
    checked = []
    for row in rows:
        normalized = {}
        for column in REQUIRED_COLUMNS:
            value = row.get(column)
            if value is None or not str(value).strip():
                raise ValueError(f"Manifest column {column!r} must be non-empty for every row.")
            normalized[column] = str(value).strip()
        normalized["split"] = normalized["split"].lower()
        checked.append(normalized)
    invalid = sorted({row["split"] for row in checked} - VALID_SPLITS)
    if invalid:
        raise ValueError(f"Unsupported split value(s): {', '.join(invalid)}")
    missing_splits = VALID_SPLITS - {row["split"] for row in checked}
    if missing_splits:
        raise ValueError(f"Manifest is missing split(s): {', '.join(sorted(missing_splits))}")
    if len({row["case_id"] for row in checked}) != len(checked):
        raise ValueError("case_id must identify a unique case in the manifest.")
    patient_splits: dict[str, set[str]] = {}
    for row in checked:
        patient_splits.setdefault(row["patient_id"], set()).add(row["split"])
    if any(len(splits) > 1 for splits in patient_splits.values()):
        raise ValueError("Patient-level leakage: patient_id occurs across multiple splits.")
    for column in ("prediction_path", "reference_path"):
        normalized = [Path(row[column]).as_posix().casefold() for row in checked]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"Manifest reuses a {column} across cases or splits.")
    return checked


def _identity(path_value: str, root: Path, case_id: str, kind: str) -> Path:
    supplied = Path(path_value)
    path = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    resolved_root = root.resolve()
    if not supplied.is_absolute() and path != resolved_root and resolved_root not in path.parents:
        raise ValueError(f"{kind} path for case {case_id!r} escapes --image-root.")
    return path


def _resolve(path_value: str, root: Path, case_id: str, kind: str) -> Path:
    path = _identity(path_value, root, case_id, kind)
    if not path.is_file():
        raise FileNotFoundError(f"{kind} for case {case_id!r} does not exist: {path}")
    return path


def _load_binary_mask(path: Path, case_id: str, kind: str):
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("NIfTI evaluation requires nibabel in the selected Python environment.") from exc
    image = nib.load(str(path))
    array = np.asarray(image.dataobj)
    if array.ndim != 3 or not np.isfinite(array).all() or not np.isin(array, (0, 1)).all():
        raise ValueError(f"{kind} mask for case {case_id!r} must be a finite 3D binary NIfTI.")
    return image, array.astype(bool, copy=False)


def _case_metrics(predicted: np.ndarray, reference: np.ndarray) -> dict:
    predicted_count = int(predicted.sum())
    reference_count = int(reference.sum())
    intersection = int(np.logical_and(predicted, reference).sum())
    union = int(np.logical_or(predicted, reference).sum())
    dice = 1.0 if predicted_count + reference_count == 0 else 2 * intersection / (predicted_count + reference_count)
    iou = 1.0 if union == 0 else intersection / union
    return {
        "dice": float(dice),
        "intersection_over_union": float(iou),
        "sensitivity": float(intersection / reference_count) if reference_count else None,
        "precision": float(intersection / predicted_count) if predicted_count else None,
        "predicted_foreground_voxels": predicted_count,
        "reference_foreground_voxels": reference_count,
        "false_positive_voxels": int(np.logical_and(predicted, ~reference).sum()),
        "false_negative_voxels": int(np.logical_and(~predicted, reference).sum()),
    }


def _bootstrap_mean_interval(values: list[float], seed: int = 0) -> list[float] | None:
    if not values:
        return None
    rng = np.random.default_rng(seed)
    samples = rng.choice(np.asarray(values, dtype=np.float64), size=(2000, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.percentile(samples, [2.5, 97.5])]


def evaluate_manifest(
    manifest: Iterable[Mapping[str, object]],
    image_root: Path = Path("."),
    *,
    model_id: str = "MONAI pretrained spleen segmentation",
    checkpoint_training_overlap: str = "Not assessed; external review is required.",
) -> dict:
    """Compute held-out per-volume segmentation metrics without running inference."""
    checked = _checked_manifest(manifest)
    root = Path(image_root)
    seen_paths: dict[str, tuple[str, str]] = {}
    for row in checked:
        for column in ("prediction_path", "reference_path"):
            identity = str(_identity(row[column], root, row["case_id"], column)).casefold()
            if identity in seen_paths:
                prior_case, prior_column = seen_paths[identity]
                raise ValueError(
                    f"Mask path is reused by case {prior_case!r} ({prior_column}) and "
                    f"case {row['case_id']!r} ({column})."
                )
            seen_paths[identity] = (row["case_id"], column)
    test_rows = [row for row in checked if row["split"] == "test"]
    cases = []
    for row in test_rows:
        case_id = row["case_id"]
        prediction_path = _resolve(row["prediction_path"], root, case_id, "Prediction")
        reference_path = _resolve(row["reference_path"], root, case_id, "Reference")
        prediction_image, predicted = _load_binary_mask(prediction_path, case_id, "Prediction")
        reference_image, reference = _load_binary_mask(reference_path, case_id, "Reference")
        if predicted.shape != reference.shape:
            raise ValueError(f"Prediction/reference shape mismatch for case {case_id!r}.")
        if not np.allclose(prediction_image.affine, reference_image.affine, atol=1e-5, rtol=1e-5):
            raise ValueError(f"Prediction/reference affine mismatch for case {case_id!r}.")
        metrics = _case_metrics(predicted, reference)
        cases.append({
            "case_id": case_id,
            "patient_id": row["patient_id"],
            "dataset": row["dataset"],
            "reference_source": row["reference_source"],
            "label_definition": row["label_definition"],
            **metrics,
        })
    metric_names = (
        "dice",
        "intersection_over_union",
        "sensitivity",
        "precision",
    )
    aggregates = {}
    for metric in metric_names:
        values = [case[metric] for case in cases if case[metric] is not None]
        aggregates[metric] = {
            "n": len(values),
            "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "bootstrap_95_percentile_interval": _bootstrap_mean_interval(values),
        }
    return {
        "model": model_id,
        "task": "Binary spleen segmentation in 3D CT NIfTI volumes; foreground label 1.",
        "evaluation_status": "Research metrics only; authorization, annotation validity, checkpoint independence, and clinical validity are not established by this evaluator.",
        "checkpoint_training_overlap": checkpoint_training_overlap,
        "preprocessing": "No preprocessing or inference performed; saved predicted and reference masks are compared in their stored voxel geometry.",
        "geometry_check": "Prediction/reference shapes must match and NIfTI affines must agree within atol=rtol=1e-5.",
        "sample_counts": {
            split: sum(row["split"] == split for row in checked)
            for split in ("train", "validation", "calibration", "test")
        },
        "test_cases_evaluated": len(cases),
        "patient_level_separation_checked": True,
        "metrics": aggregates,
        "cases": cases,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="CSV describing group-separated prediction/reference mask pairs")
    parser.add_argument("--image-root", type=Path, default=Path("."), help="Root for relative NIfTI paths")
    parser.add_argument("--output", type=Path, default=Path("evaluation_results/spleen_segmentation.json"))
    parser.add_argument("--model-id", default="MONAI pretrained spleen segmentation")
    parser.add_argument("--checkpoint-training-overlap", default="Not assessed; external review is required.")
    args = parser.parse_args(argv)
    if not args.manifest.is_file():
        parser.error(f"Manifest does not exist: {args.manifest}")
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    report = evaluate_manifest(
        manifest,
        args.image_root,
        model_id=args.model_id,
        checkpoint_training_overlap=args.checkpoint_training_overlap,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Held-out CT volumes evaluated: {report['test_cases_evaluated']}")
    for metric, values in report["metrics"].items():
        print(f"{metric}: mean={values['mean']} (n={values['n']})")
    print(f"Performance status: {report['evaluation_status']}")
    print(f"JSON results: {args.output}")
    return report


if __name__ == "__main__":
    main()
