"""Evaluate TorchXRayVision DenseNet121 on validation and held-out test data.

Manifest format: image_path, split, optional patient_id, and one binary ground-
truth column for every pathology label emitted by the model. Valid split values
are train, validation, and test. Blank labels are unknown and excluded from
that finding's metrics. Thresholds are selected per finding on validation only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch
import torchxrayvision as xrv

from imaging import prepare_image

VALID_SPLITS = {"train", "validation", "test"}


def _validate_manifest(manifest: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    required = {"image_path", "split", *labels}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {', '.join(sorted(missing))}")
    if manifest.empty:
        raise ValueError("Manifest has no rows.")

    checked = manifest.copy()
    checked["_split"] = checked["split"].astype(str).str.strip().str.lower()
    invalid_splits = sorted(set(checked["_split"]) - VALID_SPLITS)
    if invalid_splits:
        raise ValueError(f"Unsupported split value(s): {', '.join(invalid_splits)}")
    if not (checked["_split"] == "validation").any():
        raise ValueError("Manifest must contain a validation split for threshold selection.")
    if not (checked["_split"] == "test").any():
        raise ValueError("Manifest must contain a held-out test split.")

    paths = checked["image_path"]
    missing_paths = paths.isna() | paths.astype(str).str.strip().eq("")
    if missing_paths.any():
        raise ValueError(f"Manifest has {int(missing_paths.sum())} row(s) without an explicit image_path.")

    for label in labels:
        parsed = pd.to_numeric(checked[label], errors="coerce")
        invalid = checked[label].notna() & (parsed.isna() | ~parsed.isin([0, 1]))
        if invalid.any():
            rows = checked.index[invalid].tolist()[:5]
            raise ValueError(f"Ground-truth label {label!r} must contain only 0, 1, or blank; invalid row index(es): {rows}")
        checked[label] = parsed.astype(float)

    if "patient_id" in checked:
        patients = checked["patient_id"].astype("string").str.strip().replace("", pd.NA)
        if patients.isna().any():
            raise ValueError(
                "patient_id is optional, but when supplied it must be non-empty for every manifest row."
            )
        checked["patient_id"] = patients
        split_patients = pd.DataFrame({"patient_id": patients, "split": checked["_split"]})
        cross_split = split_patients.groupby("patient_id")["split"].nunique()
        leaked = cross_split[cross_split > 1]
        if not leaked.empty:
            raise ValueError(
                f"Patient-level leakage: {len(leaked)} patient_id value(s) occur across multiple splits."
            )
    return checked


def _image_path_identity(value, image_root: Path, row_index) -> Path:
    path = Path(str(value).strip())
    if not path.is_absolute():
        root = image_root.resolve()
        path = (root / path).resolve()
        if path != root and root not in path.parents:
            raise ValueError(f"Relative image_path for manifest row {row_index} escapes --image-root.")
    else:
        path = path.resolve()
    return path


def _resolve_image_path(value, image_root: Path, row_index) -> Path:
    path = _image_path_identity(value, image_root, row_index)
    if not path.is_file():
        raise FileNotFoundError(f"Image for manifest row {row_index} does not exist: {path}")
    return path


def _predict_split(rows: pd.DataFrame, labels: list[str], model, image_root: Path):
    scores_by_image, targets_by_image = [], []
    for row_index, row in rows.iterrows():
        path = _resolve_image_path(row["image_path"], image_root, row_index)
        try:
            with Image.open(path) as image:
                tensor, _ = prepare_image(image)
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise ValueError(f"Could not read image for manifest row {row_index} ({path}): {exc}") from exc
        try:
            with torch.inference_mode():
                output = model(tensor)
        except Exception as exc:
            raise RuntimeError(f"Model inference failed for manifest row {row_index} ({path}): {exc}") from exc

        if not isinstance(output, torch.Tensor):
            raise ValueError(f"Invalid model output for row {row_index}: expected a torch.Tensor.")
        if output.ndim != 2 or tuple(output.shape) != (1, len(labels)):
            raise ValueError(
                f"Invalid model output for row {row_index}: expected shape (1, {len(labels)}), got {tuple(output.shape)}."
            )
        output_scores = output[0].detach().cpu().numpy().astype(np.float64, copy=False)
        if not np.isfinite(output_scores).all():
            raise ValueError(f"Invalid model output for row {row_index}: scores contain NaN or infinity.")
        scores_by_image.append(output_scores)
        targets_by_image.append([row[label] for label in labels])

    return np.asarray(targets_by_image, dtype=np.float64), np.asarray(scores_by_image, dtype=np.float64)


def _select_max_f1_threshold(targets: np.ndarray, scores: np.ndarray):
    """Choose a per-label threshold on validation, preferring higher tied thresholds."""
    valid = np.isfinite(targets) & np.isfinite(scores)
    y, value = targets[valid].astype(np.int8), scores[valid]
    positives, negatives = int((y == 1).sum()), int((y == 0).sum())
    if not positives or not negatives:
        return None, {
            "validation_n": int(len(y)),
            "validation_missing_labels": int(len(targets) - len(y)),
            "validation_positive": positives,
            "validation_negative": negatives,
            "threshold_status": "undefined: validation requires both classes",
        }

    candidates = np.unique(np.concatenate((value, [np.nextafter(value.max(), np.inf)])))
    best_threshold, best_f1 = None, -1.0
    for threshold in candidates:
        predicted = value >= threshold
        tp = int(np.sum(predicted & (y == 1)))
        fp = int(np.sum(predicted & (y == 0)))
        fn = int(np.sum(~predicted & (y == 1)))
        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator else 0.0
        if f1 > best_f1 or (f1 == best_f1 and (best_threshold is None or threshold > best_threshold)):
            best_f1, best_threshold = f1, float(threshold)
    return best_threshold, {
        "validation_n": int(len(y)),
        "validation_missing_labels": int(len(targets) - len(y)),
        "validation_positive": positives,
        "validation_negative": negatives,
        "validation_f1": float(best_f1),
        "threshold_status": "selected by maximum validation F1",
    }


def _finding_metrics(
    label: str,
    targets: np.ndarray,
    scores: np.ndarray,
    threshold,
    threshold_info,
    *,
    test_rows: int,
):
    valid = np.isfinite(targets) & np.isfinite(scores)
    truth = targets[valid].astype(np.int8)
    value = scores[valid]
    count_pos, count_neg = int(np.sum(truth == 1)), int(np.sum(truth == 0))
    auroc = float(roc_auc_score(truth, value)) if count_pos and count_neg else None

    result = {
        "finding": label,
        "test_n": int(len(truth)),
        "test_missing_labels": int(test_rows - len(truth)),
        "test_positive": count_pos,
        "test_negative": count_neg,
        "threshold_raw_score": threshold,
        "threshold_status": threshold_info["threshold_status"],
        "validation_n": threshold_info["validation_n"],
        "validation_missing_labels": threshold_info["validation_missing_labels"],
        "validation_positive": threshold_info["validation_positive"],
        "validation_negative": threshold_info["validation_negative"],
        "validation_f1": threshold_info.get("validation_f1"),
        "auroc": auroc,
        "sensitivity": None,
        "specificity": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "f1_status": "defined",
        "confusion_matrix": None,
    }
    if threshold is None:
        result["f1_status"] = "undefined: validation threshold unavailable"
        return result
    if not len(truth):
        result["f1_status"] = "undefined: no labeled test samples"
        return result

    predicted = value >= threshold
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(truth, predicted, labels=[0, 1]).ravel())
    result["confusion_matrix"] = [[tn, fp], [fn, tp]]
    result["sensitivity"] = float(tp / (tp + fn)) if tp + fn else None
    result["specificity"] = float(tn / (tn + fp)) if tn + fp else None
    result["precision"] = float(tp / (tp + fp)) if tp + fp else None
    result["recall"] = float(tp / (tp + fn)) if tp + fn else None
    f1_denominator = 2 * tp + fp + fn
    result["f1"] = float(2 * tp / f1_denominator) if f1_denominator else None
    result["f1_status"] = "defined" if f1_denominator else "undefined: no positive truth or predicted cases"
    return result


def evaluate_manifest(
    manifest: pd.DataFrame,
    model,
    image_root: Path = Path("."),
    *,
    model_id: str | None = None,
    checkpoint_training_overlap: str = "Not assessed; dataset/checkpoint provenance needs external review.",
):
    """Evaluate only validation and test rows; select thresholds only on validation."""
    labels = list(model.pathologies)
    if (
        not labels
        or any(not isinstance(label, str) or not label.strip() for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("Model pathologies must be non-empty, unique string output labels.")
    checked = _validate_manifest(manifest, labels)
    validation = checked.loc[checked["_split"] == "validation"]
    test = checked.loc[checked["_split"] == "test"]
    image_root = Path(image_root)

    # Check identity across every split, including train, without opening or
    # inferring training files. A reused image in train and test is leakage even
    # when the patient ID column is absent or unreliable.
    seen_paths: dict[Path, tuple[object, pd.Series]] = {}
    for row_index, row in checked.iterrows():
        if row["_split"] != "train":
            _resolve_image_path(row["image_path"], image_root, row_index)
        identity = _image_path_identity(row["image_path"], image_root, row_index)
        if identity in seen_paths:
            prior_index, prior = seen_paths[identity]
            if any(not (
                (pd.isna(prior[label]) and pd.isna(row[label])) or prior[label] == row[label]
            ) for label in labels):
                raise ValueError(
                    f"Conflicting labels for duplicate image_path in rows {prior_index} "
                    f"({prior['_split']}) and {row_index} ({row['_split']})."
                )
            raise ValueError(
                f"Duplicate image_path across manifest rows {prior_index} and {row_index} "
                f"(splits {prior['_split']!r} and {row['_split']!r})."
            )
        seen_paths[identity] = (row_index, row)

    validation_targets, validation_scores = _predict_split(validation, labels, model, image_root)
    test_targets, test_scores = _predict_split(test, labels, model, image_root)

    findings = []
    for index, label in enumerate(labels):
        threshold, threshold_info = _select_max_f1_threshold(
            validation_targets[:, index], validation_scores[:, index]
        )
        findings.append(_finding_metrics(
            label,
            test_targets[:, index],
            test_scores[:, index],
            threshold,
            threshold_info,
            test_rows=len(test),
        ))

    return {
        "model": model_id or f"{type(model).__module__}.{type(model).__qualname__}",
        "output_labels": labels,
        "manifest_label_compatibility": "Required label columns match the model.pathologies label set; values are processed in model output order. Clinical label semantics require external review.",
        "checkpoint_training_overlap": checkpoint_training_overlap,
        "performance_claim_status": "Not established by the evaluator; labeling validity, patient provenance, and checkpoint independence require external evidence.",
        "score_interpretation": "Raw model scores; not calibrated probabilities.",
        "threshold_method": "Per-finding threshold maximizing F1 on validation split only; ties select the higher threshold.",
        "patient_level_separation_checked": bool(
            "patient_id" in checked.columns and checked["patient_id"].notna().any()
        ),
        "validation_images": int(len(validation)),
        "test_images": int(len(test)),
        "training_images_excluded": int((checked["_split"] == "train").sum()),
        "training_rows_evaluated": 0,
        "findings": findings,
    }


def write_results(report: dict, output_prefix: Path):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    with json_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    csv_findings = []
    for finding in report["findings"]:
        row = {key: value for key, value in finding.items() if key != "confusion_matrix"}
        matrix = finding["confusion_matrix"]
        if matrix is None:
            row.update({"tn": None, "fp": None, "fn": None, "tp": None})
        else:
            row.update({"tn": matrix[0][0], "fp": matrix[0][1], "fn": matrix[1][0], "tp": matrix[1][1]})
        csv_findings.append(row)
    pd.DataFrame(csv_findings).to_csv(csv_path, index=False)
    return json_path, csv_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="CSV with image_path, split, and actual model label columns")
    parser.add_argument("--image-root", type=Path, default=Path("."), help="Root for relative image_path entries")
    parser.add_argument("--output-prefix", type=Path, default=Path("evaluation_results/densenet121"), help="Prefix for output .json and .csv files")
    args = parser.parse_args(argv)
    if not args.manifest.is_file():
        parser.error(f"Manifest does not exist: {args.manifest}")

    manifest = pd.read_csv(args.manifest)
    model = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
    report = evaluate_manifest(
        manifest,
        model,
        args.image_root,
        model_id="TorchXRayVision DenseNet121 densenet121-res224-all",
        checkpoint_training_overlap=(
            "TorchXRayVision documentation lists NIH and RSNA among this checkpoint's training sources; "
            "record-level overlap with the evaluation cohort is not audited here."
        ),
    )
    json_path, csv_path = write_results(report, args.output_prefix)
    print(f"Validation images: {report['validation_images']}")
    print(f"Held-out test images: {report['test_images']}")
    print(f"Training images excluded: {report['training_images_excluded']}")
    print(f"Training rows evaluated: {report['training_rows_evaluated']}")
    print(f"Checkpoint/data overlap: {report['checkpoint_training_overlap']}")
    print(f"Performance claims: {report['performance_claim_status']}")
    print(f"Raw model scores; thresholds selected using validation split only.")
    print(f"JSON results: {json_path}")
    print(f"CSV results: {csv_path}")
    return report


if __name__ == "__main__":
    main()
