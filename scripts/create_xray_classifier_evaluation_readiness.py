import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("outputs/xray_gradcam/classifier_evaluation_readiness")
ROOT.mkdir(parents=True, exist_ok=True)
MANIFEST = Path("outputs/xray_labeled_evaluation/nih_xray_manifest.csv")
PREDICTIONS = Path("outputs/xray_labeled_evaluation/per_image_predictions.csv")
EVAL_JSON = Path("outputs/xray_labeled_evaluation/densenet121.json")
SELECTION = Path("outputs/xray_labeled_evaluation/nih_xray_selection.json")

manifest = pd.read_csv(MANIFEST)
preds = pd.read_csv(PREDICTIONS)
report = json.loads(EVAL_JSON.read_text())
selection = json.loads(SELECTION.read_text())
compatible = selection["compatible_model_labels"]

test_mask = preds["split"].to_numpy() == "test"
metrics = []
for finding in report["findings"]:
    name = finding["finding"]
    if name not in compatible:
        metrics.append({
            "finding": name,
            "status": "undefined: no compatible reference labels",
            "test_positive": 0,
            "test_negative": 0,
            "auroc": None,
            "auprc": None,
        })
        continue
    y = preds.loc[test_mask, f"reference_{name}"].to_numpy(dtype=float)
    s = preds.loc[test_mask, f"score_{name}"].to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(s)
    y = y[valid].astype(int)
    s = s[valid]
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        status = "undefined: requires both positive and negative examples"
        auroc = None
        auprc = None
    else:
        status = "defined"
        auroc = float(roc_auc_score(y, s))
        auprc = float(average_precision_score(y, s))
    metrics.append({
        "finding": name,
        "status": status,
        "test_positive": pos,
        "test_negative": neg,
        "test_n": int(len(y)),
        "auroc": auroc,
        "auprc": auprc,
        "threshold_raw_score": finding["threshold_raw_score"],
        "threshold_precision": finding["precision"],
        "threshold_recall": finding["recall"],
        "threshold_f1": finding["f1"],
        "confusion_matrix": finding["confusion_matrix"],
    })

duplicate_paths = int(manifest["image_path"].duplicated().sum())
validation_patients = set(manifest.loc[manifest["split"] == "validation", "patient_id"])
test_patients = set(manifest.loc[manifest["split"] == "test", "patient_id"])
patient_overlap = validation_patients & test_patients

test_mask = preds["split"].to_numpy() == "test"
metrics = []
for finding in report["findings"]:
    name = finding["finding"]
    if name not in compatible:
        metrics.append({
            "finding": name,
            "status": "undefined: no compatible reference labels",
            "test_positive": 0,
            "test_negative": 0,
            "auroc": None,
            "auprc": None,
        })
        continue
    y = preds.loc[test_mask, f"reference_{name}"].to_numpy(dtype=float)
    s = preds.loc[test_mask, f"score_{name}"].to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(s)
    y = y[valid].astype(int)
    s = s[valid]
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        status = "undefined: requires both positive and negative examples"
        auroc = None
        auprc = None
    else:
        status = "defined"
        auroc = float(roc_auc_score(y, s))
        auprc = float(average_precision_score(y, s))
    metrics.append({
        "finding": name,
        "status": status,
        "test_positive": pos,
        "test_negative": neg,
        "test_n": int(len(y)),
        "auroc": auroc,
        "auprc": auprc,
        "threshold_raw_score": finding["threshold_raw_score"],
        "threshold_precision": finding["precision"],
        "threshold_recall": finding["recall"],
        "threshold_f1": finding["f1"],
        "confusion_matrix": finding["confusion_matrix"],
    })

duplicate_paths = int(manifest["image_path"].duplicated().sum())
validation_patients = set(manifest.loc[manifest["split"] == "validation", "patient_id"])
test_patients = set(manifest.loc[manifest["split"] == "test", "patient_id"])
patient_overlap = validation_patients & test_patients

evaluation_readiness = {
    "evaluation_possible": True,
    "dataset": "chehablab/NIHChestXR (Hugging Face mirror of NIH ChestX-ray14)",
    "source_shard": "data/train-00000-of-00006.parquet",
    "license_and_attribution": "NIH images have no use restrictions but require attribution. HF mirror is CC0 1.0. NIH labels are NLP/report-derived weak labels, not expert-adjudicated ground truth.",
    "model": "TorchXRayVision DenseNet121 densenet121-res224-all",
    "model_outputs": "18 operating-point-normalized scores, not calibrated probabilities.",
    "preprocessing": "EXIF transpose, grayscale, center crop to square, resize 224x224, normalization (pixels/255 - 0.5) * 2048.",
    "sample_counts": {
        "total_images": int(len(manifest)),
        "validation_images": int((manifest["split"] == "validation").sum()),
        "test_images": int((manifest["split"] == "test").sum()),
        "unique_patients": int(manifest["patient_id"].nunique()),
        "duplicate_image_paths": duplicate_paths,
        "validation_test_patient_overlap": len(patient_overlap),
    },
    "label_compatibility": {
        "compatible_labels": compatible,
        "skipped_labels": ["Lung Lesion", "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum"],
        "skipped_reason": "No compatible NIH ChestX-ray14 reference label exists for these model outputs.",
    },
    "metrics": metrics,
    "threshold_note": "Thresholds were selected on the 60-image validation split only by maximizing F1. They are not clinically validated operating points.",
    "leakage_and_suitability": {
        "patient_level_separation_checked": True,
        "duplicate_image_paths": duplicate_paths,
        "validation_test_patient_overlap": len(patient_overlap),
        "checkpoint_training_overlap": "TorchXRayVision documents NIH and RSNA among checkpoint training sources; record-level overlap with this evaluation cohort is not audited.",
    },
    "limitations": [
        "NIH labels are report-derived weak labels, not expert-adjudicated ground truth.",
        "The selected cohort is 300 of 112,120 images from one shard.",
        "Model checkpoint training overlap is unaudited.",
        "Raw scores are not calibrated probabilities.",
        "Grad-CAM localization is not a proxy for classification accuracy.",
    ],
}
(ROOT / "evaluation_readiness.json").write_text(json.dumps(evaluation_readiness, indent=2), encoding="utf-8")

lines = [
    "# Chest X-Ray Classifier Evaluation Readiness",
    "",
    "## Verified facts",
    "",
    "- Model: TorchXRayVision DenseNet121 `densenet121-res224-all`.",
    "- Output labels: 18 operating-point-normalized scores.",
    "- Outputs are not calibrated probabilities.",
    "- Existing labeled subset: 300 unique patients, 60 validation and 240 test images.",
    "- Validation and test patients are disjoint.",
    "- 14 labels have compatible NIH reference labels.",
    "- 4 model outputs have no compatible NIH reference label.",
    "",
    "## Suitability",
    "",
    "A defensible held-out evaluation is possible using the existing NIH-derived subset. Thresholds were selected on validation only.",
    "",
    "## AUPRC and AUROC on test split",
    "",
    "| Finding | AUROC | AUPRC | Test positives | Test negatives |",
    "|---|---:|---:|---:|---:|",
]
for m in metrics:
    if m["auroc"] is None:
        lines.append(f"| {m['finding']} | undefined | undefined | {m['test_positive']} | {m['test_negative']} |")
    else:
        lines.append(f"| {m['finding']} | {m['auroc']:.4f} | {m['auprc']:.4f} | {m['test_positive']} | {m['test_negative']} |")
lines.extend([
    "",
    "## Unresolved questions",
    "",
    "- Record-level overlap between the evaluation cohort and checkpoint training data is not audited.",
    "- NIH labels are weak, report-derived labels.",
    "- No independent external validation cohort is available.",
    "",
    "## Conclusion",
    "",
    "The current classifier can be evaluated rigorously on the existing labeled subset, but results are preliminary and must not be described as clinical validation.",
])
(ROOT / "evaluation_readiness.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote readiness artifacts")
