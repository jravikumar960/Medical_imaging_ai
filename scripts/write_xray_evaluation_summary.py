import csv
import json
from pathlib import Path

ROOT = Path("c:/Users/jravi/Downloads/medical_imaging_ai")
OUT = ROOT / "outputs/xray_labeled_evaluation"
report = json.loads((OUT / "densenet121.json").read_text(encoding="utf-8"))
selection = json.loads((OUT / "nih_xray_selection.json").read_text(encoding="utf-8"))

with (OUT / "per_image_predictions.csv").open(newline="", encoding="utf-8") as handle:
    per_image_rows = list(csv.DictReader(handle))

summary = {
    "task": "Labeled chest X-ray evaluation of TorchXRayVision DenseNet121",
    "evaluation_directory": str(OUT),
    "source_dataset": selection["source_dataset"],
    "source_url": selection["source_url"],
    "official_nih_download_site": selection["official_nih_download_site"],
    "source_shard": selection["source_shard"],
    "license_note": "NIH Chest X-ray dataset has no use restrictions but requires attribution. HF mirror is CC0 1.0. NIH labels are NLP/report-derived weak labels, not expert-adjudicated ground truth.",
    "model": "TorchXRayVision DenseNet121 densenet121-res224-all",
    "checkpoint_training_overlap": report["checkpoint_training_overlap"],
    "sample_counts": {
        "total_images": report["validation_images"] + report["test_images"],
        "validation_images": report["validation_images"],
        "test_images": report["test_images"],
        "training_rows_excluded": report["training_images_excluded"],
        "unique_patients": selection["unique_patients"],
        "per_image_prediction_rows": len(per_image_rows),
    },
    "patient_level_separation_checked": report["patient_level_separation_checked"],
    "compatible_model_labels": selection["compatible_model_labels"],
    "skipped_model_labels": selection["skipped_model_labels"],
    "skipped_label_reason": "No compatible NIH ChestX-ray14 reference label exists for these model outputs; they are excluded from metric calculations.",
    "score_interpretation": "Raw model scores; not calibrated probabilities.",
    "threshold_method": report["threshold_method"],
    "threshold_choices": [
        {
            "finding": finding["finding"],
            "threshold_raw_score": finding["threshold_raw_score"],
            "threshold_status": finding["threshold_status"],
            "validation_n": finding["validation_n"],
            "validation_positive": finding["validation_positive"],
            "validation_negative": finding["validation_negative"],
        }
        for finding in report["findings"]
    ],
    "metrics": [
        {
            "finding": finding["finding"],
            "test_n": finding["test_n"],
            "test_positive": finding["test_positive"],
            "test_negative": finding["test_negative"],
            "auroc": finding["auroc"],
            "precision": finding["precision"],
            "recall": finding["recall"],
            "f1": finding["f1"],
            "sensitivity": finding["sensitivity"],
            "specificity": finding["specificity"],
            "confusion_matrix": finding["confusion_matrix"],
            "f1_status": finding["f1_status"],
        }
        for finding in report["findings"]
    ],
    "failures": [],
    "output_files": {
        "manifest": str(OUT / "nih_xray_manifest.csv"),
        "selection_json": str(OUT / "nih_xray_selection.json"),
        "evaluation_json": str(OUT / "densenet121.json"),
        "evaluation_csv": str(OUT / "densenet121.csv"),
        "per_image_predictions_csv": str(OUT / "per_image_predictions.csv"),
        "gradcam_gallery": str(OUT / "gradcam_gallery.png"),
    },
}
(OUT / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("Wrote", OUT / "evaluation_summary.json")
