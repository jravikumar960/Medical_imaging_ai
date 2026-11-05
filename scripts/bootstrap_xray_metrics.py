import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("outputs/xray_gradcam/classifier_evaluation_readiness")
PREDICTIONS = Path("outputs/xray_labeled_evaluation/per_image_predictions.csv")
REPORT = Path("outputs/xray_labeled_evaluation/densenet121.json")
SELECTION = Path("outputs/xray_labeled_evaluation/nih_xray_selection.json")

preds = pd.read_csv(PREDICTIONS)
report = json.loads(REPORT.read_text())
compatible = json.loads(SELECTION.read_text())["compatible_model_labels"]
test = preds[preds["split"] == "test"].reset_index(drop=True)

rng = np.random.default_rng(42)
n_reps = 1000
patient_ids = test["patient_id"].to_numpy()

bootstrap = {}
for finding in compatible:
    y = test[f"reference_{finding}"].to_numpy(dtype=float)
    s = test[f"score_{finding}"].to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(s)
    y = y[valid].astype(int)
    s = s[valid]
    patients = patient_ids[valid]
    aurocs = []
    auprcs = []
    n_valid_reps = 0
    for _ in range(n_reps):
        sampled_patients = rng.choice(patients, size=len(patients), replace=True)
        indices = np.array([np.where(patients == p)[0][0] for p in sampled_patients])  # one row per patient
        # Better: map patient to row index.
        # Rebuild mapping outside loop? We'll fix below.
        yy = y[indices]
        ss = s[indices]
        if len(np.unique(yy)) < 2:
            continue
        aurocs.append(roc_auc_score(yy, ss))
        auprcs.append(average_precision_score(yy, ss))
        n_valid_reps += 1
    bootstrap[finding] = {
        "n_replicates_requested": n_reps,
        "n_replicates_valid": n_valid_reps,
        "auroc_ci_95": [float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))] if aurocs else None,
        "auprc_ci_95": [float(np.percentile(auprcs, 2.5)), float(np.percentile(auprcs, 97.5))] if auprcs else None,
        "prevalence_baseline_auprc": float(y.mean()),
    }

# The above patient mapping is incorrect because patients are unique in this cohort; replace with direct row bootstrap.
# Recompute correctly.
bootstrap = {}
for finding in compatible:
    y = test[f"reference_{finding}"].to_numpy(dtype=float)
    s = test[f"score_{finding}"].to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(s)
    y = y[valid].astype(int)
    s = s[valid]
    patients = patient_ids[valid]
    unique_patients = np.unique(patients)
    patient_to_rows = {p: np.where(patients == p)[0] for p in unique_patients}
    aurocs = []
    auprcs = []
    n_valid = 0
    for _ in range(n_reps):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        indices = np.concatenate([patient_to_rows[p] for p in sampled])
        yy = y[indices]
        ss = s[indices]
        if len(np.unique(yy)) < 2:
            continue
        aurocs.append(roc_auc_score(yy, ss))
        auprcs.append(average_precision_score(yy, ss))
        n_valid += 1
    bootstrap[finding] = {
        "n_replicates_requested": n_reps,
        "n_replicates_valid": n_valid,
        "auroc_ci_95": [float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))] if aurocs else None,
        "auprc_ci_95": [float(np.percentile(auprcs, 2.5)), float(np.percentile(auprcs, 97.5))] if auprcs else None,
        "prevalence_baseline_auprc": float(y.mean()),
        "positive_count": int((y == 1).sum()),
        "negative_count": int((y == 0).sum()),
    }

(ROOT / "bootstrap_confidence_intervals.json").write_text(json.dumps({"seed": 42, "repetitions": n_reps, "method": "patient-level bootstrap with replacement", "results": bootstrap}, indent=2), encoding="utf-8")
print("wrote bootstrap")
