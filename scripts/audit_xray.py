import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("outputs/xray_labeled_evaluation")
manifest = pd.read_csv(ROOT / "nih_xray_manifest.csv")
preds = pd.read_csv(ROOT / "per_image_predictions.csv")
report = json.loads((ROOT / "densenet121.json").read_text())
selection = json.loads((ROOT / "nih_xray_selection.json").read_text())

print("manifest rows", len(manifest))
print("prediction rows", len(preds))
print("validation patients", manifest.loc[manifest.split=="validation","patient_id"].nunique())
print("test patients", manifest.loc[manifest.split=="test","patient_id"].nunique())
print("patient overlap", len(set(manifest.loc[manifest.split=="validation","patient_id"]) & set(manifest.loc[manifest.split=="test","patient_id"])))
print("compatible labels", selection["compatible_model_labels"])
print("skipped labels", selection["skipped_model_labels"])

# Check AUROC recomputation for compatible labels.
labels = selection["compatible_model_labels"]
max_diff = 0
for finding in report["findings"]:
    name = finding["finding"]
    if name not in labels:
        continue
    mask = preds["split"].to_numpy() == "test"
    y = preds.loc[mask, f"reference_{name}"].to_numpy(dtype=float)
    s = preds.loc[mask, f"score_{name}"].to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(s)
    if valid.sum() == 0 or len(np.unique(y[valid])) < 2:
        continue
    auroc = roc_auc_score(y[valid].astype(int), s[valid])
    diff = abs(auroc - finding["auroc"])
    max_diff = max(max_diff, diff)
    if diff > 1e-9:
        print("AUROC mismatch", name, auroc, finding["auroc"])
print("max AUROC diff", max_diff)

# Check thresholded predictions consistency.
bad = 0
for finding in report["findings"]:
    name = finding["finding"]
    thr = finding["threshold_raw_score"]
    if thr is None:
        continue
    expected = (preds[f"score_{name}"] >= thr).astype(int)
    actual = preds[f"predicted_{name}"].fillna(-1).astype(int)
    bad += int((expected != actual).sum())
print("threshold prediction mismatches", bad)
