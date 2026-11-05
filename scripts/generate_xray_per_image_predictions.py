import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv
from PIL import Image

from imaging import prepare_image

ROOT = Path("c:/Users/jravi/Downloads/medical_imaging_ai")
MANIFEST = ROOT / "outputs/xray_labeled_evaluation/nih_xray_manifest.csv"
IMAGE_ROOT = ROOT / "outputs/xray_labeled_evaluation"
EVAL_JSON = ROOT / "outputs/xray_labeled_evaluation/densenet121.json"
OUTPUT = ROOT / "outputs/xray_labeled_evaluation/per_image_predictions.csv"

manifest = pd.read_csv(MANIFEST)
model = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
labels = list(model.pathologies)

report = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
thresholds = {item["finding"]: item["threshold_raw_score"] for item in report["findings"]}

rows = []
for _, row in manifest.iterrows():
    path = IMAGE_ROOT / str(row["image_path"])
    with Image.open(path) as image:
        tensor, _ = prepare_image(image)
    with torch.inference_mode():
        output = model(tensor)
    scores = output[0].detach().cpu().numpy().astype(float)
    record = {
        "image_path": row["image_path"],
        "split": row["split"],
        "patient_id": row["patient_id"],
        "scan_id": row["scan_id"],
    }
    for label in labels:
        record[f"score_{label}"] = float(scores[labels.index(label)])
    for label in labels:
        threshold = thresholds.get(label)
        score = float(scores[labels.index(label)])
        if threshold is None:
            record[f"predicted_{label}"] = ""
        else:
            record[f"predicted_{label}"] = int(score >= threshold)
    for label in labels:
        if label in manifest.columns:
            value = row[label]
            record[f"reference_{label}"] = "" if pd.isna(value) else int(value)
        else:
            record[f"reference_{label}"] = ""
    rows.append(record)

fieldnames = ["image_path", "split", "patient_id", "scan_id"]
for label in labels:
    fieldnames.append(f"reference_{label}")
for label in labels:
    fieldnames.append(f"score_{label}")
for label in labels:
    fieldnames.append(f"predicted_{label}")

with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUTPUT}")
