import csv
import io
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path("c:/Users/jravi/Downloads/medical_imaging_ai")
SHARD = ROOT / "outputs/xray_labeled_evaluation/source/train-00000-of-00006.parquet"
OUT = ROOT / "outputs/xray_labeled_evaluation"
IMAGE_DIR = OUT / "images"
MANIFEST = OUT / "nih_xray_manifest.csv"
SELECTION_JSON = OUT / "nih_xray_selection.json"

MODEL_LABELS = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia",
    "Lung Lesion", "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum",
]

NIH_TO_MODEL_LABEL = {
    1: "Atelectasis", 2: "Cardiomegaly", 3: "Effusion", 4: "Infiltration",
    5: "Mass", 6: "Nodule", 7: "Pneumonia", 8: "Pneumothorax",
    9: "Consolidation", 10: "Edema", 11: "Emphysema", 12: "Fibrosis",
    13: "Pleural_Thickening", 14: "Hernia",
}
SKIPPED_MODEL_LABELS = ["Lung Lesion", "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum"]

TARGET_PATIENTS = 300
VALIDATION_PATIENTS = 60
RANDOM_SEED = 42

# Read metadata only first.
meta = pq.read_table(SHARD, columns=["patient_id", "scan_id", "labels"])
patient_ids = meta.column("patient_id").to_pylist()
scan_ids = meta.column("scan_id").to_pylist()
labels_list = meta.column("labels").to_pylist()

candidates = {}
for idx, (pid, sid, labels) in enumerate(zip(patient_ids, scan_ids, labels_list)):
    if pid not in candidates:
        candidates[pid] = {"patient_id": pid, "scan_id": int(sid), "labels": list(labels), "row_index": idx}

patient_rows = list(candidates.values())
random.seed(RANDOM_SEED)

label_patient_counts = Counter()
for row in patient_rows:
    for label in set(row["labels"]):
        label_patient_counts[label] += 1


def rarity(row):
    return sum(1.0 / (label_patient_counts[label] + 1) for label in set(row["labels"]))


ordered = sorted(patient_rows, key=lambda row: (-rarity(row), row["patient_id"]))
selected = ordered[:120]
remaining = [row for row in patient_rows if row not in selected]
random.shuffle(remaining)
selected = selected + remaining[:TARGET_PATIENTS - len(selected)]
assert len(selected) == TARGET_PATIENTS
assert len({row["patient_id"] for row in selected}) == TARGET_PATIENTS

shuffled = selected.copy()
random.shuffle(shuffled)
validation_ids = {row["patient_id"] for row in shuffled[:VALIDATION_PATIENTS]}

patient_by_id = {row["patient_id"]: row for row in selected}
for nih_index in sorted(NIH_TO_MODEL_LABEL):
    val_pos = sum(1 for pid in validation_ids if nih_index in set(patient_by_id[pid]["labels"]))
    while val_pos < 5:
        candidate_test = [pid for pid in (set(patient_by_id) - validation_ids) if nih_index in set(patient_by_id[pid]["labels"])]
        candidate_val = [pid for pid in validation_ids if nih_index not in set(patient_by_id[pid]["labels"])]
        if not candidate_test or not candidate_val:
            break
        validation_ids.remove(candidate_val[0])
        validation_ids.add(candidate_test[0])
        val_pos += 1

for row in selected:
    row["split"] = "validation" if row["patient_id"] in validation_ids else "test"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
for old_image in IMAGE_DIR.glob("*.png"):
    old_image.unlink()

selected_keys = {(row["patient_id"], row["scan_id"]): row for row in selected}
found_keys = set()
pf = pq.ParquetFile(SHARD)
for batch in pf.iter_batches(batch_size=1024, columns=["patient_id", "scan_id", "image"]):
    for record in batch.to_pylist():
        key = (record["patient_id"], int(record["scan_id"]))
        if key not in selected_keys or key in found_keys:
            continue
        row = selected_keys[key]
        image_bytes = record["image"]["bytes"]
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.convert("L").save(IMAGE_DIR / f"{row['patient_id']}_{row['scan_id']}.png", format="PNG")
        found_keys.add(key)

missing = set(selected_keys) - found_keys
if missing:
    raise RuntimeError(f"Could not extract {len(missing)} selected images: {sorted(missing)[:5]}")
fieldnames = ["image_path", "split", "patient_id", "scan_id"] + MODEL_LABELS
with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in sorted(selected, key=lambda item: (item["split"], item["patient_id"], item["scan_id"])):
        labels = set(row["labels"])
        record = {
            "image_path": f"images/{row['patient_id']}_{row['scan_id']}.png",
            "split": row["split"],
            "patient_id": row["patient_id"],
            "scan_id": row["scan_id"],
        }
        for model_label in MODEL_LABELS:
            record[model_label] = ""
        for nih_index, model_label in NIH_TO_MODEL_LABEL.items():
            record[model_label] = 1 if nih_index in labels else 0
        writer.writerow(record)


def label_counts(rows):
    counts = Counter()
    for row in rows:
        for label in set(row["labels"]):
            counts[label] += 1
    return {str(key): counts[key] for key in sorted(counts)}


selection_summary = {
    "source_dataset": "chehablab/NIHChestXR (Hugging Face mirror of NIH ChestX-ray14)",
    "source_shard": "data/train-00000-of-00006.parquet",
    "source_url": "https://huggingface.co/datasets/chehablab/NIHChestXR",
    "official_nih_download_site": "https://nihcc.app.box.com/v/ChestXray-NIHCC",
    "selection_method": "One representative scan per patient; 120 rare-finding patients plus random fill to 300; deterministic seed 42.",
    "target_patients": TARGET_PATIENTS,
    "validation_patients": VALIDATION_PATIENTS,
    "test_patients": TARGET_PATIENTS - VALIDATION_PATIENTS,
    "unique_patients": len({row["patient_id"] for row in selected}),
    "images_extracted": len(found_keys),
    "nih_label_index_map": NIH_TO_MODEL_LABEL,
    "compatible_model_labels": list(NIH_TO_MODEL_LABEL.values()),
    "skipped_model_labels": SKIPPED_MODEL_LABELS,
    "validation_label_counts": label_counts([row for row in selected if row["split"] == "validation"]),
    "test_label_counts": label_counts([row for row in selected if row["split"] == "test"]),
    "all_selected_label_counts": label_counts(selected),
    "label_note": "NIH labels are NLP/report-derived weak labels, not expert-adjudicated ground truth.",
}
SELECTION_JSON.write_text(json.dumps(selection_summary, indent=2), encoding="utf-8")
print(json.dumps(selection_summary, indent=2))
