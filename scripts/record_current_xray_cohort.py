import hashlib
import json
from pathlib import Path

import pandas as pd

MANIFEST = Path("outputs/xray_labeled_evaluation/nih_xray_manifest.csv")
IMAGE_ROOT = Path("outputs/xray_labeled_evaluation")
OUT = Path("outputs/xray_gradcam/classifier_evaluation_readiness")
OUT.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(MANIFEST)
records = []
for _, row in manifest.iterrows():
    path = IMAGE_ROOT / str(row["image_path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    records.append({
        "image_path": str(row["image_path"]),
        "split": str(row["split"]),
        "patient_id": str(row["patient_id"]),
        "scan_id": int(row["scan_id"]),
        "sha256": digest,
        "size_bytes": int(path.stat().st_size),
    })

payload = {
    "cohort_source": "chehablab/NIHChestXR Hugging Face mirror of NIH ChestX-ray14",
    "source_shard": "data/train-00000-of-00006.parquet",
    "n_images": len(records),
    "n_validation": int((manifest["split"] == "validation").sum()),
    "n_test": int((manifest["split"] == "test").sum()),
    "unique_patients": int(manifest["patient_id"].nunique()),
    "images": records,
}
(OUT / "current_test_cohort_identifiers.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("wrote", OUT / "current_test_cohort_identifiers.json")
PY