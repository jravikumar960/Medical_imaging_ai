import json
from pathlib import Path

data = json.loads(Path("outputs/mri_inference/evaluation/mri_segmentation_audit.json").read_text())
lines = [
    "# MRI Segmentation Audit",
    "",
    "Independent recomputation of saved predicted masks against same-subject BraTS reference masks.",
    "",
    "## Root cause",
    "",
    "Whole-tumor mean Dice is `0.739532`, while enhancing-tumor mean Dice is `0.017376`. The saved predicted masks contain very few label-4 voxels. Whole-tumor Dice is dominated by the larger edema/whole-tumor channel, while enhancing-tumor Dice depends on the rare enhancing channel.",
    "",
    "## Channel trace",
    "",
    "| Model channel | Threshold | Final label | Meaning |",
    "|---|---:|---:|---|",
    "| 0 | 0.5 | 1, unless channel 2 is also true | tumor core |",
    "| 1 | 0.5 | 2, unless channel 0 or 2 is true | whole-tumor remainder |",
    "| 2 | 0.5 | 4 | enhancing tumor |",
    "",
    "Final label 0 is background. Assignment order is whole tumor, then tumor core, then enhancing tumor.",
    "",
    "## Orientation",
    "",
    "Predicted masks were RAS and reference masks were LAS. Both were explicitly canonicalized to RAS before comparison. After canonicalization, affines and orientations matched.",
    "",
    "## Independent recomputation",
    "",
    "| Subject | Metric | Pred voxels | Ref voxels | Intersection | Dice | IoU |",
    "|---|---|---:|---:|---:|---:|---:|",
]
for case in data["results"]:
    for name, info in case["metrics"].items():
        lines.append(f"| {case['subject_id']} | {name} | {info['predicted_voxels']} | {info['reference_voxels']} | {info['intersection_voxels']} | {info['dice']:.6f} | {info['iou']:.6f} |")
lines.extend(["", "## Contribution counts", ""])
for case in data["results"]:
    pass
from collections import Counter
counts = Counter()
for case in data["results"]:
    for name, info in case["metrics"].items():
        if info["reference_voxels"] > 0:
            counts[name + "_ref"] += 1
        if info["predicted_voxels"] > 0:
            counts[name + "_pred"] += 1
for name in ("whole_tumor", "tumor_core_labels_1_and_4", "enhancing_tumor_label_4", "peritumoral_edema_label_2", "necrotic_nonenhancing_core_label_1"):
    lines.append(f"- {name}: reference-present cases = {counts[name+'_ref']}, predicted-present cases = {counts[name+'_pred']}")
lines.extend(["", "No clinical validation is claimed."])
Path("outputs/mri_inference/evaluation/mri_segmentation_audit.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote markdown")
