import json
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from pathlib import Path

import numpy as np

from mri_evaluation import aggregate_results, evaluate_case

ROOT = Path("outputs/mri_inference")
OUT = ROOT / "evaluation"
OUT.mkdir(parents=True, exist_ok=True)

pairs = []
for pred in sorted(ROOT.rglob("*_mask.nii.gz")):
    ref = pred.with_name(pred.name.replace("_mask.nii.gz", "-seg.nii.gz"))
    if ref.is_file():
        pairs.append((pred, ref))

results = [evaluate_case(pred, ref) for pred, ref in pairs]
aggregate = aggregate_results(results)

payload = {
    "evaluation_scope": "Existing predicted MRI masks matched to same-subject BraTS reference masks.",
    "n_pairs": len(results),
    "orientation_resolution": "Raw predicted (RAS) and reference (LAS) affines differed by a left-right axis. Both masks were explicitly canonicalized to closest canonical RAS before comparison. No resampling or arbitrary transpose was used.",
    "results": results,
    "aggregate": aggregate,
}
(OUT / "mri_segmentation_evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = [
    "# MRI Segmentation Evaluation",
    "",
    f"Cases evaluated: {len(results)}",
    "",
    "Orientation note: predicted masks were RAS and reference masks were LAS. Both were explicitly canonicalized to closest canonical RAS before comparison. Raw geometry differences are recorded in the JSON.",
    "",
    "## Aggregate metrics",
    "",
    "| Metric | Dice mean | Dice median | N |",
    "|---|---:|---:|---:|",
]
for name in ("whole_tumor", "tumor_core_labels_1_and_4", "enhancing_tumor_label_4", "peritumoral_edema_label_2", "necrotic_nonenhancing_core_label_1"):
    info = aggregate.get(name, {})
    if info:
        lines.append(f"| {name} | {info['dice_mean']:.6f} | {info['dice_median']:.6f} | {info['n_computed']} |")
lines.extend(["", "## Per-case metrics", ""])
for r in results:
    lines.append(f"### {r['subject_id']}")
    lines.append("")
    lines.append(f"- Predicted: `{r['predicted_file']}`")
    lines.append(f"- Reference: `{r['reference_file']}`")
    lines.append(f"- Raw geometry compatible: {r['raw_geometry_compatible']} ({', '.join(r['raw_geometry_reasons']) or 'none'})")
    lines.append(f"- Orientation corrected to RAS: {r['orientation_corrected_to_ras']}")
    lines.append("")
    lines.append("| Metric | Pred voxels | Ref voxels | Intersection | Dice | IoU |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, info in r["metrics"].items():
        dice = "n/a" if info.get("dice") is None else f"{info['dice']:.6f}"
        iou = "n/a" if info.get("iou") is None else f"{info['iou']:.6f}"
        lines.append(f"| {name} | {info['predicted_voxels']} | {info['reference_voxels']} | {info['intersection_voxels']} | {dice} | {iou} |")
    lines.append("")

(OUT / "mri_segmentation_evaluation.md").write_text("\n".join(lines), encoding="utf-8")
print(f"Evaluated {len(results)} cases")
print(json.dumps(aggregate, indent=2))
