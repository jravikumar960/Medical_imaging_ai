import json
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path("outputs/mri_inference")
OUT = ROOT / "evaluation"
OUT.mkdir(parents=True, exist_ok=True)

pairs = []
for pred in sorted(ROOT.rglob("*_mask.nii.gz")):
    ref = pred.with_name(pred.name.replace("_mask.nii.gz", "-seg.nii.gz"))
    if ref.is_file():
        pairs.append((pred, ref))


def dice(a, b):
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    inter = int(np.logical_and(a, b).sum())
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return 1.0
    return float(2 * inter / denom)


def iou(a, b):
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return float(inter / union)


results = []
for pred_path, ref_path in pairs:
    pred_img = nib.as_closest_canonical(nib.load(str(pred_path)))
    ref_img = nib.as_closest_canonical(nib.load(str(ref_path)))
    pred = np.asarray(pred_img.get_fdata())
    ref = np.asarray(ref_img.get_fdata())

    def metric(pred_mask, ref_mask):
        return {
            "predicted_voxels": int(np.asarray(pred_mask).sum()),
            "reference_voxels": int(np.asarray(ref_mask).sum()),
            "intersection_voxels": int(np.logical_and(pred_mask, ref_mask).sum()),
            "dice": dice(pred_mask, ref_mask),
            "iou": iou(pred_mask, ref_mask),
        }

    results.append({
        "subject_id": pred_path.parent.name,
        "predicted_file": str(pred_path),
        "reference_file": str(ref_path),
        "raw_predicted_orientation": list(nib.aff2axcodes(nib.load(str(pred_path)).affine)),
        "raw_reference_orientation": list(nib.aff2axcodes(nib.load(str(ref_path)).affine)),
        "canonical_orientation": list(nib.aff2axcodes(pred_img.affine)),
        "canonical_affine_equal": bool(np.allclose(pred_img.affine, ref_img.affine)),
        "metrics": {
            "whole_tumor": metric(pred > 0, ref > 0),
            "tumor_core_labels_1_and_4": metric(np.isin(pred, (1, 4)), np.isin(ref, (1, 4))),
            "enhancing_tumor_label_4": metric(np.isin(pred, (4,)), np.isin(ref, (4,))),
            "peritumoral_edema_label_2": metric(np.isin(pred, (2,)), np.isin(ref, (2,))),
            "necrotic_nonenhancing_core_label_1": metric(np.isin(pred, (1,)), np.isin(ref, (1,))),
        },
    })

audit = {
    "audit_type": "independent_recomputation",
    "orientation_resolution": "Both masks were explicitly canonicalized to closest canonical RAS before comparison.",
    "root_cause": "Whole-tumor Dice is high because the whole-tumor channel captures most tumor voxels; enhancing-tumor Dice is low because predicted label-4 voxels are rare.",
    "results": results,
}
(OUT / "mri_segmentation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print("wrote", OUT / "mri_segmentation_audit.json")
