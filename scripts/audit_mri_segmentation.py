import json
from pathlib import Path

import nibabel as nib
import numpy as np

from mri_evaluation import geometry_compatible

ROOT = Path("outputs/mri_inference")
EVAL_DIR = ROOT / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
EXISTING = json.loads((EVAL_DIR / "mri_segmentation_evaluation.json").read_text())


def independent_dice(pred, ref):
    pred = np.asarray(pred).astype(bool)
    ref = np.asarray(ref).astype(bool)
    inter = int(np.logical_and(pred, ref).sum())
    denom = int(pred.sum()) + int(ref.sum())
    if denom == 0:
        return 1.0
    return float(2 * inter / denom)


def independent_iou(pred, ref):
    pred = np.asarray(pred).astype(bool)
    ref = np.asarray(ref).astype(bool)
    inter = int(np.logical_and(pred, ref).sum())
    union = int(np.logical_or(pred, ref).sum())
    if union == 0:
        return 1.0
    return float(inter / union)


pairs = []
for pred in sorted(ROOT.rglob("*_mask.nii.gz")):
    ref = pred.with_name(pred.name.replace("_mask.nii.gz", "-seg.nii.gz"))
    if ref.is_file():
        pairs.append((pred, ref))

audit_results = []
for pred_path, ref_path in pairs:
    pred_img = nib.load(str(pred_path))
    ref_img = nib.load(str(ref_path))
    raw_compatible, raw_reasons = geometry_compatible(pred_img, ref_img)
    pred_can = nib.as_closest_canonical(pred_img)
    ref_can = nib.as_closest_canonical(ref_img)
    canon_compatible, canon_reasons = geometry_compatible(pred_can, ref_can)
    pred = np.asarray(pred_can.get_fdata())
    ref = np.asarray(ref_can.get_fdata())

    whole_p, whole_r = pred > 0, ref > 0
    tc_p, tc_r = np.isin(pred, (1, 4)), np.isin(ref, (1, 4))
    et_p, et_r = np.isin(pred, (4,)), np.isin(ref, (4,))
    ed_p, ed_r = np.isin(pred, (2,)), np.isin(ref, (2,))
    ncr_p, ncr_r = np.isin(pred, (1,)), np.isin(ref, (1,))

    audit_results.append({
        "subject_id": pred_path.parent.name,
        "predicted_file": str(pred_path),
        "reference_file": str(ref_path),
        "raw_geometry_compatible": raw_compatible,
        "raw_geometry_reasons": raw_reasons,
        "canonical_geometry_compatible": canon_compatible,
        "canonical_geometry_reasons": canon_reasons,
        "canonical_orientation": list(nib.aff2axcodes(pred_can.affine)),
        "canonical_affine_equal": bool(np.allclose(pred_can.affine, ref_can.affine)),
        "metrics": {
            "whole_tumor": {
                "predicted_voxels": int(whole_p.sum()),
                "reference_voxels": int(whole_r.sum()),
                "intersection_voxels": int(np.logical_and(whole_p, whole_r).sum()),
                "dice": independent_dice(whole_p, whole_r),
                "iou": independent_iou(whole_p, whole_r),
            },
            "tumor_core_labels_1_and_4": {
                "predicted_voxels": int(tc_p.sum()),
                "reference_voxels": int(tc_r.sum()),
                "intersection_voxels": int(np.logical_and(tc_p, tc_r).sum()),
                "dice": independent_dice(tc_p, tc_r),
                "iou": independent_iou(tc_p, tc_r),
            },
            "enhancing_tumor_label_4": {
                "predicted_voxels": int(et_p.sum()),
                "reference_voxels": int(et_r.sum()),
                "intersection_voxels": int(np.logical_and(et_p, et_r).sum()),
                "dice": independent_dice(et_p, et_r),
                "iou": independent_iou(et_p, et_r),
            },
            "peritumoral_edema_label_2": {
                "predicted_voxels": int(ed_p.sum()),
                "reference_voxels": int(ed_r.sum()),
                "intersection_voxels": int(np.logical_and(ed_p, ed_r).sum()),
                "dice": independent_dice(ed_p, ed_r),
                "iou": independent_iou(ed_p, ed_r),
            },
            "necrotic_nonenhancing_core_label_1": {
                "predicted_voxels": int(ncr_p.sum()),
                "reference_voxels": int(ncr_r.sum()),
                "intersection_voxels": int(np.logical_and(ncr_p, ncr_r).sum()),
                "dice": independent_dice(ncr_p, ncr_r),
                "iou": independent_iou(ncr_p, ncr_r),
            },
        },
    })
PY