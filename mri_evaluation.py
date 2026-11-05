"""Evaluation helpers for MRI segmentation masks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

WHOLE_TUMOR_LABELS = (1, 2, 4)
TUMOR_CORE_LABELS = (1, 4)
ENHANCING_LABELS = (4,)
EDEMA_LABELS = (2,)
NECROTIC_LABELS = (1,)


class GeometryMismatchError(ValueError):
    """Raised when two NIfTI images are not spatially aligned."""


def _shape(img: nib.Nifti1Image) -> list[int]:
    return [int(v) for v in img.shape[:3]]


def _spacing(img: nib.Nifti1Image) -> list[float]:
    return [float(v) for v in img.header.get_zooms()[:3]]


def _orientation(img: nib.Nifti1Image) -> list[str]:
    return list(nib.aff2axcodes(img.affine))


def geometry_summary(img: nib.Nifti1Image) -> dict[str, Any]:
    return {
        "shape": _shape(img),
        "spacing_mm": _spacing(img),
        "affine": img.affine.tolist(),
        "orientation": _orientation(img),
    }


def geometry_compatible(pred_img: nib.Nifti1Image, ref_img: nib.Nifti1Image) -> tuple[bool, list[str]]:
    reasons = []
    if _shape(pred_img) != _shape(ref_img):
        reasons.append("shape differs")
    if not np.allclose(_spacing(pred_img), _spacing(ref_img), atol=1e-6, rtol=1e-6):
        reasons.append("voxel spacing differs")
    if not np.allclose(pred_img.affine, ref_img.affine, atol=1e-6, rtol=1e-6):
        reasons.append("affine matrix differs")
    if _orientation(pred_img) != _orientation(ref_img):
        reasons.append("orientation differs")
    return (len(reasons) == 0, reasons)


def dice_score(pred_mask, ref_mask) -> float:
    pred = np.asarray(pred_mask).astype(bool)
    ref = np.asarray(ref_mask).astype(bool)
    if pred.shape != ref.shape:
        raise ValueError("Mask shapes must match.")
    intersection = int(np.logical_and(pred, ref).sum())
    pred_count = int(pred.sum())
    ref_count = int(ref.sum())
    if pred_count == 0 and ref_count == 0:
        return 1.0
    denominator = pred_count + ref_count
    if denominator == 0:
        return 1.0
    return float(2 * intersection / denominator)
def _subject_id(path):
    name = Path(path).name
    for suffix in ("_mask.nii.gz", "-seg.nii.gz", ".nii.gz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def _label_mask(mask, labels):
    return np.isin(mask, labels)


def evaluate_case(pred_path, ref_path):
    pred_path = Path(pred_path)
    ref_path = Path(ref_path)
    pred_subject = _subject_id(pred_path)
    ref_subject = _subject_id(ref_path)
    if pred_subject != ref_subject:
        raise ValueError(f"Subject mismatch: predicted {pred_subject!r} != reference {ref_subject!r}")

    pred_img = nib.load(str(pred_path))
    ref_img = nib.load(str(ref_path))

    raw_pred_summary = geometry_summary(pred_img)
    raw_ref_summary = geometry_summary(ref_img)
    raw_compatible, raw_reasons = geometry_compatible(pred_img, ref_img)

    orientation_corrected = False
    if not raw_compatible:
        pred_img = nib.as_closest_canonical(pred_img)
        ref_img = nib.as_closest_canonical(ref_img)
        orientation_corrected = True

    compatible, reasons = geometry_compatible(pred_img, ref_img)
    if not compatible:
        raise GeometryMismatchError(
            "Predicted and reference masks are not spatially aligned after explicit canonicalization: "
            + ", ".join(reasons)
        )

    pred = np.asarray(pred_img.get_fdata())
    ref = np.asarray(ref_img.get_fdata())

    pred_wt = pred > 0
    ref_wt = ref > 0
    metrics = {
        "whole_tumor": {
            "labels": "all nonzero labels",
            "predicted_voxels": int(pred_wt.sum()),
            "reference_voxels": int(ref_wt.sum()),
            "intersection_voxels": int(np.logical_and(pred_wt, ref_wt).sum()),
            "union_voxels": int(np.logical_or(pred_wt, ref_wt).sum()),
            "dice": dice_score(pred_wt, ref_wt),
            "iou": iou_score(pred_wt, ref_wt),
        }
    }

    label_definitions = {
        "necrotic_nonenhancing_core_label_1": {
            "labels": list(NECROTIC_LABELS),
            "compatibility": "reference contains label 1",
            "reference_contains_label": bool(np.isin(ref, NECROTIC_LABELS).any()),
        },
        "peritumoral_edema_label_2": {
            "labels": list(EDEMA_LABELS),
            "compatibility": "reference contains label 2",
            "reference_contains_label": bool(np.isin(ref, EDEMA_LABELS).any()),
        },
        "enhancing_tumor_label_4": {
            "labels": list(ENHANCING_LABELS),
            "compatibility": "reference contains label 4",
            "reference_contains_label": bool(np.isin(ref, ENHANCING_LABELS).any()),
        },
        "tumor_core_labels_1_and_4": {
            "labels": list(TUMOR_CORE_LABELS),
            "compatibility": "reference contains at least one of labels 1 or 4",
            "reference_contains_label": bool(np.isin(ref, TUMOR_CORE_LABELS).any()),
        },
    }

    for name, definition in label_definitions.items():
        p_mask = _label_mask(pred, tuple(definition["labels"]))
        r_mask = _label_mask(ref, tuple(definition["labels"]))
        compatible_label = definition["reference_contains_label"] or (not r_mask.any() and not p_mask.any())
        if not compatible_label:
            metrics[name] = {
                "labels": definition["labels"],
                "compatible": False,
                "reason": "reference mask does not contain this label definition",
                "predicted_voxels": int(p_mask.sum()),
                "reference_voxels": int(r_mask.sum()),
                "intersection_voxels": int(np.logical_and(p_mask, r_mask).sum()),
                "dice": None,
                "iou": None,
            }
            continue
        metrics[name] = {
            "labels": definition["labels"],
            "compatible": True,
            "predicted_voxels": int(p_mask.sum()),
            "reference_voxels": int(r_mask.sum()),
            "intersection_voxels": int(np.logical_and(p_mask, r_mask).sum()),
            "dice": dice_score(p_mask, r_mask),
            "iou": iou_score(p_mask, r_mask),
        }

    return {
        "subject_id": pred_subject,
        "predicted_file": str(pred_path),
        "reference_file": str(ref_path),
        "raw_predicted_geometry": raw_pred_summary,
        "raw_reference_geometry": raw_ref_summary,
        "raw_geometry_compatible": raw_compatible,
        "raw_geometry_reasons": raw_reasons,
        "orientation_corrected_to_ras": orientation_corrected,
        "canonical_geometry": geometry_summary(pred_img),
        "label_mapping": {
            "1": "necrotic and non-enhancing tumor core (NCR/NET)",
            "2": "peritumoral edema (ED)",
            "4": "enhancing tumor (ET)",
            "whole_tumor": "all nonzero labels (1, 2, 4)",
            "tumor_core": "labels 1 and 4",
        },
        "metrics": metrics,
        "empty_mask_handling": "Dice and IoU are 1.0 when both masks are empty; 0.0 when exactly one mask is empty.",
    }


def aggregate_results(results):
    if not results:
        return {}
    aggregate = {"n_cases": len(results)}
    for metric_name in ("whole_tumor", "necrotic_nonenhancing_core_label_1", "peritumoral_edema_label_2", "enhancing_tumor_label_4", "tumor_core_labels_1_and_4"):
        dices = [r["metrics"][metric_name]["dice"] for r in results if r["metrics"].get(metric_name, {}).get("dice") is not None]
        if dices:
            aggregate.setdefault(metric_name, {})["dice_mean"] = float(__import__("numpy").mean(dices))
            aggregate[metric_name]["dice_median"] = float(__import__("numpy").median(dices))
            aggregate[metric_name]["n_computed"] = len(dices)
    return aggregate


def iou_score(pred_mask, ref_mask) -> float:
    pred = np.asarray(pred_mask).astype(bool)
    ref = np.asarray(ref_mask).astype(bool)
    if pred.shape != ref.shape:
        raise ValueError("Mask shapes must match.")
    intersection = int(np.logical_and(pred, ref).sum())
    union = int(np.logical_or(pred, ref).sum())
    if union == 0:
        return 1.0
    return float(intersection / union)
