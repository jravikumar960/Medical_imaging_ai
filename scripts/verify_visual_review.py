"""Verify saved prediction masks for the 3-case visual review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", required=True)
    args = parser.parse_args()
    review_dir = Path(args.review_dir)
    report_path = review_dir / "spleen_evaluation_results.json"
    report = json.loads(report_path.read_text())
    manifest = {
        "review_dir": str(review_dir.resolve()),
        "source_report": str(report_path.resolve()),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "checkpoint_sha256_matches": report["checkpoint_sha256_matches"],
        "cases": [],
    }
    all_ok = True
    for case in report["cases"]:
        cid = case["case_id"]
        pred_path = Path(case["prediction_path"])
        img = nib.load(case["image_path"])
        lab = nib.load(case["label_path"])
        pred = nib.load(str(pred_path))
        img_arr = np.asarray(img.get_fdata())
        lab_arr = np.asarray(lab.get_fdata())
        pred_arr = np.asarray(pred.get_fdata())
        unique_labels = sorted(set(int(x) for x in np.unique(pred_arr)))
        voxel_count = int((pred_arr == 1).sum())
        shape_ok = list(pred.shape) == list(img.shape) == list(lab.shape)
        affine_tol = 1e-5
        affine_ct_ok = bool(np.allclose(pred.affine, img.affine, atol=affine_tol))
        affine_lab_ok = bool(np.allclose(pred.affine, lab.affine, atol=affine_tol))
        labels_ok = set(unique_labels).issubset({0, 1})
        voxel_count_ok = voxel_count == int(case["predicted_voxels"])
        tp = int(((pred_arr == 1) & (lab_arr == 1)).sum())
        fp = int(((pred_arr == 1) & (lab_arr != 1)).sum())
        fn = int(((pred_arr != 1) & (lab_arr == 1)).sum())
        denom = 2 * tp + fp + fn
        dice = (2 * tp / denom) if denom else 1.0
        dice_ok = abs(dice - float(case["dice"])) < 1e-6
        case_ok = all([shape_ok, affine_ct_ok, affine_lab_ok, labels_ok, voxel_count_ok, dice_ok])
        all_ok = all_ok and case_ok
        manifest["cases"].append({
            "case_id": cid,
            "dice_reported": float(case["dice"]),
            "dice_recomputed": float(dice),
            "source_ct_path": case["image_path"],
            "reference_label_path": case["label_path"],
            "prediction_path": str(pred_path.resolve()),
            "shape_source_ct": list(img.shape),
            "shape_reference_label": list(lab.shape),
            "shape_prediction": list(pred.shape),
            "shape_match": shape_ok,
            "affine_matches_source_ct": affine_ct_ok,
            "affine_matches_reference_label": affine_lab_ok,
            "valid_labels": unique_labels,
            "labels_valid": labels_ok,
            "predicted_voxel_count": voxel_count,
            "reported_predicted_voxels": int(case["predicted_voxels"]),
            "voxel_count_match": voxel_count_ok,
            "dice_match": dice_ok,
            "all_checks_passed": case_ok,
        })
    manifest["all_cases_passed"] = all_ok
    out = review_dir / "verification_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"all_cases_passed": all_ok, "manifest": str(out)}, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    main()
