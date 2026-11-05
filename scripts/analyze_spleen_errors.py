"""Quantitative error analysis for the 3-case offline spleen predictions."""
import argparse, json
from pathlib import Path
import nibabel as nib
import numpy as np


def confusion_counts(pred, gt):
    pred = (pred == 1)
    gt = (gt == 1)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def safe_div(num, den, default=1.0):
    if den == 0:
        return float(default)
    return float(num / den)


def metrics_from_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    dice = (2 * tp / denom) if denom else 1.0
    precision = safe_div(tp, tp + fp, default=1.0)
    recall = safe_div(tp, tp + fn, default=1.0)
    return {"dice": float(dice), "precision": float(precision), "recall": float(recall)}


def slice_stats(pred_vol, gt_vol):
    """Return list of per-slice dicts with tp,fp,fn,dice,precision,recall."""
    d = pred_vol.shape[2]
    out = []
    for zi in range(d):
        c = confusion_counts(pred_vol[:, :, zi], gt_vol[:, :, zi])
        m = metrics_from_counts(c["tp"], c["fp"], c["fn"])
        out.append({
            "slice": int(zi),
            "tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "tn": c["tn"],
            "dice": m["dice"], "precision": m["precision"], "recall": m["recall"],
            "pred_voxels": int((pred_vol[:, :, zi] == 1).sum()),
            "gt_voxels": int((gt_vol[:, :, zi] == 1).sum()),
        })
    return out


def select_worst_slices(stats, key, k=5):
    """Return top-k slices sorted by key descending (then slice asc)."""
    ordered = sorted(stats, key=lambda s: (-s[key], s["slice"]))
    return [s["slice"] for s in ordered[:k] if s[key] > 0]


def case_analysis(img, pred, gt, case):
    _check_geometry(img, pred, gt)
    p = np.asarray(pred.get_fdata()).astype(np.uint8)
    g = np.asarray(gt.get_fdata()).astype(np.uint8)
    ss = slice_stats(p, g)
    c = confusion_counts(p, g)
    m = metrics_from_counts(c["tp"], c["fp"], c["fn"])
    worst_fp = select_worst_slices(ss, "fp", 5)
    worst_fn = select_worst_slices(ss, "fn", 5)
    # worst dice among slices with any GT or pred
    eligible = [s for s in ss if (s["gt_voxels"] + s["pred_voxels"]) > 0]
    worst_dice = sorted(eligible, key=lambda s: (s["dice"], s["slice"]))[:5]
    worst_dice_slices = [s["slice"] for s in worst_dice]
    return {
        "case_id": case["case_id"],
        "shape": list(pred.shape),
        "counts": c,
        "metrics": m,
        "mean_slice_dice": float(np.mean([s["dice"] for s in eligible])) if eligible else 1.0,
        "worst_fp_slices": worst_fp,
        "worst_fn_slices": worst_fn,
        "worst_dice_slices": worst_dice_slices,
        "slice_stats": ss,
    }


def _check_geometry(img, pred, gt):
    if img.shape != pred.shape or img.shape != gt.shape:
        raise ValueError("shape mismatch")
    if not (np.allclose(img.affine, pred.affine, atol=1e-5) and np.allclose(img.affine, gt.affine, atol=1e-5)):
        raise ValueError("affine mismatch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-dir", required=True)
    args = ap.parse_args()
    rd = Path(args.review_dir)
    rep = json.loads((rd / "spleen_evaluation_results.json").read_text())
    cases_out = []
    for case in rep["cases"]:
        img = nib.load(case["image_path"])
        pred = nib.load(case["prediction_path"])
        gt = nib.load(case["label_path"])
        ca = case_analysis(img, pred, gt, case)
        cases_out.append(ca)
    # summary
    total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for ca in cases_out:
        for k in total:
            total[k] += ca["counts"][k]
    summary = {"counts": total, "metrics": metrics_from_counts(total["tp"], total["fp"], total["fn"])}
    out = {
        "review_dir": str(rd.resolve()),
        "cases": cases_out,
        "aggregate": summary,
    }
    out_path = rd / "error_analysis.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps({"wrote": str(out_path),
                      "aggregate": summary,
                      "per_case": [{"case_id": c["case_id"], "metrics": c["metrics"],
                                    "counts": c["counts"],
                                    "worst_fp_slices": c["worst_fp_slices"],
                                    "worst_fn_slices": c["worst_fn_slices"],
                                    "worst_dice_slices": c["worst_dice_slices"]} for c in cases_out]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
