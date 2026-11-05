"""Offline research viewer for spleen segmentation worst slices.

This is a research-only helper. It never touches the production CT workflow,
safety gates, or checkpoint. Given a case's image, prediction, and reference
NIfTI volumes (or a saved slice-metrics JSON), it renders the worst slices
with TP/FP/FN colour overlays and per-slice metrics to a PNG/HTML report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except Exception:  # pragma: no cover
    nib = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _load_volume(path):
    if nib is None:
        raise RuntimeError("nibabel is required for the offline viewer")
    image = nib.load(str(path))
    return np.asarray(image.get_fdata()), image


def _normalize_slice(slice_2d):
    arr = np.asarray(slice_2d, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo, hi = np.percentile(arr, (1, 99))
    if hi - lo < 1e-9:
        lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _overlay_rgb(ct_slice, pred_slice, ref_slice):
    ct = _normalize_slice(ct_slice)
    rgb = np.stack([ct, ct, ct], axis=-1)
    tp = (pred_slice > 0) & (ref_slice > 0)
    fp = (pred_slice > 0) & (ref_slice == 0)
    fn = (pred_slice == 0) & (ref_slice > 0)
    # TP green, FP red, FN blue
    rgb[tp] = [0.2, 0.9, 0.2]
    rgb[fp] = [0.95, 0.2, 0.2]
    rgb[fn] = [0.2, 0.4, 0.95]
    return rgb, tp, fp, fn


def _check_geometry(img, pred, gt):
    if img.shape != pred.shape or img.shape != gt.shape:
        raise ValueError("shape mismatch: ct%s pred%s gt%s" % (img.shape, pred.shape, gt.shape))
    if not np.allclose(img.affine, pred.affine, atol=1e-5) or not np.allclose(img.affine, gt.affine, atol=1e-5):
        raise ValueError("affine mismatch between CT, prediction, and ground-truth label")
def render_worst_slices(
    *,
    image_path=None,
    prediction_path=None,
    reference_path=None,
    slice_metrics_json=None,
    k=5,
    output_path="worst_slices.png",
    title=None,
):
    """Render worst slices to ``output_path``. Returns the output path."""
    if slice_metrics_json is not None:
        payload = json.loads(Path(slice_metrics_json).read_text(encoding="utf-8"))
        worst = payload.get("worst_slices", [])
        metrics_list = payload.get("slice_metrics", [])
        # Build lookup
        metric_by_index = {m["slice_index"]: m for m in metrics_list}
    else:
        worst = None
        metric_by_index = {}

    image = pred = ref = None
    image_img = pred_img = ref_img = None
    if image_path is not None:
        image, image_img = _load_volume(image_path)
    if prediction_path is not None:
        pred, pred_img = _load_volume(prediction_path)
    if reference_path is not None:
        ref, ref_img = _load_volume(reference_path)

    if pred_img is not None and ref_img is not None:
        if image_img is not None:
            _check_geometry(image_img, pred_img, ref_img)
        else:
            if pred_img.shape != ref_img.shape:
                raise ValueError("shape mismatch: pred%s ref%s" % (pred_img.shape, ref_img.shape))
            if not np.allclose(pred_img.affine, ref_img.affine, atol=1e-5):
                raise ValueError("affine mismatch between prediction and ground-truth label")

    if worst is None:
        if pred is None or ref is None:
            raise ValueError("prediction and reference volumes are required")
        from spleen_evaluation import worst_slices as _worst

        worst = _worst((pred, ref), k=k)
        for m in worst:
            metric_by_index.setdefault(m["slice_index"], m)

    if not worst:
        raise ValueError("no worst slices to render")

    n = len(worst)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = [axes]
    for row, item in enumerate(worst):
        idx = int(item["slice_index"])
        m = metric_by_index.get(idx, item)
        ct_slice = image[:, :, idx] if (image is not None and image.ndim == 3) else None
        p_slice = pred[:, :, idx] if (pred is not None and pred.ndim == 3) else np.zeros((1, 1))
        r_slice = ref[:, :, idx] if (ref is not None and ref.ndim == 3) else np.zeros((1, 1))
        if ct_slice is None:
            ct_slice = np.zeros_like(p_slice, dtype=float)

        rgb, tp, fp, fn = _overlay_rgb(ct_slice, p_slice, r_slice)

        ax0 = axes[row][0]
        ax0.imshow(_normalize_slice(ct_slice), cmap="gray")
        ax0.set_title(f"Slice {idx} CT")
        ax0.axis("off")

        ax1 = axes[row][1]
        ax1.imshow(rgb)
        ax1.set_title("TP (g) / FP (r) / FN (b)")
        ax1.axis("off")

        ax2 = axes[row][2]
        ax2.axis("off")
        ax2.text(
            0.0,
            1.0,
            "Metrics\n"
            f"TP={m.get('tp')} FP={m.get('fp')} FN={m.get('fn')} TN={m.get('tn')}\n"
            f"Dice={m.get('dice'):.4f}\n"
            f"Prec={m.get('precision'):.4f}\n"
            f"Recall={m.get('recall'):.4f}\n"
            f"Reason: {m.get('reason', '')}",
            va="top",
            fontsize=9,
            family="monospace",
        )

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return str(out)


def main():
    parser = argparse.ArgumentParser(description="Offline spleen research viewer (worst slices).")
    parser.add_argument("--image", help="Path to image NIfTI")
    parser.add_argument("--prediction", help="Path to prediction NIfTI")
    parser.add_argument("--reference", help="Path to reference/label NIfTI")
    parser.add_argument("--slice-metrics", help="Path to saved slice_metrics JSON")
    parser.add_argument("--k", type=int, default=5, help="Number of worst slices")
    parser.add_argument("--output", default="worst_slices.png", help="Output PNG path")
    parser.add_argument("--title", default="Spleen worst slices (research only)")
    args = parser.parse_args()

    out = render_worst_slices(
        image_path=args.image,
        prediction_path=args.prediction,
        reference_path=args.reference,
        slice_metrics_json=args.slice_metrics,
        k=args.k,
        output_path=args.output,
        title=args.title,
    )
    print("Wrote", out)


if __name__ == "__main__":
    main()