import json
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image

from imaging import gradcam, prepare_image
from xray_gradcam_localization import border_mask, mass_fraction

ROOT = Path("outputs/xray_gradcam/outlier_investigation")
ROOT.mkdir(parents=True, exist_ok=True)
VALIDATED = Path("outputs/xray_gradcam/validated_lung_masks")
ROBUST = json.loads(Path("outputs/xray_gradcam/robustness_audit/gradcam_robustness_audit.json").read_text())

outliers = ["1021_0", "1028_0", "1063_0", "1106_0"]
classifier = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
labels = list(classifier.pathologies)


def quadrant_masses(cam):
    h, w = cam.shape
    mid_h, mid_w = h // 2, w // 2
    return {
        "top_left": float(cam[:mid_h, :mid_w].sum()),
        "top_right": float(cam[:mid_h, mid_w:].sum()),
        "bottom_left": float(cam[mid_h:, :mid_w].sum()),
        "bottom_right": float(cam[mid_h:, mid_w:].sum()),
    }


def component_summary(mask):
    from skimage.measure import label

    labeled = label(mask)
    areas = sorted((int((labeled == i).sum()) for i in range(1, labeled.max() + 1)), reverse=True)
    ys, xs = np.where(mask)
    centroid = (float(xs.mean()) if len(xs) else None, float(ys.mean()) if len(ys) else None)
    touches_border = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    return {
        "component_count": int(labeled.max()),
        "largest_components": areas[:5],
        "centroid_x": centroid[0],
        "centroid_y": centroid[1],
        "touches_border": touches_border,
        "mask_area_fraction": float(mask.mean()),
    }


records = []
fig, axes = plt.subplots(len(outliers), 4, figsize=(12, 3 * len(outliers)))
for row, stem in enumerate(outliers):
    image_path = Path("outputs/xray_labeled_evaluation/images") / f"{stem}.png"
    with Image.open(image_path) as source:
        source.load()
        original = source.copy().convert("L")
    w, h = original.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    cropped = original.crop((left, top, left + side, top + side))

    classifier_tensor, classifier_cropped = prepare_image(original)
    with torch.inference_mode():
        output = classifier(classifier_tensor)
    scores = output[0].detach().cpu().numpy().astype(float)
    top_index = int(np.argmax(scores))
    cam, cam_score = gradcam(classifier, classifier_tensor, top_index)

    lung_mask = np.asarray(Image.open(VALIDATED / f"{stem}_lung_mask.png").convert("L")) > 127
    border8 = border_mask((224, 224), 8)
    border16 = border_mask((224, 224), 16)
    border24 = border_mask((224, 224), 24)

    total_mass = float(cam.sum())
    q = quadrant_masses(cam)
    q_fraction = {k: (v / total_mass if total_mass else 0.0) for k, v in q.items()}
    max_loc = np.unravel_index(int(np.argmax(cam)), cam.shape)

    robust_record = next(r for r in ROBUST["records"] if Path(r["source_image"]).stem == stem)
    record = {
        "image": str(image_path),
        "original_size": list(original.size),
        "cropped_size": list(cropped.size),
        "selected_label": labels[top_index],
        "selected_score": float(scores[top_index]),
        "score_type": "operating_point_normalized_score",
        "gradcam_target_index": top_index,
        "gradcam_score": float(cam_score),
        "heatmap_shape": list(cam.shape),
        "lung_mask": component_summary(lung_mask),
        "gradcam_mass": {
            "inside_lung": mass_fraction(cam, lung_mask),
            "outside_lung": mass_fraction(cam, ~lung_mask),
            "border_8": mass_fraction(cam, border8),
            "border_16": mass_fraction(cam, border16),
            "border_24": mass_fraction(cam, border24),
            "border_and_lung": mass_fraction(cam, border16 & lung_mask),
            "border_and_non_lung": mass_fraction(cam, border16 & ~lung_mask),
            "max_location_yx": [int(max_loc[0]), int(max_loc[1])],
            "quadrant_fractions": q_fraction,
        },
        "occlusion": robust_record["occlusion"],
        "top_predictions": robust_record["top_predictions"],
    }
    records.append(record)

    axes[row, 0].imshow(np.asarray(cropped), cmap="gray")
    axes[row, 0].set_title(f"{stem}: input")
    axes[row, 1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[row, 1].set_title("Grad-CAM")
    axes[row, 2].imshow(lung_mask, cmap="gray")
    axes[row, 2].set_title("lung mask")
    axes[row, 3].imshow(np.asarray(cropped), cmap="gray")
    axes[row, 3].imshow(cam, cmap="jet", alpha=0.35, vmin=0, vmax=1)
    axes[row, 3].set_title("overlay")
    for ax in axes[row]:
        ax.axis("off")

fig.tight_layout()
fig.savefig(ROOT / "outlier_comparison_grid.png", dpi=150)
plt.close(fig)

other_records = [r for r in ROBUST["records"] if Path(r["source_image"]).stem not in outliers]
comparison = {
    "outliers": {
        "n": len(records),
        "mean_lung_mass_fraction": float(np.mean([r["gradcam_mass"]["inside_lung"] for r in records])),
        "mean_border_16_fraction": float(np.mean([r["gradcam_mass"]["border_16"] for r in records])),
        "mean_lung_mask_area_fraction": float(np.mean([r["lung_mask"]["mask_area_fraction"] for r in records])),
    },
    "others": {
        "n": len(other_records),
        "mean_lung_mass_fraction": float(np.mean([r["region_mass_fractions_16px_border"]["inside_lung"] for r in other_records])),
        "mean_border_16_fraction": float(np.mean([r["border_mass_fractions"]["16"] for r in other_records])),
        "mean_lung_mask_area_fraction": float(np.mean([r["lung_mask_area_fraction"] for r in other_records])),
    },
}
(ROOT / "outlier_investigation.json").write_text(json.dumps({"outliers": records, "comparison": comparison}, indent=2), encoding="utf-8")
print("wrote json")
