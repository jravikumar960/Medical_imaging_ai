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

from imaging import gradcam, overlay_heatmap, prepare_image
from xray_gradcam_localization import border_mask, mass_fraction, occlude_image, region_mass_fractions

ROOT = Path("outputs/xray_gradcam/robustness_audit")
ROOT.mkdir(parents=True, exist_ok=True)
VALIDATED = Path("outputs/xray_gradcam/validated_lung_masks")
classifier = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
labels = list(classifier.pathologies)

images = [Path("outputs/first_inference/chest_xray_kaggle_normal.jpeg")]
images.extend(sorted(Path("outputs/xray_labeled_evaluation/images").glob("*.png"))[:11])

records = []
for image_path in images:
    with Image.open(image_path) as source:
        source.load()
        original = source.copy().convert("L")
    w, h = original.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    cropped = original.crop((left, top, left + side, top + side))

    lung_mask = np.asarray(Image.open(VALIDATED / f"{image_path.stem}_lung_mask.png").convert("L")) > 127

    classifier_tensor, classifier_cropped = prepare_image(original)
    with torch.inference_mode():
        baseline_output = classifier(classifier_tensor)
    baseline_scores = baseline_output[0].detach().cpu().numpy().astype(float)
    top_index = int(np.argmax(baseline_scores))
    selected_label = labels[top_index]
    cam, cam_score = gradcam(classifier, classifier_tensor, top_index)

    border_8 = border_mask((224, 224), 8)
    border_16 = border_mask((224, 224), 16)
    border_24 = border_mask((224, 224), 24)
    region_fractions = region_mass_fractions(cam, lung_mask, border_width=16)

    def score_and_top(occluded_pil):
        occluded_tensor, _ = prepare_image(occluded_pil)
        with torch.inference_mode():
            out = classifier(occluded_tensor)
        scores = out[0].detach().cpu().numpy().astype(float)
        return float(scores[top_index]), labels[int(np.argmax(scores))]

    occlusion_results = {}
    for method in ("zero", "local_mean", "gaussian_blur"):
        for region_name, mask in (("lung", lung_mask), ("border", border_16)):
            occluded = occlude_image(classifier_cropped, mask, method=method, fill_value=0, sigma=8.0)
            score, top_label = score_and_top(occluded)
            occlusion_results[f"{method}_{region_name}"] = {
                "selected_score": score,
                "selected_score_delta": float(score - baseline_scores[top_index]),
                "top_prediction": top_label,
                "top_prediction_changed": bool(top_label != selected_label),
            }

    stem = image_path.stem
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(np.asarray(classifier_cropped), cmap="gray")
    axes[0].set_title("input")
    axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(lung_mask, cmap="gray")
    axes[2].set_title("validated lung mask")
    zero_lung = occlude_image(classifier_cropped, lung_mask, method="zero")
    axes[3].imshow(np.asarray(zero_lung), cmap="gray")
    axes[3].set_title("lung occluded")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / f"{stem}_comparison.png", dpi=120)
    plt.close(fig)

    record = {
        "source_image": str(image_path),
        "original_size": list(original.size),
        "selected_label": selected_label,
        "selected_score": float(baseline_scores[top_index]),
        "score_type": "operating_point_normalized_score",
        "border_mass_fractions": {
            "8": mass_fraction(cam, border_mask((224, 224), 8)),
            "16": mass_fraction(cam, border_mask((224, 224), 16)),
            "24": mass_fraction(cam, border_mask((224, 224), 24)),
        },
        "region_mass_fractions_16px_border": region_fractions,
        "lung_mask_area_fraction": float(lung_mask.mean()),
        "occlusion": occlusion_results,
        "top_predictions": [{"finding": label, "score": float(score)} for label, score in sorted(zip(labels, baseline_scores), key=lambda item: item[1], reverse=True)[:5]],
        "artifacts": {"comparison": str(ROOT / f"{stem}_comparison.png")},
    }
    records.append(record)
    print(image_path.name, selected_label, round(record["selected_score"], 5), "lung_mass", round(region_fractions["inside_lung"], 4), "border_8", round(record["border_mass_fractions"]["8"], 4), "lung_delta_zero", round(occlusion_results["zero_lung"]["selected_score_delta"], 5))

import numpy as np

aggregate = {
    "n_images": len(records),
    "mean_lung_mass_fraction": float(np.mean([r["region_mass_fractions_16px_border"]["inside_lung"] for r in records])),
    "mean_outside_lung_fraction": float(np.mean([r["region_mass_fractions_16px_border"]["outside_lung"] for r in records])),
    "mean_border_8_mass_fraction": float(np.mean([r["border_mass_fractions"]["8"] for r in records])),
    "mean_border_16_mass_fraction": float(np.mean([r["border_mass_fractions"]["16"] for r in records])),
    "mean_border_24_mass_fraction": float(np.mean([r["border_mass_fractions"]["24"] for r in records])),
    "mean_lung_mask_area_fraction": float(np.mean([r["lung_mask_area_fraction"] for r in records])),
    "mean_zero_lung_occlusion_delta": float(np.mean([r["occlusion"]["zero_lung"]["selected_score_delta"] for r in records])),
    "mean_zero_border_occlusion_delta": float(np.mean([r["occlusion"]["zero_border"]["selected_score_delta"] for r in records])),
    "mean_local_mean_lung_occlusion_delta": float(np.mean([r["occlusion"]["local_mean_lung"]["selected_score_delta"] for r in records])),
    "mean_gaussian_lung_occlusion_delta": float(np.mean([r["occlusion"]["gaussian_blur_lung"]["selected_score_delta"] for r in records])),
    "top_prediction_changes_after_lung_occlusion": {
        "zero": sum(1 for r in records if r["occlusion"]["zero_lung"]["top_prediction_changed"]),
        "local_mean": sum(1 for r in records if r["occlusion"]["local_mean_lung"]["top_prediction_changed"]),
        "gaussian_blur": sum(1 for r in records if r["occlusion"]["gaussian_blur_lung"]["top_prediction_changed"]),
    },
    "outlier_images_by_validated_lung_mass": [r["source_image"] for r in records if r["region_mass_fractions_16px_border"]["inside_lung"] < 0.1 or r["region_mass_fractions_16px_border"]["inside_lung"] > 0.6],
}
(ROOT / "gradcam_robustness_audit.json").write_text(json.dumps({"records": records, "aggregate": aggregate}, indent=2), encoding="utf-8")
print("aggregate", aggregate)
print("wrote", ROOT / "gradcam_robustness_audit.json")
