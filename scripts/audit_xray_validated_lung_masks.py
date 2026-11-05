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
from PIL import Image, ImageOps

from imaging import gradcam, overlay_heatmap, prepare_image
from xray_gradcam_localization import border_mask, mass_fraction, occlude_image

ROOT = Path("outputs/xray_gradcam/validated_lung_masks")
ROOT.mkdir(parents=True, exist_ok=True)
segmentation_model = xrv.baseline_models.chestx_det.PSPNet()
segmentation_model.eval()
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

    arr = np.asarray(cropped, dtype=np.float32)
    tensor = torch.from_numpy((arr / 255.0 - 0.5) * 2048.0).unsqueeze(0).unsqueeze(0)
    with torch.inference_mode():
        segmentation_output = segmentation_model(tensor)
    logits = segmentation_output[0].detach().cpu().numpy()
    lung_mask_512 = (logits[4] > 0) | (logits[5] > 0)
    lung_mask_img = Image.fromarray((lung_mask_512.astype(np.uint8) * 255))
    lung_mask = np.asarray(lung_mask_img.resize((224, 224), Image.Resampling.NEAREST), dtype=np.uint8) > 0

    classifier_tensor, classifier_cropped = prepare_image(original)
    with torch.inference_mode():
        baseline_output = classifier(classifier_tensor)
    baseline_scores = baseline_output[0].detach().cpu().numpy().astype(float)
    top_index = int(np.argmax(baseline_scores))
    selected_label = labels[top_index]
    cam, cam_score = gradcam(classifier, classifier_tensor, top_index)

    border = border_mask((224, 224), 16)
    lung_fraction = mass_fraction(cam, lung_mask)
    outside_fraction = mass_fraction(cam, ~lung_mask)
    border_fraction = mass_fraction(cam, border)

    stem = image_path.stem
    cropped_path = ROOT / f"{stem}_cropped.png"
    heatmap_path = ROOT / f"{stem}_heatmap.png"
    overlay_path = ROOT / f"{stem}_overlay.png"
    lung_mask_path = ROOT / f"{stem}_lung_mask.png"
    occluded_lung_path = ROOT / f"{stem}_occluded_lung.png"
    occluded_border_path = ROOT / f"{stem}_occluded_border.png"

    classifier_cropped.save(cropped_path)
    plt.imsave(heatmap_path, cam, cmap="jet", vmin=0, vmax=1)
    fig = overlay_heatmap(classifier_cropped, cam)
    fig.savefig(overlay_path, dpi=120, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    plt.imsave(lung_mask_path, lung_mask.astype(np.uint8) * 255, cmap="gray", vmin=0, vmax=255)

    def score_for(occluded_pil):
        occluded_tensor, _ = prepare_image(occluded_pil)
        with torch.inference_mode():
            out = classifier(occluded_tensor)
        return float(out[0, top_index].detach().cpu())

    lung_occluded_image = occlude_image(classifier_cropped, lung_mask, fill_value=0)
    border_occluded_image = occlude_image(classifier_cropped, border, fill_value=0)
    lung_occluded_image.save(occluded_lung_path)
    border_occluded_image.save(occluded_border_path)
    lung_score = score_for(lung_occluded_image)
    border_score = score_for(border_occluded_image)

    record = {
        "source_image": str(image_path),
        "original_size": list(original.size),
        "cropped_size": [224, 224],
        "lung_segmentation_model": "torchxrayvision.baseline_models.chestx_det.PSPNet",
        "lung_segmentation_targets": ["Left Lung", "Right Lung"],
        "lung_mask_threshold": "logit > 0 (sigmoid probability > 0.5)",
        "lung_mask_area_fraction": float(lung_mask.mean()),
        "selected_label": selected_label,
        "selected_score": float(baseline_scores[top_index]),
        "score_type": "operating_point_normalized_score",
        "gradcam_mass_inside_lung_fraction": lung_fraction,
        "gradcam_mass_outside_lung_fraction": outside_fraction,
        "gradcam_mass_border_fraction": border_fraction,
        "occlusion": {
            "baseline_selected_score": float(baseline_scores[top_index]),
            "lung_region_occluded_selected_score": lung_score,
            "border_region_occluded_selected_score": border_score,
            "lung_occlusion_score_delta": float(lung_score - baseline_scores[top_index]),
            "border_occlusion_score_delta": float(border_score - baseline_scores[top_index]),
            "fill_value": 0,
            "occlusion_applied_to": "224x224 model input after standard center crop and resize",
        },
        "top_predictions": [{"finding": label, "score": float(score)} for label, score in sorted(zip(labels, baseline_scores), key=lambda item: item[1], reverse=True)[:5]],
        "artifacts": {
            "cropped": str(cropped_path),
            "heatmap": str(heatmap_path),
            "overlay": str(overlay_path),
            "lung_mask": str(lung_mask_path),
            "occluded_lung": str(occluded_lung_path),
            "occluded_border": str(occluded_border_path),
        },
    }
    records.append(record)
    print(image_path.name, selected_label, round(record["selected_score"], 5), "lung_mass", round(lung_fraction, 4), "border_mass", round(border_fraction, 4), "lung_delta", round(record["occlusion"]["lung_occlusion_score_delta"], 5))

import numpy as np

aggregate = {
    "n_images": len(records),
    "mean_lung_mass_fraction": float(np.mean([r["gradcam_mass_inside_lung_fraction"] for r in records])),
    "mean_outside_lung_fraction": float(np.mean([r["gradcam_mass_outside_lung_fraction"] for r in records])),
    "mean_border_mass_fraction": float(np.mean([r["gradcam_mass_border_fraction"] for r in records])),
    "mean_lung_mask_area_fraction": float(np.mean([r["lung_mask_area_fraction"] for r in records])),
    "mean_lung_occlusion_delta": float(np.mean([r["occlusion"]["lung_occlusion_score_delta"] for r in records])),
    "mean_border_occlusion_delta": float(np.mean([r["occlusion"]["border_occlusion_score_delta"] for r in records])),
}
(ROOT / "validated_lung_mask_audit.json").write_text(json.dumps({"records": records, "aggregate": aggregate}, indent=2), encoding="utf-8")
print("aggregate", aggregate)
print("wrote", ROOT / "validated_lung_mask_audit.json")
