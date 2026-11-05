import base64
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image

from imaging import gradcam, overlay_heatmap, prepare_image

ROOT = Path("outputs/xray_gradcam")
ROOT.mkdir(parents=True, exist_ok=True)
OUT = ROOT
model = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
labels = list(model.pathologies)

images = [Path("outputs/first_inference/chest_xray_kaggle_normal.jpeg")]
images.extend(sorted(Path("outputs/xray_labeled_evaluation/images").glob("*.png"))[:3])

records = []
for image_path in images:
    with Image.open(image_path) as source:
        source.load()
        original = source.copy()
    tensor, cropped = prepare_image(original)
    with torch.inference_mode():
        output = model(tensor)
    scores = output[0].detach().cpu().numpy().astype(float)
    top_index = int(np.argmax(scores))
    cam, cam_score = gradcam(model, tensor, top_index)
    selected_label = labels[top_index]
    selected_score = float(scores[top_index])

    stem = image_path.stem
    original_path = OUT / f"{stem}_original.png"
    heatmap_path = OUT / f"{stem}_heatmap.png"
    overlay_path = OUT / f"{stem}_overlay.png"
    original.convert("RGB").save(original_path)

    plt.imsave(heatmap_path, cam, cmap="jet", vmin=0, vmax=1)
    fig = overlay_heatmap(cropped, cam)
    fig.savefig(overlay_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # Border-vs-center mass diagnostic. Border is 16 px on each side.
    border = np.zeros_like(cam, dtype=bool)
    border[:16, :] = True
    border[-16:, :] = True
    border[:, :16] = True
    border[:, -16:] = True
    center = ~border
    cam_mass = float(cam.sum())
    border_mass = float(cam[border].sum())
    center_mass = float(cam[center].sum())

    record = {
        "source_image": str(image_path),
        "original_size": list(original.size),
        "original_mode": original.mode,
        "cropped_size": list(cropped.size),
        "preprocessing": {
            "exif_transpose": True,
            "grayscale_conversion": "L",
            "center_crop_to_square": True,
            "resize": "224x224 bilinear",
            "normalization": "(pixels / 255.0 - 0.5) * 2048.0",
            "tensor_shape": list(tensor.shape),
        },
        "selected_label": selected_label,
        "selected_score": selected_score,
        "score_type": "operating_point_normalized_score",
        "gradcam_target_layer": "model.features.denseblock4.denselayer16.conv2",
        "gradcam_score": cam_score,
        "heatmap_shape": list(cam.shape),
        "heatmap_min": float(cam.min()),
        "heatmap_max": float(cam.max()),
        "border_mass_fraction": border_mass / cam_mass if cam_mass else 0.0,
        "center_mass_fraction": center_mass / cam_mass if cam_mass else 0.0,
        "top_predictions": [{"finding": label, "score": float(score)} for label, score in sorted(zip(labels, scores), key=lambda item: item[1], reverse=True)[:5]],
        "artifacts": {
            "original": str(original_path),
            "heatmap": str(heatmap_path),
            "overlay": str(overlay_path),
        },
    }
    records.append(record)
    print(image_path, original.size, selected_label, round(selected_score, 5), "border_mass_fraction", round(record["border_mass_fraction"], 4))

(ROOT / "gradcam_audit.json").write_text(__import__("json").dumps(records, indent=2), encoding="utf-8")
print("wrote", ROOT / "gradcam_audit.json")
