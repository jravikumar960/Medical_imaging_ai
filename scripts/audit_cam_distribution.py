import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image

from imaging import gradcam, prepare_image
from xray_gradcam_localization import border_mask, estimate_unvalidated_lung_mask, mass_fraction

model = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
image_paths = sorted(Path("outputs/xray_labeled_evaluation/images").glob("*.png"))[:10]

print(f"Auditing {len(image_paths)} images for Grad-CAM edge/border vs lung activation distribution:")
for img_path in image_paths:
    with Image.open(img_path) as img:
        img.load()
        original = img.copy()
    tensor, cropped = prepare_image(original)
    with torch.inference_mode():
        out = model(tensor)
    scores = out[0].detach().cpu().numpy()
    top_idx = int(np.argmax(scores))
    label = model.pathologies[top_idx]

    cam_b4, _ = gradcam(model, tensor, top_idx, layer_name="denseblock4")
    cam_b3, _ = gradcam(model, tensor, top_idx, layer_name="denseblock3")

    lung_mask = estimate_unvalidated_lung_mask(cropped)
    border_16 = border_mask((224, 224), 16)
    bottom_32 = np.zeros((224, 224), dtype=bool)
    bottom_32[-32:, :] = True

    b4_lung = mass_fraction(cam_b4, lung_mask)
    b4_border = mass_fraction(cam_b4, border_16)
    b4_bottom = mass_fraction(cam_b4, bottom_32)

    b3_lung = mass_fraction(cam_b3, lung_mask)
    b3_border = mass_fraction(cam_b3, border_16)
    b3_bottom = mass_fraction(cam_b3, bottom_32)

    print(f"Image: {img_path.name} | Label: {label} (score: {scores[top_idx]:.3f})")
    print(f"  DenseBlock4 -> Lung mass: {b4_lung:.3f}, Border (16px) mass: {b4_border:.3f}, Bottom (32px) mass: {b4_bottom:.3f}")
    print(f"  DenseBlock3 -> Lung mass: {b3_lung:.3f}, Border (16px) mass: {b3_border:.3f}, Bottom (32px) mass: {b3_bottom:.3f}")
