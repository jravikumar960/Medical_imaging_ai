"""Image preparation and Grad-CAM helpers for the local CXR research demo."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def prepare_image(pil_image: Image.Image):
    """Apply TorchXRayVision-style 8-bit normalization, center crop and resize.

    Non-square images are center-cropped to a square before resizing to 224x224.
    Square images are resized directly. Center cropping removes outer border
    pixels prior to scaling, which reduces measured border activation on
    non-square inputs. Square images are unchanged by cropping (no-op).
    """
    oriented = ImageOps.exif_transpose(pil_image)
    if oriented.mode in {"I;16", "I;16B", "I;16L"}:
        pixels = np.asarray(oriented, dtype=np.float32)
        low, high = np.percentile(pixels, (0.5, 99.5))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError("16-bit grayscale image has no usable intensity range.")
        gray = np.clip((pixels - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
        image = Image.fromarray(gray)
    elif oriented.mode in {"I", "F"}:
        raise ValueError("Only 8-bit PNG/JPEG or decoded 8-bit DICOM pixels are supported.")
    else:
        image = oriented.convert("L")
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError("Image has no pixels.")
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    image = image.crop((left, top, left + side, top + side)).resize(
        (224, 224), Image.Resampling.BILINEAR
    )
    pixels = np.asarray(image, dtype=np.float32)
    normalized = (pixels / 255.0 - 0.5) * 2048.0
    return torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0), image


def open_uploaded_image(uploaded_file, expected_format: str | None = None) -> Image.Image:
    """Decode one still raster image, optionally requiring a specific file format."""
    image = Image.open(uploaded_file)
    if expected_format is not None and image.format != expected_format:
        raise ValueError(f"Expected a {expected_format} image.")
    if getattr(image, "n_frames", 1) != 1:
        raise ValueError("Animated or multi-frame raster images are not supported.")
    image.verify()
    uploaded_file.seek(0)
    decoded = Image.open(uploaded_file)
    decoded.load()
    return decoded


def gradcam_layer(model, layer_name: str = "denseblock4"):
    """Return DenseNet-121's dense-block convolution for spatial CAMs."""
    try:
        if layer_name == "denseblock4":
            layer = model.features.denseblock4.denselayer16.conv2
        elif layer_name == "denseblock3":
            layer = model.features.denseblock3.denselayer12.conv2
        else:
            # Fallback for toy models or generic structure
            layer = getattr(getattr(getattr(model.features, layer_name, None), "denselayer16", None), "conv2", None)
            if layer is None:
                layer = model.features.denseblock4.denselayer16.conv2
    except AttributeError as exc:
        raise RuntimeError(f"Could not resolve Grad-CAM target layer: {layer_name}.") from exc
    if not isinstance(layer, torch.nn.Module):
        raise RuntimeError("The selected Grad-CAM layer is not a valid module.")
    return layer


def gradcam(model, input_tensor, target_index, layer_name: str = "denseblock4", return_raw: bool = False):
    layer = gradcam_layer(model, layer_name=layer_name)
    activations, gradients = [], []

    def save_activation(_module, _inputs, output):
        activations.append(output)

    def save_gradient(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = backward_handle = None
    try:
        forward_handle = layer.register_forward_hook(save_activation)
        backward_handle = layer.register_full_backward_hook(save_gradient)
        model.zero_grad(set_to_none=True)
        output = model(input_tensor.detach().clone().requires_grad_(True))
        if output.ndim != 2 or not 0 <= target_index < output.shape[1]:
            raise IndexError("Selected finding index is outside model output.")
        score = output[0, target_index]
        score.backward()
        if not activations or not gradients:
            raise RuntimeError("Could not collect Grad-CAM activations and gradients.")
        weights = gradients[-1].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations[-1]).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
        raw_cam = cam.detach().cpu().numpy()
        if not np.isfinite(raw_cam).all():
            raw_cam = np.nan_to_num(raw_cam, nan=0.0, posinf=0.0, neginf=0.0)
        spread = float(raw_cam.max() - raw_cam.min())
        if spread > 0:
            norm_cam = (raw_cam - raw_cam.min()) / spread
        else:
            norm_cam = np.zeros_like(raw_cam)
        if return_raw:
            return norm_cam, raw_cam, float(score.detach().cpu())
        return norm_cam, float(score.detach().cpu())
    finally:
        if forward_handle is not None:
            forward_handle.remove()
        if backward_handle is not None:
            backward_handle.remove()


def gradcam_detailed(model, input_tensor, target_index, layer_name: str = "denseblock4"):
    """Compute Grad-CAM and return (normalized_cam, raw_cam, score)."""
    return gradcam(model, input_tensor, target_index, layer_name=layer_name, return_raw=True)


def overlay_heatmap(gray_image: Image.Image, cam: np.ndarray):
    """Build a matplotlib figure overlaying a normalized CAM on the preview image."""
    if cam.shape != (224, 224) or not np.isfinite(cam).all():
        raise ValueError("Heatmap must be a finite 224x224 array.")
    base = np.asarray(gray_image.resize((224, 224)), dtype=np.float32) / 255.0
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(base, cmap="gray")
    ax.imshow(cam, cmap="jet", alpha=0.38, vmin=0, vmax=1)
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig
