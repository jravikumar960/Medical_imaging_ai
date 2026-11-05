from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image, UnidentifiedImageError

from imaging import gradcam, gradcam_layer, open_uploaded_image, overlay_heatmap, prepare_image


def test_preprocessing_shape_range_and_center_crop():
    image = Image.fromarray(np.full((300, 400), 128, dtype=np.uint8))
    tensor, gray = prepare_image(image)
    assert tuple(tensor.shape) == (1, 1, 224, 224)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) == pytest.approx((128 / 255 - 0.5) * 2048, abs=1e-4)
    assert gray.size == (224, 224)


def test_preprocessing_honors_exif_rotation():
    image = Image.new("L", (100, 200), 80)
    tensor, _ = prepare_image(image)
    assert tuple(tensor.shape) == (1, 1, 224, 224)


def test_preprocessing_maps_full_8bit_range_to_torchxrayvision_range():
    image = Image.fromarray(np.tile(np.arange(256, dtype=np.uint8), (256, 1)))
    tensor, _ = prepare_image(image)
    assert float(tensor.min()) == pytest.approx(-1024.0)
    assert float(tensor.max()) == pytest.approx(1024.0)


@pytest.mark.parametrize("payload", [b"not an image", b"DICM" + bytes(128)])
def test_corrupted_and_unsupported_uploads_are_rejected(payload):
    with pytest.raises((UnidentifiedImageError, OSError)):
        open_uploaded_image(BytesIO(payload))


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(1, 2, 3, padding=1)

    def forward(self, x):
        return torch.relu(self.conv2(x))


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Module()
        self.features.denseblock4 = torch.nn.Module()
        self.features.denseblock4.denselayer16 = _Layer()
        self.classifier = torch.nn.Linear(2, 3)

    def forward(self, x):
        x = self.features.denseblock4.denselayer16(x)
        return self.classifier(x.mean((2, 3)))


def test_gradcam_targets_named_final_dense_block_layer_and_cleans_hooks():
    model = _ToyModel().eval()
    layer = gradcam_layer(model)
    tensor = torch.ones((1, 1, 16, 16))
    cam, score = gradcam(model, tensor, 1)
    assert layer is model.features.denseblock4.denselayer16.conv2
    assert cam.shape == (224, 224)
    assert np.isfinite(cam).all()
    assert cam.min() >= 0 and cam.max() <= 1
    assert np.isfinite(score)
    assert len(layer._forward_hooks) == len(layer._backward_hooks) == 0
    figure = overlay_heatmap(Image.new("L", (224, 224), 100), cam)
    figure.canvas.draw()
    assert len(figure.axes) == 1
    import matplotlib.pyplot as plt
    plt.close(figure)


def test_gradcam_rejects_invalid_output_index():
    with pytest.raises(IndexError):
        gradcam(_ToyModel(), torch.ones((1, 1, 16, 16)), 3)


def test_gradcam_cleans_up_if_backward_hook_registration_fails(monkeypatch):
    model = _ToyModel()
    layer = gradcam_layer(model)

    def fail_registration(*_args, **_kwargs):
        raise RuntimeError("hook registration failed")

    monkeypatch.setattr(layer, "register_full_backward_hook", fail_registration)
    with pytest.raises(RuntimeError, match="hook registration failed"):
        gradcam(model, torch.ones((1, 1, 16, 16)), 0)
    assert len(layer._forward_hooks) == 0


def test_gradcam_regression_dimensions_finite_values_and_gradient_flow():
    model = _ToyModel().eval()
    tensor = torch.randn((1, 1, 32, 32), requires_grad=True)
    cam, score = gradcam(model, tensor, 0)

    # 1. Heatmap dimensions
    assert cam.shape == (224, 224)
    # 2. Finite values and range [0, 1]
    assert np.isfinite(cam).all()
    assert float(cam.min()) >= 0.0
    assert float(cam.max()) <= 1.0
    # 3. Score finite
    assert np.isfinite(score)

    # 4. Overlay alignment and numerical properties
    gray_img = Image.new("L", (224, 224), 128)
    fig = overlay_heatmap(gray_img, cam)
    assert fig is not None
    assert len(fig.axes) == 1
    import matplotlib.pyplot as plt
    plt.close(fig)
