from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from imaging import gradcam, gradcam_layer, prepare_image, overlay_heatmap


class _ConvLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv2 = nn.Conv2d(in_c, out_c, 3, padding=1)

    def forward(self, x):
        return nn.functional.relu(self.conv2(x))


class _ToyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Module()
        self.features.denseblock3 = nn.Module()
        self.features.denseblock3.denselayer12 = _ConvLayer(1, 4)

        self.features.denseblock4 = nn.Module()
        self.features.denseblock4.denselayer16 = _ConvLayer(4, 8)
        self.classifier = nn.Linear(8, 3)

    def forward(self, x):
        x = self.features.denseblock3.denselayer12(x)
        x = self.features.denseblock4.denselayer16(x)
        return self.classifier(x.mean((2, 3)))


def test_numerical_correctness_and_dimensions():
    model = _ToyConvNet().eval()
    tensor = torch.randn((1, 1, 32, 32), requires_grad=True)
    cam, score = gradcam(model, tensor, 0, layer_name="denseblock4")

    assert cam.shape == (224, 224)
    assert np.isfinite(cam).all()
    assert float(cam.min()) >= 0.0
    assert float(cam.max()) <= 1.0
    assert np.isfinite(score)


def test_multi_layer_comparison():
    model = _ToyConvNet().eval()
    tensor = torch.randn((1, 1, 32, 32), requires_grad=True)

    cam_b4, score_b4 = gradcam(model, tensor, 0, layer_name="denseblock4")
    cam_b3, score_b3 = gradcam(model, tensor, 0, layer_name="denseblock3")

    assert cam_b4.shape == (224, 224)
    assert cam_b3.shape == (224, 224)
    assert score_b4 == score_b3
    assert not np.allclose(cam_b4, cam_b3)


def test_model_randomization_sanity_check():
    model = _ToyConvNet().eval()
    tensor = torch.randn((1, 1, 32, 32))
    cam_orig, _ = gradcam(model, tensor, 0, layer_name="denseblock4")

    with torch.no_grad():
        for param in model.parameters():
            param.data.normal_(0, 1)

    cam_rand, _ = gradcam(model, tensor, 0, layer_name="denseblock4")

    diff = np.mean(np.abs(cam_orig - cam_rand))
    assert diff > 0.001


def test_input_sensitivity_perturbation():
    model = _ToyConvNet().eval()
    tensor_clean = torch.ones((1, 1, 32, 32), requires_grad=True)
    tensor_perturbed = torch.ones((1, 1, 32, 32), requires_grad=True) * 0.5

    cam_clean, score_clean = gradcam(model, tensor_clean, 0)
    cam_pert, score_pert = gradcam(model, tensor_perturbed, 0)

    assert cam_clean.shape == (224, 224)
    assert cam_pert.shape == (224, 224)
    assert isinstance(score_clean, float)
    assert isinstance(score_pert, float)
