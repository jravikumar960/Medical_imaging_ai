"""Unit tests for Grad-CAM layer comparison (DenseBlock3 vs DenseBlock4)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from imaging import gradcam, gradcam_layer, gradcam_detailed


class _MockConvLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv2 = nn.Conv2d(in_c, out_c, 3, padding=1)

    def forward(self, x):
        return nn.functional.relu(self.conv2(x))


class _MockDenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pathologies = ["FindingA", "FindingB"]
        self.features = nn.Module()
        self.features.denseblock3 = nn.Module()
        self.features.denseblock3.denselayer12 = _MockConvLayer(1, 4)

        self.features.denseblock4 = nn.Module()
        self.features.denseblock4.denselayer16 = _MockConvLayer(4, 8)
        self.classifier = nn.Linear(8, 2)

    def forward(self, x):
        x = self.features.denseblock3.denselayer12(x)
        x = self.features.denseblock4.denselayer16(x)
        return self.classifier(x.mean((2, 3)))


def test_layers_resolve_correctly():
    model = _MockDenseNet().eval()
    layer3 = gradcam_layer(model, "denseblock3")
    layer4 = gradcam_layer(model, "denseblock4")
    assert layer3 is model.features.denseblock3.denselayer12.conv2
    assert layer4 is model.features.denseblock4.denselayer16.conv2


def test_both_layers_produce_finite_correctly_shaped_cam():
    model = _MockDenseNet().eval()
    tensor = torch.randn((1, 1, 32, 32), requires_grad=True)

    cam3, score3 = gradcam(model, tensor, 0, layer_name="denseblock3")
    cam4, score4 = gradcam(model, tensor, 0, layer_name="denseblock4")

    assert cam3.shape == (224, 224)
    assert cam4.shape == (224, 224)
    assert np.isfinite(cam3).all()
    assert np.isfinite(cam4).all()
    assert np.isfinite(score3)
    assert np.isfinite(score4)


def test_target_finding_selection_works_with_either_layer():
    model = _MockDenseNet().eval()
    tensor = torch.randn((1, 1, 32, 32), requires_grad=True)

    for idx in [0, 1]:
        cam3, _ = gradcam(model, tensor, idx, layer_name="denseblock3")
        cam4, _ = gradcam(model, tensor, idx, layer_name="denseblock4")
        assert cam3.shape == (224, 224)
        assert cam4.shape == (224, 224)


def test_prediction_scores_unchanged_when_layer_changes():
    model = _MockDenseNet().eval()
    tensor = torch.randn((1, 1, 32, 32), requires_grad=True)

    _, score3 = gradcam(model, tensor, 0, layer_name="denseblock3")
    _, score4 = gradcam(model, tensor, 0, layer_name="denseblock4")
    assert score3 == score4


def test_constant_or_invalid_cam_outputs_handled_safely():
    # Test that gradcam_detailed returns finite arrays even when raw gradients are zero/constant
    model = _MockDenseNet().eval()
    tensor = torch.zeros((1, 1, 32, 32), requires_grad=True)
    norm_cam, raw_cam, score = gradcam_detailed(model, tensor, 0, layer_name="denseblock4")
    assert norm_cam.shape == (224, 224)
    assert raw_cam.shape == (224, 224)
    assert np.isfinite(norm_cam).all()
    assert np.isfinite(raw_cam).all()
