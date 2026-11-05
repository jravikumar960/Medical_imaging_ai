import csv
import numpy as np
import pytest
import torch
from PIL import Image
from pathlib import Path

from modality_training import IMAGE_SIZE, LABEL_TO_INDEX, build_model, build_transform
import modality_verifier


def _checkpoint(tmp_path, *, threshold=0.5, malformed=False):
    path = tmp_path / "modality_classifier.pt"
    if malformed:
        torch.save({"foo": "bar"}, path)
        return path
    model = build_model()
    model.eval()
    payload = {
        "state_dict": model.state_dict(),
        "threshold": threshold,
        "labels": LABEL_TO_INDEX,
        "architecture": "torchvision.mobilenet_v3_small",
        "input_size": IMAGE_SIZE,
        "preprocessing": {
            "resize": [IMAGE_SIZE, IMAGE_SIZE],
            "color": "RGB",
            "mean": [0.5, 0.5, 0.5],
            "std": [0.25, 0.25, 0.25],
        },
    }
    torch.save(payload, path)
    return path


def test_load_returns_none_when_checkpoint_missing(tmp_path):
    assert modality_verifier.load_modality_verifier(tmp_path / "missing.pt") is None


def test_load_returns_none_for_malformed_checkpoint(tmp_path):
    path = _checkpoint(tmp_path, malformed=True)
    assert modality_verifier.load_modality_verifier(path) is None


def test_load_validates_and_verifies_image(tmp_path, monkeypatch):
    path = _checkpoint(tmp_path, threshold=0.25)
    monkeypatch.setenv("CXR_MODALITY_CLASSIFIER_PATH", str(path))
    # Ensure default path resolution uses env var.
    verifier = modality_verifier.load_modality_verifier()
    assert verifier is not None
    assert verifier.threshold == 0.25

    image = Image.new("RGB", (64, 64), "gray")
    result = verifier.verify(image)
    assert isinstance(result.is_cxr, bool)
    assert 0.0 <= result.score <= 1.0
    assert result.threshold == 0.25


def test_verifier_accepts_supported_grayscale_rgb_and_resolutions(tmp_path):
    class ConfidentModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_shapes = []

        def forward(self, tensor):
            self.input_shapes.append(tuple(tensor.shape))
            return torch.tensor([[0.0, 5.0]])

    path = _checkpoint(tmp_path, threshold=0.99)
    verifier = modality_verifier.load_modality_verifier(path)
    assert verifier is not None
    model = ConfidentModel()
    verifier.model = model
    high_bit_pixels = np.linspace(1000, 50000, 512 * 384, dtype=np.uint16).reshape(384, 512)
    images = [
        Image.new("L", (64, 32), 120),
        Image.new("RGB", (1024, 768), (90, 120, 150)),
        Image.fromarray(high_bit_pixels),
    ]

    results = [verifier.verify(image) for image in images]

    assert all(result.is_cxr and result.score > 0.99 for result in results)
    assert model.input_shapes == [(1, 3, 224, 224)] * 3


def test_verifier_threshold_boundary_and_reviewed_099_operating_point(tmp_path):
    class FixedModel(torch.nn.Module):
        def forward(self, _tensor):
            return torch.tensor([[0.0, 4.59512]])

    image = Image.new("L", (300, 180), 120)
    path = _checkpoint(tmp_path, threshold=0.99)
    verifier = modality_verifier.load_modality_verifier(path)
    assert verifier is not None
    verifier.model = FixedModel()

    result = verifier.verify(image)

    assert result.score == pytest.approx(0.99, abs=1e-5)
    assert result.is_cxr is True
    verifier.threshold = result.score + 1e-6
    assert verifier.verify(image).is_cxr is False


def test_observed_0992_score_passes_reviewed_099_threshold_but_not_old_cutoff(tmp_path):
    class ObservedScoreModel(torch.nn.Module):
        def forward(self, _tensor):
            return torch.tensor([[0.0, float(torch.logit(torch.tensor(0.992))) ]])

    path = _checkpoint(tmp_path, threshold=0.99)
    verifier = modality_verifier.load_modality_verifier(path)
    assert verifier is not None
    verifier.model = ObservedScoreModel()
    image = Image.new("L", (320, 240), 120)

    result = verifier.verify(image)

    assert result.score == pytest.approx(0.992, abs=1e-6)
    assert result.is_cxr is True
    verifier.threshold = 0.9998095631599426
    assert verifier.verify(image).is_cxr is False


def test_shipped_verifier_rejects_labeled_non_xray_categories_and_accepts_cxr():
    dataset_root = Path(__file__).resolve().parents[1] / "outputs" / "xray_verifier_dataset"
    manifest_path = dataset_root / "manifest.csv"
    if not manifest_path.is_file():
        pytest.skip("The labeled verifier evaluation collection is not installed.")
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    selected = [next(row for row in rows if row["label"] == "cxr")]
    for category in ("cars", "buildings", "people", "everyday_objects"):
        selected.append(
            next(
                row for row in rows
                if row["label"] == "non_cxr" and row["category"] == category
            )
        )

    verifier = modality_verifier.load_modality_verifier()
    assert verifier is not None
    assert verifier.threshold == 0.99
    for row in selected:
        with Image.open(dataset_root / row["image_path"]) as image:
            result = verifier.verify(image)
        assert result.is_cxr is (row["label"] == "cxr"), row["image_id"]


def test_verify_rejects_invalid_model_output(tmp_path, monkeypatch):
    path = _checkpoint(tmp_path)
    verifier = modality_verifier.load_modality_verifier(path)
    assert verifier is not None

    class BadModel(torch.nn.Module):
        def __call__(self, _tensor):
            return torch.tensor([[float("nan"), 0.0]])

    verifier.model = BadModel()
    with pytest.raises(ValueError, match="non-finite"):
        verifier.verify(Image.new("RGB", (8, 8), "white"))