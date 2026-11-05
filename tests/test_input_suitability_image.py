from io import BytesIO

import pytest
from PIL import Image

import input_suitability
import modality_verifier
from modality_verifier import ModalityVerificationResult


def test_require_verified_image_rejects_missing_verifier(monkeypatch):
    monkeypatch.setattr("modality_verifier.load_modality_verifier", lambda _path=None: None)
    with pytest.raises(input_suitability.InputSuitabilityError, match="no validated image-modality classifier"):
        input_suitability.require_verified_cxr_modality_image(Image.new("RGB", (4, 4)), "image/png")


def test_require_verified_image_accepts_verified_result(monkeypatch):
    class FakeVerifier:
        def verify(self, _image):
            return ModalityVerificationResult(is_cxr=True, score=0.9, threshold=0.5)

    monkeypatch.setattr(
        "modality_verifier.load_modality_verifier", lambda _path=None: FakeVerifier()
    )
    result = input_suitability.require_verified_cxr_modality_image(
        Image.new("RGB", (4, 4)), "image/png"
    )
    assert result.is_cxr is True


def test_require_verified_image_rejects_non_cxr(monkeypatch):
    class FakeVerifier:
        def verify(self, _image):
            return ModalityVerificationResult(is_cxr=False, score=0.1, threshold=0.5)

    monkeypatch.setattr(
        "modality_verifier.load_modality_verifier", lambda _path=None: FakeVerifier()
    )
    with pytest.raises(
        input_suitability.InputSuitabilityError,
        match="CXR score 0.100000; required threshold 0.500000",
    ):
        input_suitability.require_verified_cxr_modality_image(Image.new("RGB", (4, 4)), "image/png")


def test_rejection_message_preserves_display_precision_of_score_and_threshold(monkeypatch):
    class FakeVerifier:
        def verify(self, _image):
            return ModalityVerificationResult(
                is_cxr=False,
                score=0.992,
                threshold=0.9998095631599426,
            )

    monkeypatch.setattr(
        "modality_verifier.load_modality_verifier", lambda _path=None: FakeVerifier()
    )
    with pytest.raises(
        input_suitability.InputSuitabilityError,
        match="CXR score 0.992000; required threshold 0.999810",
    ):
        input_suitability.require_verified_cxr_modality_image(Image.new("RGB", (4, 4)), "image/png")


def test_require_verified_image_fail_closed_on_invalid_result(monkeypatch):
    class FakeVerifier:
        def verify(self, _image):
            raise ValueError("bad")

    monkeypatch.setattr(
        "modality_verifier.load_modality_verifier", lambda _path=None: FakeVerifier()
    )
    with pytest.raises(input_suitability.InputSuitabilityError, match="invalid result"):
        input_suitability.require_verified_cxr_modality_image(Image.new("RGB", (4, 4)), "image/png")