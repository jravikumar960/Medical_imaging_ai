"""Runtime loading and validation for the chest X-ray modality verifier.

This module is intentionally fail-closed: a missing, malformed, or invalid
artifact resolves to ``None`` rather than allowing unverified inference.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from modality_training import (
    IMAGE_SIZE,
    LABEL_TO_INDEX,
    build_model,
    build_transform,
    pil_rgb,
)


DEFAULT_CLASSIFIER_ENV = "CXR_MODALITY_CLASSIFIER_PATH"
DEFAULT_CLASSIFIER_PATH = Path("models") / "modality_classifier.pt"
REQUIRED_CHECKPOINT_KEYS = {
    "state_dict",
    "threshold",
    "labels",
    "architecture",
    "input_size",
    "preprocessing",
}
EXPECTED_ARCHITECTURE = "torchvision.mobilenet_v3_small"
EXPECTED_PREPROCESSING = {
    "resize": [IMAGE_SIZE, IMAGE_SIZE],
    "color": "RGB",
    "mean": [0.5, 0.5, 0.5],
    "std": [0.25, 0.25, 0.25],
}


@dataclass(frozen=True)
class ModalityVerificationResult:
    is_cxr: bool
    score: float
    threshold: float


class ModalityVerifier:
    def __init__(self, model: torch.nn.Module, threshold: float, checkpoint_path: Path):
        self.model = model
        self.threshold = float(threshold)
        self.checkpoint_path = checkpoint_path
        self.transform = build_transform(training=False)

    def verify(self, image: Image.Image) -> ModalityVerificationResult:
        rgb = pil_rgb(image)
        tensor = self.transform(rgb).unsqueeze(0)
        if not torch.isfinite(tensor).all():
            raise ValueError("Modality verifier received a non-finite input tensor.")
        with torch.inference_mode():
            logits = self.model(tensor)
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 2
            or logits.shape != (1, len(LABEL_TO_INDEX))
        ):
            raise ValueError("Modality verifier returned invalid logits shape.")
        if not torch.isfinite(logits).all():
            raise ValueError("Modality verifier returned non-finite logits.")
        probabilities = torch.softmax(logits, dim=1)
        if not torch.isfinite(probabilities).all():
            raise ValueError("Modality verifier returned non-finite probabilities.")
        score = float(probabilities[0, LABEL_TO_INDEX["cxr"]].detach().cpu())
        if not np.isfinite(score):
            raise ValueError("Modality verifier returned a non-finite score.")
        return ModalityVerificationResult(
            is_cxr=score >= self.threshold,
            score=score,
            threshold=self.threshold,
        )
def _validate_checkpoint(payload: Any, checkpoint_path: Path) -> tuple[torch.nn.Module, float]:
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a mapping.")
    missing = REQUIRED_CHECKPOINT_KEYS - set(payload)
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {', '.join(sorted(missing))}.")
    if payload.get("architecture") != EXPECTED_ARCHITECTURE:
        raise ValueError("Checkpoint has an unexpected architecture.")
    if payload.get("labels") != LABEL_TO_INDEX:
        raise ValueError("Checkpoint has an unexpected class mapping.")
    try:
        input_size = int(payload.get("input_size"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint has an invalid input size.") from exc
    if input_size != IMAGE_SIZE:
        raise ValueError("Checkpoint has an unexpected input size.")
    try:
        threshold = float(payload.get("threshold"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint has an invalid threshold.") from exc
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Checkpoint threshold must be finite and in [0, 1].")
    if payload.get("preprocessing") != EXPECTED_PREPROCESSING:
        raise ValueError("Checkpoint has an unexpected preprocessing configuration.")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Checkpoint has an invalid state dictionary.")

    model = build_model()
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise ValueError("Checkpoint state dictionary does not match the model.") from exc
    model.eval()

    dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.inference_mode():
        output = model(dummy)
    if not isinstance(output, torch.Tensor) or not torch.isfinite(output).all():
        raise ValueError("Checkpoint failed a finite-output validation check.")
    if output.shape != (1, len(LABEL_TO_INDEX)):
        raise ValueError("Checkpoint failed an output-shape validation check.")

    sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
    if sidecar.is_file():
        expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
        actual = _sha256_file(checkpoint_path)
        if expected and expected != actual:
            raise ValueError("Checkpoint SHA-256 sidecar does not match.")

    return model, threshold


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_modality_verifier(path: Path | str | None = None) -> ModalityVerifier | None:
    """Load a validated verifier, or return None when it is unavailable/invalid."""
    resolved = _resolve_path(path)
    if resolved is None or not resolved.is_file():
        return None
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
        model, threshold = _validate_checkpoint(payload, resolved)
    except Exception:
        return None
    return ModalityVerifier(model=model, threshold=threshold, checkpoint_path=resolved)


def _resolve_path(path: Path | str | None) -> Path | None:
    if path is None:
        raw = os.environ.get(DEFAULT_CLASSIFIER_ENV)
        if raw:
            return Path(raw)
        return DEFAULT_CLASSIFIER_PATH
    return Path(path)