"""Optional integrity and safe-load check for the locally supplied MONAI weights."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai", reason="checkpoint smoke requires the isolated MONAI environment")

from segmentation import UNetConfig, load_pretrained_unet  # noqa: E402


CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "bundles"
    / "spleen_ct_segmentation"
    / "models"
    / "model.pt"
)
EXPECTED_SIZE = 19_297_197
EXPECTED_SHA256 = "502c312893994ec071a85c4f2e7d83a43f7789969faa73dae62f1291177f50fe"


def test_official_bundle_checkpoint_hash_and_safe_strict_load():
    if not CHECKPOINT.is_file():
        pytest.skip("optional local MONAI checkpoint is not present")

    assert CHECKPOINT.stat().st_size == EXPECTED_SIZE
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256

    # load_pretrained_unet uses torch.load(weights_only=True) and strict=True.
    model = load_pretrained_unet(CHECKPOINT, UNetConfig())
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    assert isinstance(payload, dict) and len(payload) == 148
    assert all(torch.equal(model.state_dict()[key].cpu(), value.cpu()) for key, value in payload.items())
