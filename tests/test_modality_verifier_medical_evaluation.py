import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import scripts.evaluate_modality_verifier_medical as ev


def test_select_ct_slices_deterministic():
    assert ev.select_ct_slices((10, 10, 10), 3) == [0, 4, 9]
    assert ev.select_ct_slices((10, 10, 2), 5) == [0, 1]


def test_pil_from_array_rejects_non_finite():
    with pytest.raises(ValueError):
        ev.pil_from_array(np.array([[1.0, np.nan]], dtype=np.float32))


def test_evaluate_image_records_raw_score_threshold_and_decision():
    class FakeVerifier:
        def verify(self, image):
            assert image.mode == "RGB"
            return type("Result", (), {"score": 0.25, "threshold": 0.75, "is_cxr": False})()

    metadata = {"modality": "ct", "group_id": "g1"}
    row = ev.evaluate_image(FakeVerifier(), Image.new("RGB", (4, 4)), metadata)
    assert row["modality"] == "ct"
    assert row["raw_score"] == 0.25
    assert row["threshold"] == 0.75
    assert row["accepted_as_cxr"] is False


def test_group_summary_counts_acceptances():
    rows = [
        {"accepted_as_cxr": False},
        {"accepted_as_cxr": True},
    ]
    summary = ev.group_summary("g1", rows)
    assert summary == {
        "group_id": "g1",
        "image_count": 2,
        "accepted_count": 1,
        "any_accepted": True,
        "all_accepted": False,
    }


def test_evaluate_ct_uses_production_verifier_path(tmp_path, monkeypatch):
    class FakeVerifier:
        def verify(self, image):
            return type("Result", (), {"score": 0.0, "threshold": 0.5, "is_cxr": False})()

    volume = np.zeros((4, 5, 6), dtype=np.float32)
    volume[1:3, 1:4, 1:5] = 100.0
    path = tmp_path / "ct.nii.gz"
    import nibabel as nib
    nib.save(nib.Nifti1Image(volume, np.eye(4)), str(path))

    rows, group = ev.evaluate_ct(FakeVerifier(), path, slice_count=2)
    assert len(rows) == 2
    assert group["image_count"] == 2
    assert group["accepted_count"] == 0
    assert all(row["modality"] == "ct" for row in rows)


def test_evaluate_mri_case_skips_missing_sequence(tmp_path, monkeypatch):
    class FakeVerifier:
        def verify(self, image):
            return type("Result", (), {"score": 0.0, "threshold": 0.5, "is_cxr": False})()

    case_dir = tmp_path / "BraTS-GLI-00001-000"
    case_dir.mkdir()
    # Only one sequence is present.
    import nibabel as nib
    arr = np.ones((4, 5, 6), dtype=np.float32)
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(case_dir / "BraTS-GLI-00001-000-t1c.nii.gz"))

    rows, group, reason = ev.evaluate_mri_case(FakeVerifier(), case_dir)
    assert rows == []
    assert group is None
    assert "missing sequence file" in reason