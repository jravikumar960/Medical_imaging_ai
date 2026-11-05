import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from evaluate_segmentation import evaluate_manifest


def _write_mask(path, values, affine=None):
    if affine is None:
        affine = np.eye(4)
    nib.save(nib.Nifti1Image(np.asarray(values, dtype=np.uint8), affine), path)


def _manifest(root):
    rows = []
    for split in ("train", "validation", "calibration", "test"):
        prediction_name = f"{split}-prediction.nii.gz"
        reference_name = f"{split}-reference.nii.gz"
        prediction = np.zeros((3, 3, 3), dtype=np.uint8)
        reference = np.zeros_like(prediction)
        if split == "test":
            prediction.flat[[0, 1]] = 1
            reference.flat[[1, 2]] = 1
        _write_mask(root / prediction_name, prediction)
        _write_mask(root / reference_name, reference)
        rows.append({
            "case_id": f"case-{split}",
            "patient_id": f"patient-{split}",
            "split": split,
            "prediction_path": prediction_name,
            "reference_path": reference_name,
            "dataset": "Synthetic unit-test fixture",
            "reference_source": "Synthetic unit-test fixture",
            "label_definition": "Synthetic binary foreground for metric testing only",
        })
    return rows


def test_held_out_ct_metrics_are_reported_with_groups_and_limitations(tmp_path):
    report = evaluate_manifest(_manifest(tmp_path), tmp_path)
    assert report["sample_counts"] == {
        "train": 1,
        "validation": 1,
        "calibration": 1,
        "test": 1,
    }
    assert report["test_cases_evaluated"] == 1
    assert report["patient_level_separation_checked"] is True
    assert report["metrics"]["dice"]["mean"] == pytest.approx(0.5)
    assert report["metrics"]["intersection_over_union"]["mean"] == pytest.approx(1 / 3)
    assert report["cases"][0]["false_positive_voxels"] == 1
    assert report["cases"][0]["false_negative_voxels"] == 1
    assert "not established" in report["evaluation_status"]


def test_ct_evaluator_rejects_patient_leakage_and_missing_split(tmp_path):
    manifest = _manifest(tmp_path)
    next(row for row in manifest if row["split"] == "test")["patient_id"] = "patient-train"
    with pytest.raises(ValueError, match="Patient-level leakage"):
        evaluate_manifest(manifest, tmp_path)

    manifest = _manifest(tmp_path)
    manifest = [row for row in manifest if row["split"] != "calibration"]
    with pytest.raises(ValueError, match="missing split"):
        evaluate_manifest(manifest, tmp_path)


@pytest.mark.parametrize("mismatch", ["shape", "affine"])
def test_ct_evaluator_rejects_prediction_reference_geometry_mismatch(tmp_path, mismatch):
    manifest = _manifest(tmp_path)
    prediction_name = "test-prediction.nii.gz"
    if mismatch == "shape":
        _write_mask(tmp_path / prediction_name, np.zeros((2, 3, 3), dtype=np.uint8))
    else:
        affine = np.eye(4)
        affine[0, 3] = 3
        _write_mask(tmp_path / prediction_name, np.zeros((3, 3, 3), dtype=np.uint8), affine)
    with pytest.raises(ValueError, match="shape mismatch|affine mismatch"):
        evaluate_manifest(manifest, tmp_path)
