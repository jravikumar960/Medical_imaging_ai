import numpy as np

from mri_evaluation import GeometryMismatchError, dice_score, geometry_compatible, iou_score


def test_dice_perfect_match():
    mask = np.array([1, 1, 0, 0])
    assert dice_score(mask, mask) == 1.0


def test_dice_no_overlap():
    pred = np.array([1, 1, 0, 0])
    ref = np.array([0, 0, 1, 1])
    assert dice_score(pred, ref) == 0.0


def test_empty_masks_return_one():
    empty = np.zeros(4, dtype=np.uint8)
    assert dice_score(empty, empty) == 1.0
    assert iou_score(empty, empty) == 1.0


def test_one_empty_mask_returns_zero():
    empty = np.zeros(4, dtype=np.uint8)
    nonempty = np.array([1, 0, 0, 0])
    assert dice_score(nonempty, empty) == 0.0
    assert iou_score(nonempty, empty) == 0.0


def test_iou_partial_overlap():
    pred = np.array([1, 1, 0, 0])
    ref = np.array([1, 0, 1, 0])
    assert iou_score(pred, ref) == 1 / 3


def test_geometry_mismatch_shape():
    import nibabel as nib

    pred = nib.Nifti1Image(np.zeros((2, 2, 2)), __import__("numpy").eye(4))
    ref = nib.Nifti1Image(np.zeros((3, 2, 2)), __import__("numpy").eye(4))
    compatible, reasons = geometry_compatible(pred, ref)
    assert not compatible
    assert "shape differs" in reasons


def test_geometry_mismatch_raises_during_evaluation(tmp_path):
    import nibabel as nib

    from mri_evaluation import evaluate_case

    pred = tmp_path / "a_mask.nii.gz"
    ref = tmp_path / "a-seg.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4)), str(pred))
    nib.save(nib.Nifti1Image(np.zeros((3, 2, 2), dtype=np.uint8), np.eye(4)), str(ref))
    try:
        evaluate_case(pred, ref)
    except GeometryMismatchError:
        return
    raise AssertionError("GeometryMismatchError was not raised")
def test_evaluate_case_handles_empty_masks_and_labels(tmp_path):
    import nibabel as nib

    from mri_evaluation import evaluate_case

    pred = tmp_path / "case_mask.nii.gz"
    ref = tmp_path / "case-seg.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4)), str(pred))
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4)), str(ref))
    result = evaluate_case(pred, ref)
    assert result["subject_id"] == "case"
    assert result["metrics"]["whole_tumor"]["dice"] == 1.0
    assert result["metrics"]["whole_tumor"]["predicted_voxels"] == 0
    assert result["metrics"]["enhancing_tumor_label_4"]["dice"] == 1.0
