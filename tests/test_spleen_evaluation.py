import csv
import json
from pathlib import Path

import numpy as np
import pytest

import spleen_evaluation
from spleen_evaluation import (
    CaseEvaluation,
    EvaluationReport,
    confusion_counts,
    discover_case_pairs,
    dice_from_counts,
    dice_score,
    precision_from_counts,
    precision_score,
    recall_from_counts,
    recall_score,
    validate_shape_and_affine,
    _write_reports,
)


def test_confusion_counts_and_metrics_perfect_overlap():
    pred = np.array([[[1, 1], [0, 0]]], dtype=np.uint8)
    ref = pred.copy()
    counts = confusion_counts(pred, ref)
    assert counts == {"tp": 2, "fp": 0, "fn": 0, "tn": 2}
    assert dice_score(pred, ref) == 1.0
    assert precision_score(pred, ref) == 1.0
    assert recall_score(pred, ref) == 1.0


def test_empty_masks_are_safe():
    empty = np.zeros((2, 2, 2), dtype=np.uint8)
    counts = confusion_counts(empty, empty)
    assert counts == {"tp": 0, "fp": 0, "fn": 0, "tn": 8}
    assert dice_from_counts(0, 0, 0) == 1.0
    assert precision_from_counts(0, 0, 0) == 1.0
    assert recall_from_counts(0, 0, 0) == 1.0
    assert dice_score(empty, empty) == 1.0


def test_empty_prediction_with_nonempty_reference():
    pred = np.zeros((1, 2, 2), dtype=np.uint8)
    ref = np.ones((1, 2, 2), dtype=np.uint8)
    counts = confusion_counts(pred, ref)
    assert counts["tp"] == 0 and counts["fn"] == 4
    assert dice_score(pred, ref) == 0.0
    assert precision_score(pred, ref) == 1.0
    assert recall_score(pred, ref) == 0.0


def test_empty_reference_with_nonempty_prediction():
    pred = np.ones((1, 2, 2), dtype=np.uint8)
    ref = np.zeros((1, 2, 2), dtype=np.uint8)
    counts = confusion_counts(pred, ref)
    assert counts["fp"] == 4
    assert dice_score(pred, ref) == 0.0
    assert precision_score(pred, ref) == 0.0
    assert recall_score(pred, ref) == 1.0


def test_partial_overlap_metrics():
    pred = np.array([1, 1, 0, 0], dtype=np.uint8)
    ref = np.array([1, 0, 1, 0], dtype=np.uint8)
    assert dice_from_counts(1, 1, 1) == pytest.approx(2 / 4)
    assert precision_from_counts(1, 1, 2) == 0.5
    assert recall_from_counts(1, 1, 2) == 0.5


def test_shape_mismatch_rejected(tmp_path):
    import nibabel as nib

    affine = np.eye(4)
    img = nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.float32), affine)
    lbl = nib.Nifti1Image(np.zeros((2, 3, 5), dtype=np.uint8), affine)
    img_path = tmp_path / "img.nii.gz"
    lbl_path = tmp_path / "lbl.nii.gz"
    nib.save(img, img_path)
    nib.save(lbl, lbl_path)
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_shape_and_affine(img_path, lbl_path)


def test_affine_mismatch_rejected(tmp_path):
    import nibabel as nib

    img = nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.float32), np.eye(4))
    lbl = nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.uint8), np.diag([2.0, 2.0, 2.0, 1.0]))
    img_path = tmp_path / "img.nii.gz"
    lbl_path = tmp_path / "lbl.nii.gz"
    nib.save(img, img_path)
    nib.save(lbl, lbl_path)
    with pytest.raises(ValueError, match="affine mismatch"):
        validate_shape_and_affine(img_path, lbl_path)
def test_discover_case_pairs_reports_extra_labels_and_unmatched(tmp_path):
    import nibabel as nib

    root = tmp_path / "Task09_Spleen"
    image_dir = root / "imagesTr"
    label_dir = root / "labelsTr"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    affine = np.eye(4)
    for name in ("spleen_01.nii.gz", "spleen_02.nii.gz"):
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), affine), image_dir / name)
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), affine), label_dir / "spleen_01.nii.gz")
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 3), dtype=np.uint8), affine), label_dir / "spleen_02.nii.gz")
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), affine), label_dir / "SpleenLabelmap_1.nii.gz")
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), affine), label_dir / "spleen_03.nii.gz")

    pairs, unmatched_images, extra_labels, mismatched = discover_case_pairs(root)
    assert [pair.case_id for pair in pairs] == ["spleen_01"]
    assert unmatched_images == []
    assert "SpleenLabelmap_1.nii.gz" in extra_labels
    assert "spleen_03.nii.gz" in extra_labels
    assert len(mismatched) == 1
    assert mismatched[0]["case_id"] == "spleen_02"


def test_write_reports_creates_csv_and_json(tmp_path):
    case = CaseEvaluation(
        case_id="spleen_01",
        status="success",
        image_path="img.nii.gz",
        label_path="lbl.nii.gz",
        shape=[2, 2, 2],
        predicted_voxels=2,
        reference_voxels=2,
        true_positives=2,
        false_positives=0,
        false_negatives=0,
        true_negatives=6,
        dice=1.0,
        precision=1.0,
        recall=1.0,
        runtime_seconds=0.1,
    )
    report = EvaluationReport(
        dataset_root="root",
        evaluation_split="training-set evaluation only",
        checkpoint_path="model.pt",
        checkpoint_sha256="abc",
        checkpoint_sha256_matches=False,
        preprocessing={"transforms": []},
        total_images=1,
        matched_pairs=1,
        evaluated_cases=1,
        successful_cases=1,
        failed_cases=0,
        cases=[case.__dict__.copy()],
        aggregate={"mean_dice": 1.0},
    )
    _write_reports(report, tmp_path)
    assert (tmp_path / "spleen_evaluation_results.json").is_file()
    assert (tmp_path / "spleen_evaluation_results.csv").is_file()
    with (tmp_path / "spleen_evaluation_results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["case_id"] == "spleen_01"
    assert rows[0]["dice"] == "1.0"