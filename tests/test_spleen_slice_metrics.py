"""Tests for per-slice spleen metrics and worst-slice selection."""

from __future__ import annotations

import numpy as np

from spleen_evaluation import (
    slice_metrics,
    per_slice_metrics,
    compute_slice_metrics,
    worst_slices,
    identify_worst_slices,
    rank_worst_slices,
    worst_slice_indices,
    worst_slices_for_case,
    confusion_counts,
)


def test_slice_metrics_perfect_single_slice():
    pred = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    ref = pred.copy()
    metrics = slice_metrics(pred, ref)
    assert len(metrics) == 1
    m = metrics[0]
    assert m["tp"] == 2
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert m["dice"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0


def test_slice_metrics_3d_counts_per_slice():
    # Shape (2,2,3): each slice differs
    pred = np.zeros((2, 2, 3), dtype=np.uint8)
    ref = np.zeros((2, 2, 3), dtype=np.uint8)
    pred[:, :, 0] = 1
    ref[:, :, 0] = 1  # perfect
    pred[:, :, 1] = 1
    # ref slice1 empty -> over-seg
    ref[:, :, 2] = 1
    # pred slice2 empty -> under-seg
    metrics = slice_metrics(pred, ref)
    assert len(metrics) == 3
    assert metrics[0]["dice"] == 1.0
    assert metrics[1]["fp"] == 4 and metrics[1]["fn"] == 0
    assert metrics[2]["fp"] == 0 and metrics[2]["fn"] == 4
    # worst should be slice 1 or 2 (dice 0), tie broken by errors then index
    worst = worst_slice_indices(pred, ref, k=1)
    assert worst == [1] or worst == [2]


def test_worst_slices_returns_dicts_sorted():
    metrics = [
        {"slice_index": 0, "tp": 4, "fp": 0, "fn": 0, "tn": 0, "dice": 1.0, "precision": 1.0, "recall": 1.0},
        {"slice_index": 1, "tp": 0, "fp": 4, "fn": 0, "tn": 0, "dice": 0.0, "precision": 0.0, "recall": 0.0},
        {"slice_index": 2, "tp": 2, "fp": 2, "fn": 2, "tn": 0, "dice": 0.5, "precision": 0.5, "recall": 0.5},
    ]
    worst = worst_slices(metrics, k=2)
    assert [w["slice_index"] for w in worst] == [1, 2]
    assert "reason" in worst[0]


def test_worst_slices_accepts_arrays():
    pred = np.zeros((2, 2, 2), dtype=np.uint8)
    ref = np.zeros((2, 2, 2), dtype=np.uint8)
    pred[:, :, 0] = 1
    ref[:, :, 1] = 1
    worst = worst_slices((pred, ref), k=2)
    assert len(worst) == 2
    assert worst[0]["dice"] == 0.0


def test_aliases():
    pred = np.zeros((2, 2, 2), dtype=np.uint8)
    ref = pred.copy()
    assert per_slice_metrics(pred, ref) == compute_slice_metrics(pred, ref)
    assert identify_worst_slices(pred, ref, k=1) == rank_worst_slices(pred, ref, k=1)


def test_empty_slice_convention():
    pred = np.zeros((2, 2, 1), dtype=np.uint8)
    ref = np.zeros((2, 2, 1), dtype=np.uint8)
    m = slice_metrics(pred, ref)[0]
    assert m["dice"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0


def test_axis_parameter():
    pred = np.zeros((3, 2, 2), dtype=np.uint8)
    ref = np.zeros((3, 2, 2), dtype=np.uint8)
    pred[0] = 1
    ref[0] = 1
    metrics = slice_metrics(pred, ref, axis=0)
    assert len(metrics) == 3
    assert metrics[0]["dice"] == 1.0


def test_worst_slices_for_case():
    pred = np.zeros((2, 2, 2), dtype=np.uint8)
    ref = np.zeros((2, 2, 2), dtype=np.uint8)
    pred[:, :, 0] = 1
    ref[:, :, 0] = 1
    ws = worst_slices_for_case(pred, ref, k=1)
    assert len(ws) == 1
    assert ws[0]["dice"] == 1.0 or ws[0]["dice"] == 0.0


def test_shape_mismatch_raises():
    import pytest

    pred = np.zeros((2, 2, 2), dtype=np.uint8)
    ref = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        slice_metrics(pred, ref)