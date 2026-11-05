import base64
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import render_visual_review as rvr  # noqa: E402


def test_compose_slice_rgba_shape_dtype_and_size():
    ct = np.arange(12, dtype=np.float64).reshape(3, 4)  # width 3? shape (3,4)
    # Make a 2D slice with recognizable values.
    ct2d = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float64)
    mask = np.zeros_like(ct2d, dtype=np.uint8)
    mask[0, 1] = 1
    rgba = rvr.compose_slice_rgba(ct2d, mask)
    assert rgba.shape == (2, 3, 4)
    assert rgba.dtype == np.uint8
    assert rgba.tobytes().__len__() == 2 * 3 * 4


def test_compose_slice_rgba_overlay_and_background():
    ct2d = np.full((2, 3), 0.0, dtype=np.float64)  # below window -> 0 after clip? lo=-57, 0 maps ~0.34
    mask = np.zeros((2, 3), dtype=np.uint8)
    mask[1, 2] = 1
    rgba = rvr.compose_slice_rgba(ct2d, mask)
    # mask pixel is red
    assert tuple(rgba[1, 2]) == (255, 56, 48, 255)
    # non-mask pixel: gray from windowing (all channels equal, alpha 255)
    assert rgba[0, 0, 0] == rgba[0, 0, 1] == rgba[0, 0, 2]
    assert rgba[0, 0, 3] == 255
    # flat byte order: pixel (x=2,y=1) is at offset (1*3+2)*4
    flat = rgba.reshape(-1, 4)
    assert tuple(flat[1 * 3 + 2]) == (255, 56, 48, 255)


def test_compose_slice_rgba_rejects_shape_mismatch():
    ct2d = np.zeros((2, 3), dtype=np.float64)
    mask = np.zeros((2, 4), dtype=np.uint8)
    try:
        rvr.compose_slice_rgba(ct2d, mask)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_window_ct_known_value():
    # value at lower bound maps to 0, upper bound maps to 255
    arr = np.array([[-57.0, 164.0]], dtype=np.float64)
    out = rvr.window_ct(arr)
    assert out[0, 0] == 0
    assert out[0, 1] == 255


def test_zlib_roundtrip_matches_canvas_bytes():
    ct2d = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    mask = np.zeros((2, 3), dtype=np.uint8)
    rgba = rvr.compose_slice_rgba(ct2d, mask)
    gz = base64.b64encode(__import__("zlib").compress(rgba.tobytes())).decode("ascii")
    raw = zlib.decompress(base64.b64decode(gz))
    assert raw == rgba.tobytes()
    assert len(raw) == 2 * 3 * 4
import numpy as np


def test_overlay_pred_mode_paints_prediction_red():
    ct = np.zeros((2, 3), dtype=np.float64)
    pred = np.zeros((2, 3), dtype=np.uint8)
    pred[0, 2] = 1
    gt = np.zeros((2, 3), dtype=np.uint8)
    rgba = rvr.compose_overlay_rgba(ct, pred, gt, "pred")
    assert tuple(rgba[0, 2]) == (255, 56, 48, 255)
    assert rgba[1, 0, 0] == rgba[1, 0, 1] == rgba[1, 0, 2]


def test_overlay_gt_mode_paints_ground_truth_blue():
    ct = np.zeros((2, 3), dtype=np.float64)
    pred = np.zeros((2, 3), dtype=np.uint8)
    gt = np.zeros((2, 3), dtype=np.uint8)
    gt[1, 1] = 1
    rgba = rvr.compose_overlay_rgba(ct, pred, gt, "gt")
    assert tuple(rgba[1, 1]) == (60, 120, 255, 255)


def test_overlay_compare_mode_colors():
    ct = np.zeros((2, 2), dtype=np.float64)
    pred = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    gt = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    rgba = rvr.compose_overlay_rgba(ct, pred, gt, "compare")
    assert tuple(rgba[0, 0]) == (40, 200, 80, 255)   # TP
    assert tuple(rgba[0, 1]) == (255, 56, 48, 255)   # FP
    assert tuple(rgba[1, 0]) == (60, 120, 255, 255)  # FN
    assert rgba[1, 1, 0] == rgba[1, 1, 1] == rgba[1, 1, 2]  # background gray


def test_overlay_rejects_geometry_mismatch():
    ct = np.zeros((2, 3), dtype=np.float64)
    pred = np.zeros((2, 3), dtype=np.uint8)
    gt = np.zeros((2, 4), dtype=np.uint8)
    try:
        rvr.compose_overlay_rgba(ct, pred, gt, "compare")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on shape mismatch")
    try:
        rvr.compose_overlay_rgba(np.zeros((2, 3)), np.zeros((3, 2), dtype=np.uint8),
                                np.zeros((2, 3), dtype=np.uint8), "pred")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on shape mismatch")


def test_overlay_unknown_mode_raises():
    ct = np.zeros((2, 2), dtype=np.float64)
    pred = np.zeros((2, 2), dtype=np.uint8)
    gt = np.zeros((2, 2), dtype=np.uint8)
    try:
        rvr.compose_overlay_rgba(ct, pred, gt, "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on unknown mode")


def test_overlay_voxel_indexing_flat_order():
    ct = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
    pred = np.zeros((2, 2), dtype=np.uint8)
    gt = np.zeros((2, 2), dtype=np.uint8)
    rgba = rvr.compose_overlay_rgba(ct, pred, gt, "pred")
    flat = rgba.reshape(-1, 4)
    assert flat[1, 0] == flat[1, 1] == flat[1, 2]
    assert flat[2, 0] == flat[2, 1] == flat[2, 2]
    assert rgba.tobytes().__len__() == 2 * 2 * 4


def test_overlay_windowing_consistent_across_modes():
    ct = np.full((2, 2), 100.0, dtype=np.float64)
    pred = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    gt = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    r1 = rvr.compose_overlay_rgba(ct, pred, gt, "pred")
    r2 = rvr.compose_overlay_rgba(ct, pred, gt, "gt")
    r3 = rvr.compose_overlay_rgba(ct, pred, gt, "compare")
    assert r1[1, 1, 0] == r2[1, 1, 0] == r3[1, 1, 0]
    assert r1[1, 1, 3] == r2[1, 1, 3] == r3[1, 1, 3] == 255