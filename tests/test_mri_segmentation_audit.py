import numpy as np

from mri_evaluation import GeometryMismatchError, dice_score, geometry_compatible, iou_score


def channel_threshold_to_labels(probabilities, threshold=0.5):
    channels = np.asarray(probabilities) >= threshold
    tumor_core, whole_tumor, enhancing_tumor = channels
    labels = np.zeros(tumor_core.shape, dtype=np.uint8)
    labels[whole_tumor] = 2
    labels[tumor_core] = 1
    labels[enhancing_tumor] = 4
    return labels


def test_channel_threshold_label_mapping():
    probabilities = np.zeros((3, 2, 2, 2), dtype=np.float32)
    probabilities[:, 0, 0, 0] = 1.0
    probabilities[0, 0, 0, 1] = 1.0
    probabilities[1, 0, 0, 1] = 1.0
    labels = channel_threshold_to_labels(probabilities)
    assert labels[0, 0, 0] == 4
    assert labels[0, 0, 1] == 1

def test_known_synthetic_masks():
    pred = np.array([1, 1, 2, 4, 0], dtype=np.uint8)
    ref = np.array([1, 2, 2, 4, 0], dtype=np.uint8)
    assert dice_score(pred == 4, ref == 4) == 1.0
    assert iou_score(pred > 0, ref > 0) == 1.0


def test_orientation_handling_canonicalizes():
    import nibabel as nib

    affine_ras = np.eye(4)
    affine_las = np.diag([-1.0, 1.0, 1.0, 1.0])
    affine_las[0, 3] = 1.0
    pred = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), affine_ras)
    ref = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), affine_las)
    raw_compatible, raw_reasons = geometry_compatible(pred, ref)
    assert not raw_compatible
    assert "orientation differs" in raw_reasons or "affine matrix differs" in raw_reasons

    pred_can = nib.as_closest_canonical(pred)
    ref_can = nib.as_closest_canonical(ref)
    compatible, reasons = geometry_compatible(pred_can, ref_can)
    assert compatible
    assert reasons == []


def test_geometry_mismatch_shape():
    import nibabel as nib

    pred = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4))
    ref = nib.Nifti1Image(np.zeros((3, 2, 2), dtype=np.uint8), np.eye(4))
    compatible, reasons = geometry_compatible(pred, ref)
    assert not compatible
    assert "shape differs" in reasons
def test_official_postprocessing_label_mapping():
    import numpy as np

    # Channel 0 = tumor core, channel 1 = whole tumor, channel 2 = enhancing tumor.
    channels = np.zeros((3, 2, 2, 2), dtype=bool)
    channels[:, 0, 0, 0] = True
    channels[0, 0, 0, 1] = True
    channels[1, 0, 0, 1] = True
    tumor_core, whole_tumor, enhancing_tumor = channels
    labels = np.zeros(tumor_core.shape, dtype=np.uint8)
    labels[whole_tumor] = 2
    labels[tumor_core] = 1
    labels[enhancing_tumor] = 4
    assert labels[0, 0, 0] == 4
    assert labels[0, 0, 1] == 1
