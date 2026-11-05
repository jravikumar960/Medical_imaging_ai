"""Synthetic MRI software tests; these are not medical-model evaluation results."""
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

import mri_segmentation


def write_sequences(directory: Path, shape=(8, 8, 8), affine=None, extension=".nii.gz"):
    affine = np.eye(4) if affine is None else np.asarray(affine, dtype=np.float64)
    values = np.linspace(1.0, 255.0, num=int(np.prod(shape)), dtype=np.float32).reshape(shape)
    paths = {}
    for offset, channel in enumerate(mri_segmentation.MRI_CHANNELS):
        path = directory / f"{channel}{extension}"
        image = nib.Nifti1Image(values + offset, affine)
        nib.save(image, str(path))
        paths[channel] = path
    return paths


def test_preprocessing_normalizes_four_channels_and_canonicalizes_ras(tmp_path):
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    affine[0, 3] = 7
    paths = write_sequences(tmp_path, affine=affine)

    flair, tensor = mri_segmentation._validated_inputs(paths)

    assert tuple(tensor.shape) == (1, 4, 8, 8, 8)
    assert torch.isfinite(tensor).all()
    assert tuple(nib.aff2axcodes(flair.affine)) == ("R", "A", "S")
    for channel in tensor[0]:
        assert abs(float(channel.mean())) < 1e-5


def test_rejects_missing_sequence_before_model_loading(tmp_path):
    paths = write_sequences(tmp_path)
    del paths["t2"]
    calls = []

    with pytest.raises(mri_segmentation.MRISegmentationError, match="Exactly T1c"):
        mri_segmentation.run_segmentation(paths, tmp_path / "output.nii.gz", model_loader=lambda: calls.append(True))

    assert calls == []


@pytest.mark.parametrize(
    ("shape", "affine", "message"),
    [
        ((8, 8, 8, 2), np.eye(4), "finite 3D"),
        ((8, 8, 8), np.diag([2.0, 1.0, 1.0, 1.0]), "1 mm isotropic"),
    ],
)
def test_rejects_unsupported_shape_or_spacing_before_loading(tmp_path, shape, affine, message):
    paths = write_sequences(tmp_path, shape=shape, affine=affine)
    calls = []

    with pytest.raises(mri_segmentation.MRISegmentationError, match=message):
        mri_segmentation.run_segmentation(paths, tmp_path / "output.nii.gz", model_loader=lambda: calls.append(True))

    assert calls == []


@pytest.mark.parametrize("mismatch", ["shape", "affine"])
def test_rejects_non_aligned_sequences_before_loading(tmp_path, mismatch):
    paths = write_sequences(tmp_path)
    channel_path = paths["t2"]
    image = nib.load(str(channel_path))
    array = np.asarray(image.dataobj, dtype=np.float32)
    affine = image.affine.copy()
    if mismatch == "shape":
        array = array[:-1, :, :]
    else:
        affine[0, 3] += 1.0
    nib.save(nib.Nifti1Image(array, affine), str(channel_path))
    calls = []

    with pytest.raises(mri_segmentation.MRISegmentationError, match="co-registered"):
        mri_segmentation.run_segmentation(paths, tmp_path / "output.nii.gz", model_loader=lambda: calls.append(True))

    assert calls == []


def test_rejects_oblique_geometry_before_loading(tmp_path):
    angle = 0.2
    affine = np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    paths = write_sequences(tmp_path, affine=affine)
    calls = []

    with pytest.raises(mri_segmentation.MRISegmentationError, match="oblique inputs are unsupported"):
        mri_segmentation.run_segmentation(paths, tmp_path / "output.nii.gz", model_loader=lambda: calls.append(True))

    assert calls == []


def test_rejects_corrupt_or_empty_sequence_before_loading(tmp_path):
    paths = write_sequences(tmp_path)
    paths["flair"].write_bytes(b"not a nifti")
    calls = []

    with pytest.raises(mri_segmentation.MRISegmentationError, match="Could not read"):
        mri_segmentation.run_segmentation(paths, tmp_path / "output.nii.gz", model_loader=lambda: calls.append(True))

    assert calls == []


def test_thresholds_outputs_writes_mask_and_uses_expected_labels(tmp_path):
    pytest.importorskip("monai")
    paths = write_sequences(tmp_path)
    output_path = tmp_path / "predicted.nii.gz"

    class FixedLogits(torch.nn.Module):
        def forward(self, inputs):
            result = torch.full((inputs.shape[0], 3, *inputs.shape[2:]), -10.0)
            result[:, 0, 1:3, :, :] = 10.0
            result[:, 1, 5:7, :, :] = 10.0
            result[:, 2, 3:5, :, :] = 10.0
            return result

    result = mri_segmentation.run_segmentation(paths, output_path, model=FixedLogits())

    assert output_path.is_file()
    assert result["shape"] == (8, 8, 8)
    assert result["mask"].dtype == np.uint8
    assert set(np.unique(result["mask"])) == {0, 1, 2, 4}
    assert result["label_counts"]["1"] == 128
    assert result["label_counts"]["2"] == 128
    assert result["label_counts"]["4"] == 128
    saved = nib.load(str(output_path))
    assert tuple(saved.shape) == result["shape"]
    assert np.allclose(saved.affine, result["affine"])


def test_verified_public_checkpoint_loads_strictly():
    pytest.importorskip("monai")
    model = mri_segmentation.load_model()
    assert not model.training
    assert model.convInit.conv.weight.shape[:2] == (16, 4)
