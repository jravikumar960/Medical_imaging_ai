"""Opt-in smoke test for real MONAI transforms and a small 3D U-Net.

The test uses generated volume data and random, explicitly untrained weights.
It checks software wiring and geometry only, never segmentation quality.
"""
import numpy as np
import pytest
import torch

nib = pytest.importorskip("nibabel", reason="optional segmentation environment requires nibabel")
pytest.importorskip("monai", reason="optional segmentation environment requires MONAI")

from monai.networks.nets import UNet

import segmentation
from segmentation import UNetConfig, probabilities_to_mask, save_mask_nifti, segment_nifti


def _invert_synthetic_probabilities(modules, preprocessing, transformed, probabilities):
    """Use the same MetaTensor construction and Invertd settings as the wrapper."""
    image = transformed["image"]
    transformed["pred"] = type(image)(probabilities)
    inverter = modules[5](
        keys="pred",
        transform=preprocessing,
        orig_keys="image",
        nearest_interp=False,
        to_tensor=True,
    )
    return inverter(transformed)["pred"]


def test_real_monai_inverse_transform_restores_synthetic_prediction_to_source_grid(tmp_path):
    """Exercise actual MONAI preprocessing/inversion without calling a model or the inference gate."""
    shape = (12, 14, 8)
    affine = np.asarray(
        [[0.0, -0.9, 0.0, 120.0], [-0.8, 0.0, 0.0, -80.0], [0.0, 0.0, 5.0, 30.0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    values = np.linspace(-100.0, 200.0, num=np.prod(shape), dtype=np.float32).reshape(shape)
    source_image = nib.Nifti1Image(values, affine)
    source_image.set_qform(affine, code=1)
    source_image.set_sform(affine, code=1)
    input_path = tmp_path / "synthetic_oblique_ct.nii.gz"
    output_path = tmp_path / "synthetic_restored_mask.nii.gz"
    nib.save(source_image, input_path)

    modules = segmentation._optional_modules()
    preprocessing = segmentation._preprocessing_transforms(modules)
    transformed = preprocessing({"image": str(input_path)})
    image = transformed["image"]
    assert tuple(image.shape[1:]) != shape  # RAS reorientation plus target spacing changed the working grid.
    assert tuple(nib.aff2axcodes(image.affine)) == ("R", "A", "S")

    # Deterministic synthetic class probabilities; no network or checkpoint is loaded or called.
    spatial_shape = tuple(image.shape[1:])
    probabilities = torch.zeros((2, *spatial_shape), dtype=torch.float32)
    probabilities[0] = 1.0
    region = tuple(slice(length // 4, max(length // 4 + 1, 3 * length // 4)) for length in spatial_shape)
    probabilities[(0, *region)] = 0.0
    probabilities[(1, *region)] = 1.0

    restored = _invert_synthetic_probabilities(modules, preprocessing, transformed, probabilities)
    assert tuple(restored.shape) == (2, *shape)
    np.testing.assert_allclose(restored.affine.cpu().numpy(), affine, atol=1e-5, rtol=1e-5)

    mask = probabilities_to_mask(restored, shape)
    geometry = segmentation.VolumeGeometry(
        shape=shape,
        affine=affine,
        spacing=(0.8, 0.9, 5.0),
        axis_codes=tuple(nib.aff2axcodes(affine)),
    )
    save_mask_nifti(nib, output_path, mask, geometry, source_image.header)
    saved = nib.load(output_path)
    assert tuple(saved.shape) == shape
    np.testing.assert_allclose(saved.affine, affine, atol=1e-5, rtol=1e-5)
    assert tuple(nib.aff2axcodes(saved.affine)) == tuple(nib.aff2axcodes(source_image.affine))
    assert tuple(saved.header.get_zooms()) == pytest.approx((0.8, 0.9, 5.0), abs=1e-5)
    assert saved.get_data_dtype() == np.dtype(np.uint8)
    assert set(np.unique(np.asarray(saved.dataobj)).tolist()).issubset({0, 1})


@pytest.mark.parametrize("metadata_state", ["empty", "malformed"])
def test_monai_inverse_transform_rejects_missing_or_malformed_history(tmp_path, metadata_state):
    shape = (12, 14, 8)
    affine = np.asarray(
        [[0.0, -0.9, 0.0, 120.0], [-0.8, 0.0, 0.0, -80.0], [0.0, 0.0, 5.0, 30.0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    source = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine)
    source.set_qform(affine, code=1)
    source.set_sform(affine, code=1)
    input_path = tmp_path / f"synthetic_{metadata_state}_history.nii.gz"
    nib.save(source, input_path)

    modules = segmentation._optional_modules()
    preprocessing = segmentation._preprocessing_transforms(modules)
    transformed = preprocessing({"image": str(input_path)})
    image = transformed["image"]
    spatial_shape = tuple(image.shape[1:])
    probabilities = torch.zeros((2, *spatial_shape), dtype=torch.float32)
    probabilities[0] = 1.0
    if metadata_state == "empty":
        image.applied_operations = []
    else:
        image.applied_operations = [{"class": "not-a-valid-MONAI-operation"}]

    with pytest.raises(RuntimeError, match="applying transform"):
        _invert_synthetic_probabilities(modules, preprocessing, transformed, probabilities)


def test_real_monai_preprocess_forward_inverse_and_nifti_export(tmp_path):
    shape = (24, 24, 24)
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    volume = np.linspace(-100.0, 200.0, num=np.prod(shape), dtype=np.float32).reshape(shape)
    source_image = nib.Nifti1Image(volume, affine)
    source_image.set_qform(affine, code=1)
    source_image.set_sform(affine, code=1)
    input_path = tmp_path / "synthetic_ct.nii.gz"
    output_path = tmp_path / "synthetic_mask.nii.gz"
    nib.save(source_image, input_path)

    # Small random U-Net: this smoke test is not pretrained model evidence.
    config = UNetConfig(channels=(4, 8, 16), strides=(2, 2), num_res_units=1)
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=config.channels,
        strides=config.strides,
        num_res_units=config.num_res_units,
        norm="batch",
    )
    result = segment_nifti(
        input_path,
        output_path,
        geometry_attested=True,
        model=model,
        model_status="untrained",
        allow_untrained=True,
        config=config,
        device="cpu",
        roi_size=(8, 8, 8),
        sw_batch_size=2,
        overlap=0.25,
    )

    source = nib.load(input_path)
    mask = nib.load(output_path)
    assert result.model_status == "untrained"
    assert result.mask_shape == shape
    assert tuple(mask.shape) == tuple(source.shape)
    assert nib.aff2axcodes(mask.affine) == nib.aff2axcodes(source.affine)
    np.testing.assert_allclose(mask.affine, source.affine, atol=1e-6)
    assert set(np.unique(mask.get_fdata()).astype(int)).issubset({0, 1})
    assert mask.get_data_dtype() == np.dtype(np.uint8)
