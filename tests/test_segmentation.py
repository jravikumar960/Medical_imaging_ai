import hashlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import torch

import segmentation
import orientation_review
from segmentation import (
    GeometryError,
    SegmentationDependencyError,
    SegmentationError,
    UNetConfig,
    build_unet,
    inspect_nifti_geometry,
    infer_preprocessed_volume,
    probabilities_to_mask,
    save_mask_nifti,
    segment_nifti,
    validate_geometry,
    validate_volume_array,
)


class _SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, value):
        return torch.cat((-value * self.weight, value * self.weight), dim=1)


def _identity_inferer(*, inputs, predictor, **_kwargs):
    return predictor(inputs)


def _install_fake_nifti_monai(monkeypatch, volume, affine):
    class Header:
        def copy(self):
            return Header()

        def get_data_dtype(self):
            return np.dtype("int16")

    class SourceImage:
        shape = volume.shape
        header = Header()

        def __init__(self):
            self.affine = affine

        def get_fdata(self, dtype):
            return volume.astype(dtype)

        def get_qform(self, coded=True):
            return affine, 1

        def get_sform(self, coded=True):
            return affine, 1

    class SavedImage:
        def __init__(self, data, saved_affine, header):
            self.data = data
            self.affine = saved_affine
            self.header = header

        def set_data_dtype(self, dtype):
            self.dtype = dtype

    class FakeNib:
        saved = None

        @staticmethod
        def load(_path):
            return SourceImage()

        @staticmethod
        def aff2axcodes(_affine):
            return ("R", "A", "S")

        @classmethod
        def Nifti1Image(cls, data, saved_affine, header):
            cls.saved = SavedImage(data, saved_affine, header)
            return cls.saved

        @classmethod
        def save(cls, image, path):
            cls.path = path

    class FakeTransform:
        def __init__(self, name, **kwargs):
            self.name = name
            self.kwargs = kwargs

        def __call__(self, item):
            if self.name == "LoadImaged":
                item["image"] = torch.from_numpy(volume.copy()).unsqueeze(0)
            return item

    class FakeCompose:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, item):
            for transform in self.transforms:
                item = transform(item)
            return item

    class FakeInvert:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, item):
            return item

    fake_modules = (
        FakeNib,
        _identity_inferer,
        FakeCompose,
        lambda **kwargs: FakeTransform("EnsureChannelFirstd", **kwargs),
        lambda **kwargs: FakeTransform("EnsureTyped", **kwargs),
        FakeInvert,
        lambda **kwargs: FakeTransform("LoadImaged", **kwargs),
        lambda **kwargs: FakeTransform("Orientationd", **kwargs),
        lambda **kwargs: FakeTransform("ScaleIntensityRanged", **kwargs),
        lambda **kwargs: FakeTransform("Spacingd", **kwargs),
        object,
    )
    monkeypatch.setattr(segmentation, "_optional_modules", lambda: fake_modules)
    return FakeNib


def test_validate_synthetic_3d_geometry_needs_explicit_attestation():
    affine = np.diag([1.5, 1.5, 2.0, 1.0])
    info = validate_geometry((12, 10, 8), affine, geometry_attested=True)
    assert info.shape == (12, 10, 8)
    assert info.spacing == (1.5, 1.5, 2.0)
    with pytest.raises(GeometryError, match="not attested"):
        validate_geometry((12, 10, 8), affine, geometry_attested=False)


@pytest.mark.parametrize(
    "shape,affine",
    [
        ((10, 10), np.eye(4)),
        ((10, 10, 10), np.full((4, 4), np.nan)),
        ((10, 10, 10), np.diag([1, 0, 1, 1])),
    ],
)
def test_rejects_invalid_volume_geometry(shape, affine):
    with pytest.raises((GeometryError, SegmentationError)):
        validate_geometry(shape, affine, geometry_attested=True)


def test_rejects_disagreeing_qform_sform_instead_of_guessing_geometry():
    qform = np.eye(4)
    sform = np.diag([-1.0, 1.0, 1.0, 1.0])
    with pytest.raises(GeometryError, match="qform and sform disagree"):
        validate_geometry((8, 8, 8), np.eye(4), geometry_attested=True, qform=qform, sform=sform)


def test_header_inspection_reports_geometry_without_attesting_anatomy(tmp_path):
    nib = pytest.importorskip("nibabel")
    affine = np.diag([1.0, -1.0, 2.5, 1.0])
    image = nib.Nifti1Image(np.zeros((6, 7, 8), dtype=np.int16), affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    path = tmp_path / "synthetic_header_only.nii.gz"
    nib.save(image, path)

    result = inspect_nifti_geometry(path)
    assert result["shape"] == [6, 7, 8]
    assert result["spacing_mm"] == [1.0, 1.0, 2.5]
    assert result["axis_codes"] == ["R", "P", "S"]
    assert result["qform_code"] == result["sform_code"] == 1
    assert result["datatype"] == "int16"
    assert result["voxel_count"] == 6 * 7 * 8


def test_rejects_oversized_volumes_before_reading_voxels():
    with pytest.raises(SegmentationError, match="voxel limit"):
        validate_geometry((600, 600, 500), np.eye(4), geometry_attested=True)


def test_validate_volume_array_rejects_2d_non_numeric_and_non_finite():
    with pytest.raises(SegmentationError, match="3D"):
        validate_volume_array(np.zeros((8, 8), dtype=np.float32))
    with pytest.raises(SegmentationError, match="numeric"):
        validate_volume_array(np.full((4, 4, 4), "x"))
    invalid = np.zeros((4, 4, 4), dtype=np.float32)
    invalid[1, 1, 1] = np.inf
    with pytest.raises(SegmentationError, match="NaN or infinite"):
        validate_volume_array(invalid)


def test_small_model_inference_uses_batch_channel_and_3d_spatial_dimensions():
    model = _SmallModel()
    source = np.linspace(0.0, 1.0, 8 * 10 * 12, dtype=np.float32).reshape(8, 10, 12)
    probabilities = infer_preprocessed_volume(
        source,
        model,
        _identity_inferer,
        roi_size=(4, 4, 4),
        sw_batch_size=1,
    )
    assert tuple(probabilities.shape) == (2, 8, 10, 12)
    assert not model.training
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities.sum(dim=0), torch.ones((8, 10, 12)))


def test_inference_rejects_wrong_tensor_shape_and_non_finite_logits():
    source = np.zeros((5, 6, 7), dtype=np.float32)
    model = _SmallModel()

    def bad_shape(*, inputs, **_kwargs):
        return torch.zeros((1, 1, *inputs.shape[2:]))

    with pytest.raises(SegmentationError, match="two-class spatial contract"):
        infer_preprocessed_volume(source, model, bad_shape)

    def bad_values(*, inputs, **_kwargs):
        out = torch.zeros((1, 2, *inputs.shape[2:]))
        out[0, 1, 0, 0, 0] = float("nan")
        return out

    with pytest.raises(SegmentationError, match="non-finite"):
        infer_preprocessed_volume(source, model, bad_values)


def test_mask_dimensions_and_labels_are_checked():
    probs = torch.zeros((2, 4, 5, 6))
    probs[1, 1:3, 2:4, 1:5] = 1
    mask = probabilities_to_mask(probs, (4, 5, 6))
    assert mask.shape == (4, 5, 6)
    assert set(np.unique(mask)) == {0, 1}
    with pytest.raises(SegmentationError, match="dimensions"):
        probabilities_to_mask(probs, (5, 5, 6))
    probs[0, 0, 0, 0] = float("nan")
    with pytest.raises(SegmentationError, match="non-finite"):
        probabilities_to_mask(probs, (4, 5, 6))


def test_mask_writer_preserves_source_affine_and_header(tmp_path):
    class Header:
        def __init__(self):
            self.dtype = None

        def copy(self):
            return Header()

    class FakeNifti:
        def __init__(self, data, affine, header):
            self.data = data
            self.affine = affine
            self.header = header

        def set_data_dtype(self, dtype):
            self.header.dtype = dtype

    class FakeNib:
        image = None

        @classmethod
        def Nifti1Image(cls, data, affine, header):
            cls.image = FakeNifti(data, affine, header)
            return cls.image

        @classmethod
        def save(cls, image, path):
            cls.path = path

    affine = np.diag([1.5, -1.5, 2.0, 1.0])
    geometry = validate_geometry((4, 5, 6), affine, geometry_attested=True)
    mask = np.zeros((4, 5, 6), dtype=np.uint8)
    output = tmp_path / "nested" / "mask.nii.gz"
    save_mask_nifti(FakeNib, output, mask, geometry, Header())
    np.testing.assert_array_equal(FakeNib.image.affine, affine)
    assert FakeNib.image.data.shape == (4, 5, 6)
    assert FakeNib.image.header.dtype is np.uint8
    assert FakeNib.path == str(output)


def test_cli_path_validation_and_checkpoint_requirements_precede_optional_imports(tmp_path):
    missing = tmp_path / "missing.nii.gz"
    output = tmp_path / "mask.nii.gz"
    with pytest.raises(SegmentationError, match="does not exist"):
        segment_nifti(missing, output)

    source = tmp_path / "source.nii.gz"
    source.write_bytes(b"synthetic placeholder; not a NIfTI image")
    with pytest.raises(SegmentationError, match="No pretrained checkpoint"):
        segment_nifti(source, output, geometry_attested=True)
    with pytest.raises(SegmentationError, match="cannot be marked pretrained"):
        segment_nifti(source, output, geometry_attested=True, allow_untrained=True, model_status="pretrained")
    output.write_bytes(b"existing")
    with pytest.raises(SegmentationError, match="already exists"):
        segment_nifti(source, output, geometry_attested=True)


def test_mocked_synthetic_nifti_pipeline_preserves_dimensions_and_affine(tmp_path, monkeypatch):
    shape = (8, 10, 12)
    volume = np.linspace(-100.0, 200.0, num=np.prod(shape), dtype=np.float32).reshape(shape)
    affine = np.diag([1.5, -1.5, 2.0, 1.0])
    fake_nib = _install_fake_nifti_monai(monkeypatch, volume, affine)
    source = tmp_path / "synthetic.nii.gz"
    source.write_bytes(b"fixture consumed by mocked NIfTI loader")
    output = tmp_path / "output" / "mask.nii.gz"
    result = segment_nifti(
        source,
        output,
        geometry_attested=True,
        model=_SmallModel(),
        model_status="untrained",
        allow_untrained=True,
    )
    assert result.mask_shape == shape
    assert result.output_path == output
    assert result.model_status == "untrained"
    assert fake_nib.saved.data.shape == shape
    assert fake_nib.saved.data.dtype == np.uint8
    assert fake_nib.saved.dtype is np.uint8
    np.testing.assert_array_equal(fake_nib.saved.affine, affine)
    assert fake_nib.path == str(output)


def test_pretrained_library_call_requires_review_before_checkpoint_loader(tmp_path, monkeypatch):
    shape = (4, 5, 6)
    volume = np.zeros(shape, dtype=np.float32)
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    _install_fake_nifti_monai(monkeypatch, volume, affine)
    source = tmp_path / "synthetic-pretrained.nii.gz"
    source.write_bytes(b"synthetic bytes; no model inference is run")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint placeholder")
    output = tmp_path / "mask.nii.gz"
    loader_calls = []
    monkeypatch.setattr(segmentation, "load_pretrained_unet", lambda *_args: loader_calls.append(True))

    with pytest.raises(GeometryError, match="signed, volume-specific orientation review"):
        segment_nifti(source, output, geometry_attested=True, checkpoint_path=checkpoint)
    assert loader_calls == []


@pytest.mark.parametrize("review", [{}, {"schema": "wrong-schema"}])
def test_invalid_pretrained_review_fails_before_checkpoint_loader(tmp_path, monkeypatch, review):
    shape = (4, 5, 6)
    volume = np.zeros(shape, dtype=np.float32)
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    _install_fake_nifti_monai(monkeypatch, volume, affine)
    source = tmp_path / "synthetic-invalid-review.nii.gz"
    source.write_bytes(b"synthetic bytes; no model inference is run")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint placeholder")
    loader_calls = []
    monkeypatch.setattr(segmentation, "load_pretrained_unet", lambda *_args: loader_calls.append(True))
    monkeypatch.setenv("CT_ORIENTATION_REVIEW_HMAC_KEY", "test-key-long-enough-for-signature-validation")

    with pytest.raises(GeometryError):
        segment_nifti(
            source,
            tmp_path / "mask.nii.gz",
            geometry_attested=True,
            checkpoint_path=checkpoint,
            orientation_review_manifest=review,
        )
    assert loader_calls == []


def test_invalid_nifti_origin_review_fails_before_checkpoint_loader(tmp_path, monkeypatch):
    nib = pytest.importorskip("nibabel")
    source = tmp_path / "synthetic-nifti-origin.nii.gz"
    image = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.int16), np.diag([1.0, 1.0, 2.0, 1.0]))
    nib.save(image, source)
    geometry = segmentation.inspect_nifti_geometry(source)
    record_path = tmp_path / "nifti-origin-record.json"
    record_path.write_bytes(b"{}")
    review = {
        "schema": segmentation.orientation_review.NIFTI_ORIGIN_ORIENTATION_REVIEW_SCHEMA,
        "volume_sha256": segmentation.orientation_review.sha256_file(source),
        "geometry_sha256": segmentation.orientation_review.geometry_sha256(geometry),
        "origin_record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
        "signature_hmac_sha256": "0" * 64,
    }
    monkeypatch.setenv("CT_ORIENTATION_REVIEW_HMAC_KEY", "synthetic-test-review-key-at-least-thirty-two")
    loader_calls = []
    monkeypatch.setattr(segmentation, "load_pretrained_unet", lambda *_args: loader_calls.append(True))

    with pytest.raises(GeometryError, match="signature verification failed"):
        segment_nifti(
            source,
            tmp_path / "mask.nii.gz",
            geometry_attested=True,
            orientation_review_manifest=review,
            nifti_origin_record_path=record_path,
            checkpoint_path=tmp_path / "checkpoint.pt",
        )
    assert loader_calls == []


def test_synthetic_nifti_review_signing_and_verification_only():
    key = b"synthetic-test-key-not-reviewer-approval"
    geometry = {
        "shape": [2, 3, 4],
        "axis_codes": ["R", "A", "S"],
    }
    now = datetime.now(timezone.utc)
    review = {
        "schema": orientation_review.NIFTI_ORIGIN_ORIENTATION_REVIEW_SCHEMA,
        "origin_record_sha256": "a" * 64,
        "volume_sha256": "b" * 64,
        "geometry_sha256": orientation_review.geometry_sha256(geometry),
        "nifti_axis_codes": ["R", "A", "S"],
        "reviewer_id": "SYNTHETIC TEST ONLY - NOT A REAL REVIEWER",
        "reviewer_independent": True,
        "laterality_verified": True,
        "slice_direction_verified": True,
        "physical_slice_direction": "inferior_to_superior",
        "orientation_reference": "Synthetic fixture; no anatomical reference used.",
        "orientation_method": "Unit test only.",
        "reviewed_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=1)).isoformat(),
        "review_notes": "Synthetic signature verification fixture, not reviewer approval.",
        "signature_hmac_sha256": "",
    }

    signed_review = orientation_review.sign_orientation_review(review, key=key)
    assert len(signed_review["signature_hmac_sha256"]) == 64
    orientation_review.verify_nifti_origin_orientation_review(
        signed_review, "b" * 64, geometry, "a" * 64, key=key
    )
    altered_review = {**signed_review, "review_notes": "Altered after signing."}
    with pytest.raises(orientation_review.OrientationReviewError, match="signature verification failed"):
        orientation_review.verify_nifti_origin_orientation_review(
            altered_review, "b" * 64, geometry, "a" * 64, key=key
        )


def test_orientation_review_signer_rejects_short_explicit_key():
    with pytest.raises(orientation_review.OrientationReviewKeyUnavailable):
        orientation_review.sign_orientation_review({}, key=b"short")


def test_unet_configuration_is_validated():
    with pytest.raises(SegmentationError, match="one more entry"):
        UNetConfig(channels=(8, 16), strides=(2, 2))
    with pytest.raises(SegmentationError, match="positive integers"):
        UNetConfig(channels=(8, 0), strides=(2,))


def test_optional_untrained_monai_architecture_has_clear_dependency_error_when_absent(monkeypatch):
    monkeypatch.setattr(segmentation, "_optional_modules", lambda: (_ for _ in ()).throw(SegmentationDependencyError("optional deps missing")))
    with pytest.raises(SegmentationDependencyError, match="optional deps missing"):
        build_unet()
