import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from modality_training import (
    LABEL_TO_INDEX,
    ModalityImageDataset,
    _transform,
    build_model,
    classification_metrics,
    select_threshold,
    train_and_evaluate,
    validate_manifest,
)


def _manifest(tmp_path):
    rows = []
    for split_index, split in enumerate(("train", "validation", "calibration", "test")):
        examples = [
            ("cxr", "chest_xray", "patient", 70 + split_index),
            ("non_cxr", "cars", "car-maker", 100 + split_index),
            ("non_cxr", "buildings", "architect", 130 + split_index),
            ("non_cxr", "people", "photographer", 160 + split_index),
            ("non_cxr", "everyday_objects", "creator", 190 + split_index),
        ]
        for label, category, group_suffix, pixel in examples:
            path = f"{split}-{category}.png"
            Image.new("L", (24, 24), pixel).save(tmp_path / path)
            rows.append(
                {
                    "image_path": path,
                    "label": label,
                    "split": split,
                    "group_id": f"{split}-{category}-{group_suffix}",
                    "source": "fixture",
                    "category": category,
                    "license": "CC0",
                    "attribution": "synthetic test fixture",
                }
            )
    return pd.DataFrame(rows)


def test_manifest_requires_group_isolation_and_license_provenance(tmp_path):
    manifest = _manifest(tmp_path)
    checked = validate_manifest(manifest, tmp_path)
    assert len(checked) == 20

    manifest.loc[manifest["image_path"] == "test-chest_xray.png", "group_id"] = (
        "train-chest_xray-patient"
    )
    with pytest.raises(ValueError, match="Group leakage"):
        validate_manifest(manifest, tmp_path)

    manifest = _manifest(tmp_path)
    manifest.loc[manifest["image_path"] == "test-chest_xray.png", "license"] = ""
    with pytest.raises(ValueError, match="license"):
        validate_manifest(manifest, tmp_path)


def test_manifest_rejects_cross_split_exact_image_duplicates(tmp_path):
    manifest = _manifest(tmp_path)
    (tmp_path / "test-chest_xray.png").write_bytes(
        (tmp_path / "train-chest_xray.png").read_bytes()
    )
    with pytest.raises(ValueError, match="Exact image duplicate occurs"):
        validate_manifest(manifest, tmp_path)


def test_manifest_rejects_missing_classes_and_path_traversal(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.loc[manifest["split"] == "test", "label"] = "cxr"
    manifest.loc[manifest["split"] == "test", "category"] = "chest_xray"
    with pytest.raises(ValueError, match="both cxr and non_cxr"):
        validate_manifest(manifest, tmp_path)

    manifest = _manifest(tmp_path)
    manifest.loc[
        manifest["image_path"] == "test-chest_xray.png", "image_path"
    ] = "..\\outside.png"
    with pytest.raises(ValueError, match="escapes"):
        validate_manifest(manifest, tmp_path)


def test_dataset_decodes_images_into_model_tensor(tmp_path):
    manifest = validate_manifest(_manifest(tmp_path), tmp_path)
    rows = manifest.loc[manifest["split"] == "train"]
    dataset = ModalityImageDataset(rows, _transform(training=False))
    tensor, label = dataset[0]
    assert tuple(tensor.shape) == (3, 224, 224)
    assert label in LABEL_TO_INDEX.values()


def test_classifier_is_small_randomly_initialized_binary_model():
    model = build_model()
    output = model(torch.zeros(1, 3, 224, 224))
    assert tuple(output.shape) == (1, 2)
    assert sum(parameter.numel() for parameter in model.parameters()) < 3_000_000


def test_threshold_enforces_calibration_false_accept_limit():
    targets = np.asarray([0] * 6 + [1] * 6)
    scores = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95])
    threshold = select_threshold(targets, scores, max_false_accept_rate=0.5)
    metrics = classification_metrics(targets, scores, threshold)
    assert metrics["false_accept_rate_non_cxr"] <= 0.5
    assert metrics["confusion_matrix"] == [[3, 3], [0, 6]]
    assert metrics["false_reject_rate_cxr"] == 0.0
    assert metrics["precision_cxr"] == 0.6666666666666666
    assert metrics["recall_cxr"] == 1.0


def test_threshold_uses_validated_score_headroom_when_calibration_allows_it():
    targets = np.asarray([0] * 3000 + [1] * 150)
    scores = np.asarray([0.01] * 3000 + [0.99980956] * 150)

    threshold = select_threshold(targets, scores, max_false_accept_rate=0.001)
    metrics = classification_metrics(targets, scores, threshold)

    assert threshold == 0.99
    assert metrics["confusion_matrix"] == [[3000, 0], [0, 150]]
    assert metrics["false_accept_rate_non_cxr"] == 0.0
    assert metrics["false_reject_rate_cxr"] == 0.0


def test_threshold_headroom_never_exceeds_calibration_false_accept_limit():
    targets = np.asarray([0] * 3000 + [1] * 150)
    scores = np.asarray([0.01] * 2996 + [0.995] * 4 + [0.99980956] * 150)

    threshold = select_threshold(targets, scores, max_false_accept_rate=0.001)
    metrics = classification_metrics(targets, scores, threshold)

    assert threshold > 0.99
    assert metrics["false_accept_rate_non_cxr"] <= 0.001


def test_threshold_requires_enough_calibration_negatives():
    with pytest.raises(ValueError, match="at least 3000 non-CXR"):
        select_threshold(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0.1, 0.2, 0.8, 0.9]),
            max_false_accept_rate=0.001,
        )


def test_one_epoch_training_writes_research_only_metrics_and_checkpoint(tmp_path):
    rows = []
    for split_index, split in enumerate(("train", "validation", "calibration", "test")):
        examples = [
            ("cxr", "chest_xray", 10),
            ("non_cxr", "cars", 20),
            ("non_cxr", "buildings", 30),
            ("non_cxr", "people", 40),
            ("non_cxr", "everyday_objects", 50),
        ]
        for image_index, (label, category, pixel_offset) in enumerate(examples):
            path = f"{split}-{category}.png"
            pixel = pixel_offset + split_index * 50
            Image.new("L", (24, 24), pixel).save(tmp_path / path)
            rows.append(
                {
                    "image_path": path,
                    "label": label,
                    "split": split,
                    "group_id": f"{split}-{category}-{image_index}",
                    "source": "synthetic-fixture",
                    "category": category,
                    "license": "synthetic",
                    "attribution": "generated test image",
                }
            )
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    output_dir = tmp_path / "model-output"

    report = train_and_evaluate(
        manifest_path,
        tmp_path,
        output_dir,
        epochs=1,
        batch_size=8,
        max_false_accept_rate=0.9,
    )

    assert report["status"] == "research_only_not_connected_to_application_inference"
    assert report["held_out_test"]["n"] == 5
    assert report["held_out_test"]["confusion_matrix"]
    assert report["held_out_test_by_category"]["cars"]["n"] == 1
    assert (output_dir / "modality_classifier.pt").is_file()
    assert (output_dir / "evaluation.json").is_file()
