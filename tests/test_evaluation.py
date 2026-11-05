import json
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from evaluate import evaluate_manifest, write_results
class _Model:
    pathologies = ["FindingA", "FindingB"]
    def __init__(self, invalid_output=None):
        self.invalid_output = invalid_output
        self.calls = 0
    def __call__(self, tensor):
        self.calls += 1
        if self.invalid_output == "shape":
            return torch.zeros((1, 1))
        if self.invalid_output == "nan":
            return torch.tensor([[float("nan"), 0.5]])
        intensity = (tensor.mean().item() + 1024.0) / 2048.0
        return torch.tensor([[intensity, 1.0 - intensity]])
def _manifest(tmp_path):
    examples = [
        ("train.png", 10, "train", "p-train", 0, 1),
        ("val-neg.png", 51, "validation", "p-val-neg", 0, 1),
        ("val-pos.png", 204, "validation", "p-val-pos", 1, 0),
        ("test-neg.png", 51, "test", "p-test-neg", 0, 1),
        ("test-pos.png", 204, "test", "p-test-pos", 1, 0),
    ]
    rows = []
    for name, pixel, split, patient, finding_a, finding_b in examples:
        Image.new("L", (40, 40), pixel).save(tmp_path / name)
        rows.append({
            "image_path": name,
            "split": split,
            "patient_id": patient,
            "FindingA": finding_a,
            "FindingB": finding_b,
        })
    return pd.DataFrame(rows)


def test_validation_thresholds_and_test_metrics_are_separate(tmp_path):
    model = _Model()
    manifest = _manifest(tmp_path)
    report = evaluate_manifest(manifest, model, tmp_path)
    assert report["validation_images"] == 2
    assert report["test_images"] == 2
    assert report["training_rows_evaluated"] == 0
    assert report["training_images_excluded"] == 1
    assert report["output_labels"] == _Model.pathologies
    assert "external review" in report["checkpoint_training_overlap"]
    assert "Not established" in report["performance_claim_status"]
    assert model.calls == 4  # two validation and two test; the training image was not inferred
    assert report["patient_level_separation_checked"] is True
    assert report["score_interpretation"] == "Raw model scores; not calibrated probabilities."
    for finding in report["findings"]:
        assert finding["threshold_raw_score"] is not None
        assert finding["threshold_status"] == "selected by maximum validation F1"
        assert finding["auroc"] == 1.0
        assert finding["confusion_matrix"] == [[1, 0], [0, 1]]
        assert finding["sensitivity"] == finding["specificity"] == 1.0
        assert finding["precision"] == finding["recall"] == 1.0
        assert finding["f1"] == 1.0
        assert finding["f1_status"] == "defined"
        assert finding["test_missing_labels"] == 0
        assert finding["validation_missing_labels"] == 0

    thresholds = [finding["threshold_raw_score"] for finding in report["findings"]]
    manifest.loc[manifest["image_path"] == "test-neg.png", "FindingA"] = 1
    manifest.loc[manifest["image_path"] == "test-pos.png", "FindingA"] = 0
    Image.new("L", (40, 40), 250).save(tmp_path / "test-neg.png")
    Image.new("L", (40, 40), 5).save(tmp_path / "test-pos.png")
    changed_test_report = evaluate_manifest(manifest, _Model(), tmp_path)
    assert [finding["threshold_raw_score"] for finding in changed_test_report["findings"]] == thresholds


def test_missing_label_column_is_rejected(tmp_path):
    manifest = _manifest(tmp_path).drop(columns="FindingB")
    with pytest.raises(ValueError, match="FindingB"):
        evaluate_manifest(manifest, _Model(), tmp_path)


def test_invalid_model_label_metadata_is_rejected(tmp_path):
    class EmptyLabelModel(_Model):
        pathologies = ["FindingA", ""]

    with pytest.raises(ValueError, match="non-empty, unique string"):
        evaluate_manifest(_manifest(tmp_path), EmptyLabelModel(), tmp_path)


def test_invalid_paths_are_rejected(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.loc[manifest["split"] == "test", "image_path"] = "missing.png"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        evaluate_manifest(manifest, _Model(), tmp_path)


def test_relative_paths_cannot_escape_image_root(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.loc[manifest["split"] == "test", "image_path"] = "../outside.png"
    with pytest.raises(ValueError, match="escapes --image-root"):
        evaluate_manifest(manifest, _Model(), tmp_path)


def test_duplicate_validation_or_test_image_paths_are_rejected(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.loc[manifest["image_path"] == "test-pos.png", "image_path"] = "test-neg.png"
    duplicate = manifest.index[manifest["image_path"] == "test-neg.png"][-1]
    manifest.loc[duplicate, ["FindingA", "FindingB"]] = [0, 1]
    with pytest.raises(ValueError, match="Duplicate image_path"):
        evaluate_manifest(manifest, _Model(), tmp_path)


def test_train_and_test_cannot_reuse_the_same_image(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.loc[manifest["image_path"] == "train.png", "image_path"] = "test-pos.png"
    with pytest.raises(ValueError, match="Conflicting labels.*train.*test"):
        evaluate_manifest(manifest, _Model(), tmp_path)


def test_duplicate_image_with_conflicting_labels_is_rejected(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.loc[manifest["image_path"] == "test-pos.png", "image_path"] = "test-neg.png"
    manifest.loc[manifest["image_path"] == "test-neg.png", "FindingA"] = [0, 1]
    with pytest.raises(ValueError, match="Conflicting labels for duplicate image_path"):
        evaluate_manifest(manifest, _Model(), tmp_path)


def test_patient_id_column_must_be_complete_when_present(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.loc[manifest["image_path"] == "test-pos.png", "patient_id"] = " "
    with pytest.raises(ValueError, match="non-empty for every manifest row"):
        evaluate_manifest(manifest, _Model(), tmp_path)


def test_validation_single_class_makes_threshold_metrics_undefined(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.loc[manifest["split"] == "validation", "FindingA"] = 0
    report = evaluate_manifest(manifest, _Model(), tmp_path)
    finding = report["findings"][0]
    assert finding["threshold_raw_score"] is None
    assert "requires both classes" in finding["threshold_status"]
    assert finding["test_n"] == 2
    assert finding["auroc"] == 1.0
    assert finding["confusion_matrix"] is None
    assert finding["sensitivity"] is None


def test_test_class_imbalance_reports_undefined_metrics(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.loc[manifest["split"] == "test", "FindingA"] = 1
    report = evaluate_manifest(manifest, _Model(), tmp_path)
    finding = report["findings"][0]
    assert finding["test_positive"] == 2
    assert finding["test_negative"] == 0
    assert finding["auroc"] is None
    assert finding["specificity"] is None
    assert finding["sens" \
    "itivity"] == 0.5


def test_blank_labels_are_excluded_and_all_missing_is_reported(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.loc[manifest["split"] == "test", "FindingB"] = np.nan
    report = evaluate_manifest(manifest, _Model(), tmp_path)
    finding = report["findings"][1]
    assert finding["test_n"] == 0
    assert finding["test_missing_labels"] == 2
    assert finding["validation_missing_labels"] == 0
    assert finding["auroc"] is None
    assert finding["confusion_matrix"] is None
    assert finding["f1"] is None
    assert finding["f1_status"] == "undefined: no labeled test samples"


def test_missing_validation_labels_are_counted(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.loc[manifest["image_path"] == "val-neg.png", "FindingB"] = np.nan
    report = evaluate_manifest(manifest, _Model(), tmp_path)
    finding = report["findings"][1]
    assert finding["validation_missing_labels"] == 1
    assert finding["validation_n"] == 1
    assert finding["threshold_raw_score"] is None
    assert finding["f1"] is None
    assert finding["f1_status"] == "undefined: validation threshold unavailable"


@pytest.mark.parametrize("bad_output", ["shape", "nan"])
def test_invalid_model_outputs_are_rejected(tmp_path, bad_output):
    with pytest.raises(ValueError, match="Invalid model output"):
        evaluate_manifest(_manifest(tmp_path), _Model(invalid_output=bad_output), tmp_path)


def test_patient_overlap_and_nonbinary_label_are_rejected(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.loc[manifest["split"] == "test", "patient_id"] = "p-val-neg"
    with pytest.raises(ValueError, match="Patient-level leakage"):
        evaluate_manifest(manifest, _Model(), tmp_path)

    manifest = _manifest(tmp_path)
    manifest["FindingA"] = manifest["FindingA"].astype(object)
    manifest.loc[manifest["split"] == "validation", "FindingA"] = "uncertain"
    with pytest.raises(ValueError, match="only 0, 1, or blank"):
        evaluate_manifest(manifest, _Model(), tmp_path)


def test_results_are_written_as_machine_readable_json_and_csv(tmp_path):
    report = evaluate_manifest(_manifest(tmp_path), _Model(), tmp_path)
    json_path, csv_path = write_results(report, tmp_path / "results" / "run")
    assert json_path.is_file() and csv_path.is_file()
    saved_json = json.loads(json_path.read_text(encoding="utf-8"))
    saved_csv = pd.read_csv(csv_path)
    assert saved_json["test_images"] == 2
    assert set(saved_csv["finding"]) == {"FindingA", "FindingB"}
