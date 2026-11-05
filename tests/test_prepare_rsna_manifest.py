import pandas as pd
import pytest
from PIL import Image

from prepare_rsna_manifest import MODEL_LABELS, prepare_manifest


def _source(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    image_root = tmp_path / "images"
    image_root.mkdir()
    for exam_id, pixel in [("exam-a", 40), ("exam-b", 80), ("exam-c", 120), ("exam-d", 160)]:
        Image.new("L", (20, 20), pixel).save(image_root / f"{exam_id}.jpg")

    # exam-a has two distinct opacity boxes; this is a legitimate repeated
    # patientId and should become one image-level label row.
    labels = pd.DataFrame([
        {"patientId": "exam-a", "Target": 1, "x": 1, "y": 2, "width": 5, "height": 6},
        {"patientId": "exam-a", "Target": 1, "x": 8, "y": 9, "width": 4, "height": 5},
        {"patientId": "exam-b", "Target": 0, "x": None, "y": None, "width": None, "height": None},
        {"patientId": "exam-c", "Target": 1, "x": 3, "y": 4, "width": 7, "height": 8},
        {"patientId": "exam-d", "Target": 0, "x": None, "y": None, "width": None, "height": None},
    ])
    labels_path = tmp_path / "stage_2_train_labels.csv"
    labels.to_csv(labels_path, index=False)
    split_map = pd.DataFrame([
        {"patientId": "exam-a", "patient_id": "person-a", "split": "train"},
        {"patientId": "exam-b", "patient_id": "person-b", "split": "validation"},
        {"patientId": "exam-c", "patient_id": "person-c", "split": "test"},
        {"patientId": "exam-d", "patient_id": "person-d", "split": "test"},
    ])
    split_path = tmp_path / "split_map.csv"
    split_map.to_csv(split_path, index=False)
    return labels_path, split_path, image_root


def test_preparation_maps_only_supported_labels_and_preserves_source(tmp_path):
    labels, split_map, image_root = _source(tmp_path)
    original_bytes = labels.read_bytes()
    output = tmp_path / "prepared" / "manifest.csv"
    manifest = prepare_manifest(labels, split_map, image_root, output)

    assert output.is_file()
    assert labels.read_bytes() == original_bytes
    assert len(manifest) == 4
    assert manifest.columns.tolist() == ["image_path", "split", "patient_id", *MODEL_LABELS]
    assert set(manifest["split"]) == {"train", "validation", "test"}
    assert manifest.loc[manifest["image_path"] == "exam-a.jpg", "Pneumonia"].iloc[0] == 1
    assert manifest.loc[manifest["image_path"] == "exam-a.jpg", "Lung Opacity"].iloc[0] == 1
    assert manifest["Atelectasis"].isna().all()
    assert manifest["Consolidation"].isna().all()
    assert manifest["patient_id"].tolist().count("person-a") == 1


def test_preparation_rejects_patient_overlap_across_partitions(tmp_path):
    labels, split_map, image_root = _source(tmp_path)
    splits = pd.read_csv(split_map)
    splits.loc[splits["patientId"] == "exam-b", "patient_id"] = "person-a"
    splits.to_csv(split_map, index=False)
    with pytest.raises(ValueError, match="Patient-level leakage"):
        prepare_manifest(labels, split_map, image_root, tmp_path / "manifest.csv")


def test_preparation_rejects_using_exam_id_as_patient_id(tmp_path):
    labels, split_map, image_root = _source(tmp_path)
    splits = pd.read_csv(split_map)
    splits["patient_id"] = splits["patientId"]
    splits.to_csv(split_map, index=False)
    with pytest.raises(ValueError, match="cannot simply copy RSNA patientId"):
        prepare_manifest(labels, split_map, image_root, tmp_path / "manifest.csv")


def test_preparation_rejects_missing_and_invalid_targets(tmp_path):
    labels, split_map, image_root = _source(tmp_path)
    source = pd.read_csv(labels).drop(columns="Target")
    source.to_csv(labels, index=False)
    with pytest.raises(ValueError, match="Target"):
        prepare_manifest(labels, split_map, image_root, tmp_path / "manifest.csv")

    labels, split_map, image_root = _source(tmp_path / "invalid")
    source = pd.read_csv(labels)
    source.loc[0, "Target"] = None
    source.to_csv(labels, index=False)
    with pytest.raises(ValueError, match="only explicit 0 or 1"):
        prepare_manifest(labels, split_map, image_root, tmp_path / "missing-target-manifest.csv")

    labels, split_map, image_root = _source(tmp_path / "invalid_value")
    source = pd.read_csv(labels)
    source.loc[0, "Target"] = 2
    source.to_csv(labels, index=False)
    with pytest.raises(ValueError, match="only explicit 0 or 1"):
        prepare_manifest(labels, split_map, image_root, tmp_path / "invalid-value-manifest.csv")


def test_preparation_rejects_conflicting_targets_and_exact_duplicate_rows(tmp_path):
    labels, split_map, image_root = _source(tmp_path)
    source = pd.read_csv(labels)
    source.loc[1, "Target"] = 0
    source.to_csv(labels, index=False)
    with pytest.raises(ValueError, match="Conflicting Target"):
        prepare_manifest(labels, split_map, image_root, tmp_path / "manifest.csv")

    labels, split_map, image_root = _source(tmp_path / "duplicate")
    source = pd.read_csv(labels)
    source = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    source.to_csv(labels, index=False)
    with pytest.raises(ValueError, match="duplicate records"):
        prepare_manifest(labels, split_map, image_root, tmp_path / "duplicate-manifest.csv")


def test_preparation_rejects_unmapped_ids_missing_images_and_bad_splits(tmp_path):
    labels, split_map, image_root = _source(tmp_path)
    source = pd.read_csv(split_map).iloc[:-1]
    source.to_csv(split_map, index=False)
    with pytest.raises(ValueError, match="cover labels exactly"):
        prepare_manifest(labels, split_map, image_root, tmp_path / "manifest.csv")

    labels, split_map, image_root = _source(tmp_path / "missing")
    (image_root / "exam-c.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        prepare_manifest(labels, split_map, image_root, tmp_path / "missing-manifest.csv")

    labels, split_map, image_root = _source(tmp_path / "bad_split")
    source = pd.read_csv(split_map)
    source.loc[0, "split"] = "holdout"
    source.to_csv(split_map, index=False)
    with pytest.raises(ValueError, match="Unsupported split"):
        prepare_manifest(labels, split_map, image_root, tmp_path / "bad-split-manifest.csv")


def test_preparation_refuses_to_overwrite_source_or_existing_output(tmp_path):
    labels, split_map, image_root = _source(tmp_path)
    with pytest.raises(ValueError, match="must not overwrite"):
        prepare_manifest(labels, split_map, image_root, labels)
    output = tmp_path / "manifest.csv"
    output.write_text("keep me", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_manifest(labels, split_map, image_root, output)
    assert output.read_text(encoding="utf-8") == "keep me"
