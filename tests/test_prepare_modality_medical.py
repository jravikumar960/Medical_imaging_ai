from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from modality_training import validate_manifest
from scripts import prepare_modality_data as preparation


def _medical_manifest(tmp_path, rows):
    staging = tmp_path / "staging"
    staging.mkdir(exist_ok=True)
    manifest_rows = []
    for index, row in enumerate(rows):
        filename = f"medical-{index}.png"
        Image.new("L", (24, 24), 100 + index).save(staging / filename)
        manifest_rows.append(
            {
                "image_id": row.get("image_id", f"medical-{index}"),
                "local_path": filename,
                "group_id": row.get("group_id", f"group-{index}"),
                "category": row.get("category", "ct"),
                "source": row.get("source", "fixture source"),
                "source_title": row.get("source_title", "fixture title"),
                "license": row.get("license", "CC BY 4.0"),
                "attribution": row.get("attribution", "fixture attribution"),
                "license_url": row.get(
                    "license_url", "https://creativecommons.org/licenses/by/4.0/"
                ),
                "source_url": row.get("source_url", "https://example.org/source"),
                "landing_url": row.get("landing_url", "https://example.org/landing"),
                **({"split": row["split"]} if "split" in row else {}),
                **(
                    {"grouping_verified": row["grouping_verified"]}
                    if "grouping_verified" in row
                    else {}
                ),
            }
        )
    manifest_path = tmp_path / "medical_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    return manifest_path, staging


def test_medical_manifest_requires_columns(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    manifest_path = tmp_path / "medical_manifest.csv"
    pd.DataFrame([{"image_id": "x"}]).to_csv(manifest_path, index=False)
    with pytest.raises(preparation.PreparationError, match="missing columns"):
        preparation._load_medical_manifest(manifest_path, staging)


def test_medical_manifest_rejects_path_traversal(tmp_path):
    manifest_path, staging = _medical_manifest(
        tmp_path, [{"local_path": "../outside.png"}]
    )
    frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    frame.loc[0, "local_path"] = "../outside.png"
    frame.to_csv(manifest_path, index=False)
    with pytest.raises(preparation.PreparationError, match="escapes"):
        preparation._load_medical_manifest(manifest_path, staging)
def test_medical_manifest_rejects_invalid_category(tmp_path):
    manifest_path, staging = _medical_manifest(tmp_path, [{"category": "brain"}])
    with pytest.raises(preparation.PreparationError, match="invalid category"):
        preparation._load_medical_manifest(manifest_path, staging)


def test_medical_manifest_rejects_duplicate_hashes(tmp_path):
    manifest_path, staging = _medical_manifest(tmp_path, [{}, {}])
    (staging / "medical-1.png").write_bytes((staging / "medical-0.png").read_bytes())
    with pytest.raises(preparation.PreparationError, match="Duplicate medical image"):
        preparation._load_medical_manifest(manifest_path, staging)


def test_medical_manifest_rejects_split_leakage(tmp_path):
    manifest_path, staging = _medical_manifest(
        tmp_path,
        [
            {"group_id": "shared-group", "split": "train"},
            {"group_id": "shared-group", "split": "test"},
        ],
    )
    with pytest.raises(preparation.PreparationError, match="Group leakage"):
        preparation._load_medical_manifest(manifest_path, staging)


def test_medical_manifest_rejects_missing_license(tmp_path):
    manifest_path, staging = _medical_manifest(tmp_path, [{"license": ""}])
    with pytest.raises(preparation.PreparationError, match="empty license"):
        preparation._load_medical_manifest(manifest_path, staging)


def test_medical_manifest_rejects_missing_file(tmp_path):
    manifest_path, staging = _medical_manifest(tmp_path, [{}])
    (staging / "medical-0.png").unlink()
    with pytest.raises(preparation.PreparationError, match="does not exist"):
        preparation._load_medical_manifest(manifest_path, staging)


def test_medical_manifest_successful_preparation(tmp_path):
    manifest_path, staging = _medical_manifest(
        tmp_path,
        [
            {"category": "ct", "group_id": "patient-1", "grouping_verified": "true"},
            {"category": "mri", "group_id": "patient-2"},
        ],
    )
    rows = preparation._load_medical_manifest(manifest_path, staging)
    assert len(rows) == 2
    assert {row["category"] for row in rows} == {"ct", "mri"}
    assert rows[0]["grouping_verified"] is True
    assert rows[1]["grouping_verified"] is False

    image_root = tmp_path / "dataset"
    manifest_rows, total_bytes, errors = preparation._download_selected(None, rows, image_root)
    assert errors == []
    assert len(manifest_rows) == 2
    assert total_bytes > 0
    assert {row["category"] for row in manifest_rows} == {"ct", "mri"}

    checked = None
    assert len(manifest_rows) == 2


def test_medical_manifest_check_only_preflight(tmp_path, monkeypatch):
    manifest_path, staging = _medical_manifest(
        tmp_path,
        [{"category": "ultrasound", "group_id": "patient-3"}],
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setattr(
        preparation, "NEGATIVE_TARGETS_BY_SPLIT",
        {split: {category: 0 for category in preparation.REQUIRED_NON_CXR_CATEGORIES} for split in preparation.SPLITS},
    )
    monkeypatch.setattr(preparation, "_openimages_metadata", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(preparation, "_select_openimages_by_split", lambda *a, **k: [])
    monkeypatch.setattr(preparation, "_check_quota", lambda *a, **k: None)
    monkeypatch.setattr(
        preparation,
        "_collect_taix_rows",
        lambda *a, **k: (
            [],
            {s: {"planned": t, "queried": False} for s, t in preparation.TAIX_TARGETS.items()},
        ),
    )
    report = preparation.prepare_dataset(
        output_dir,
        check_only=True,
        photo_source="openimages",
        medical_manifest_path=manifest_path,
        medical_staging_dir=staging,
    )
    assert report["medical_manifest_rows"] == 1
    assert report["medical_grouping_status"] in {"verified", "unverified", "mixed"}