from pathlib import Path

import pytest
import pandas as pd

from modality_training import REQUIRED_NON_CXR_CATEGORIES, SPLITS
from scripts import prepare_modality_data as preparation


def _records():
    records = {}
    for split in SPLITS:
        for category in REQUIRED_NON_CXR_CATEGORIES:
            for index in range(2):
                image_id = f"{split}-{category}-{index}"
                records[image_id] = {
                    "image_id": image_id,
                    "group_id": f"{split}-{category}-author-{index}",
                    "split": split,
                    "categories": [category],
                    "original_url": f"https://images.example/{image_id}.jpg",
                    "landing_url": f"https://source.example/{image_id}",
                    "license": preparation.OPEN_IMAGES_LICENSE,
                    "author": f"author-{index}",
                    "title": category,
                    "subset": "validation",
                }
    return records


def test_selects_exact_licensed_negative_quotas_with_group_safe_splits():
    targets = {
        split: {category: 1 for category in REQUIRED_NON_CXR_CATEGORIES}
        for split in SPLITS
    }
    selected = preparation._select_openimages_by_split(_records(), targets)
    assert len(selected) == len(SPLITS) * len(REQUIRED_NON_CXR_CATEGORIES)
    assert all(row["license"] == preparation.OPEN_IMAGES_LICENSE for row in selected)
    assert {
        (row["split"], row["category"]) for row in selected
    } == {
        (split, category)
        for split in SPLITS
        for category in REQUIRED_NON_CXR_CATEGORIES
    }


def test_selection_rejects_unlicensed_or_missing_attribution_candidates():
    records = _records()
    records["train-cars-0"]["license"] = "unknown"
    records["train-cars-1"]["author"] = ""
    targets = {
        split: {category: 1 for category in REQUIRED_NON_CXR_CATEGORIES}
        for split in SPLITS
    }
    with pytest.raises(preparation.PreparationError, match="0/1 unique licensed cars"):
        preparation._select_openimages_by_split(records, targets)


def test_openimages_group_assignment_is_reproducible_and_exclusive():
    first = preparation._openimages_split("https://flickr.example/author/123")
    second = preparation._openimages_split("https://flickr.example/author/123")
    assert first == second
    assert first in SPLITS


def test_commons_author_group_split_is_deterministic():
    group = "commons-artist:photo maker"
    assert preparation._commons_split(group) == preparation._commons_split(group)
    assert preparation._commons_split(group) in SPLITS


@pytest.mark.parametrize(
    ("license_url", "license_name", "expected"),
    [
        (
            "https://creativecommons.org/licenses/by/4.0/",
            "CC BY 4.0",
            "CC BY 4.0",
        ),
        (
            "https://creativecommons.org/licenses/by/2.0/",
            "CC BY 2.0",
            "CC BY 2.0",
        ),
        (
            "http://creativecommons.org/publicdomain/zero/1.0/deed.en",
            "CC0",
            "CC0",
        ),
        (
            "https://creativecommons.org/publicdomain/zero/1.0/deed.en",
            "CC0",
            "CC0",
        ),
        (
            "https://creativecommons.org/licenses/by-sa/4.0",
            "CC BY-SA 4.0",
            None,
        ),
        (
            "https://creativecommons.org/licenses/by-nc/4.0",
            "CC BY-NC 4.0",
            None,
        ),
        (
            "https://creativecommons.org/licenses/by/4.0",
            "CC BY-SA 4.0",
            None,
        ),
        ("", "CC BY 4.0", None),
    ],
)
def test_commons_license_filter_is_an_exact_per_file_allowlist(
    license_url, license_name, expected
):
    metadata = {
        "LicenseUrl": {"value": license_url},
        "LicenseShortName": {"value": license_name},
    }
    actual = preparation._commons_license(metadata)
    if expected is None:
        assert actual is None
    else:
        assert actual is not None
        assert actual[0] == expected


def test_commons_selection_preserves_group_splits_and_all_categories():
    records = {}
    for split in SPLITS:
        for category in REQUIRED_NON_CXR_CATEGORIES:
            image_id = f"{split}-{category}"
            records[image_id] = {
                "image_id": image_id,
                "categories": {category},
                "split": split,
                "group_id": f"commons-artist:{split}-{category}",
                "license": "CC BY 4.0",
                "landing_url": f"https://commons.wikimedia.org/wiki/File:{image_id}",
                "original_url": f"https://upload.wikimedia.org/{image_id}.jpg",
                "author": f"{split}-{category}",
                "title": image_id,
            }
    targets = {
        split: {category: 1 for category in REQUIRED_NON_CXR_CATEGORIES}
        for split in SPLITS
    }
    selected = preparation._select_commons_by_split(records, targets)
    assert len(selected) == len(SPLITS) * len(REQUIRED_NON_CXR_CATEGORIES)
    assert len({row["image_id"] for row in selected}) == len(selected)
    assert {
        (row["split"], row["category"]) for row in selected
    } == {
        (split, category)
        for split in SPLITS
        for category in REQUIRED_NON_CXR_CATEGORIES
    }


def test_commons_selection_fails_closed_on_missing_per_split_quota():
    records = {
        "person-image": {
            "image_id": "person-image",
            "categories": {"people"},
            "split": "test",
        }
    }
    targets = {
        split: {category: 0 for category in REQUIRED_NON_CXR_CATEGORIES}
        for split in SPLITS
    }
    targets["test"]["people"] = 1
    targets["test"]["cars"] = 1
    with pytest.raises(preparation.PreparationError, match="0/1 distinct allowlisted cars"):
        preparation._select_commons_by_split(records, targets)


def test_commons_metadata_uses_file_license_artist_and_category_records(monkeypatch):
    title = "File:Commons sample.jpg"
    image_url = "https://upload.wikimedia.org/sample.jpg"
    page_url = "https://commons.wikimedia.org/wiki/File:Commons_sample.jpg"
    records = {}
    preparation._add_commons_file_records(
        records,
        [
            {
                "title": title,
                "imageinfo": [
                    {
                        "url": image_url,
                        "descriptionurl": page_url,
                        "mime": "image/jpeg",
                        "size": 1234,
                        "extmetadata": {
                            "LicenseUrl": {
                                "value": "https://creativecommons.org/licenses/by/4.0"
                            },
                            "LicenseShortName": {"value": "CC BY 4.0"},
                            "Artist": {
                                "value": '<a href="/wiki/User:PhotoMaker">Photo Maker</a>'
                            },
                        },
                    }
                ],
            }
        ],
        "cars",
    )
    record = records[title]
    assert record["license"] == "CC BY 4.0"
    assert record["author"] == "Photo Maker"
    assert record["landing_url"] == page_url
    assert record["original_url"] == image_url
    assert record["size_bytes"] == 1234


def test_commons_metadata_rejects_non_allowlisted_and_unattributed_files(monkeypatch):
    title = "File:Commons sample.jpg"
    records = {}
    preparation._add_commons_file_records(
        records,
        [
            {
                "title": title,
                "imageinfo": [
                    {
                        "url": "https://upload.wikimedia.org/sample.jpg",
                        "descriptionurl": (
                            "https://commons.wikimedia.org/wiki/File:Commons_sample.jpg"
                        ),
                        "mime": "image/jpeg",
                        "extmetadata": {
                            "LicenseUrl": {
                                "value": "https://creativecommons.org/licenses/by-nc/4.0"
                            },
                            "LicenseShortName": {"value": "CC BY-NC 4.0"},
                            "Artist": {"value": "Photo Maker"},
                        },
                    }
                ],
            }
        ],
        "cars",
    )
    assert records == {}


def test_commons_check_only_uses_per_file_gate_and_downloads_no_images(
    monkeypatch, tmp_path: Path
):
    selected = []
    for split in SPLITS:
        for category, count in preparation.NEGATIVE_TARGETS_BY_SPLIT[split].items():
            for index in range(count):
                selected.append(
                    {
                        "split": split,
                        "category": category,
                        "label": "non_cxr",
                        "image_id": f"{split}-{category}-{index}",
                    }
                )
    monkeypatch.setattr(
        preparation,
        "_commons_metadata",
        lambda _session, _cache_dir: (
            {},
            {},
            {"complete": True, "quotas_met": True, "stop_reason": None},
        ),
    )
    monkeypatch.setattr(
        preparation,
        "_select_commons_by_split",
        lambda _records, _targets: selected,
    )
    monkeypatch.setattr(
        preparation,
        "_commons_quota_availability",
        lambda _records, targets: targets,
    )
    monkeypatch.setattr(
        preparation,
        "_collect_taix_rows",
        lambda _session, targets: ([], dict(targets)),
    )
    monkeypatch.setattr(
        preparation,
        "_validate_license_review",
        lambda *_args: pytest.fail("Open Images review must not be used for Commons"),
    )
    report = preparation.prepare_dataset(
        tmp_path / "prepared",
        check_only=True,
        photo_source="commons",
    )
    assert report["photo_source"] == "commons"
    assert report["images_downloaded"] == 0
    assert report["license"]["negative_class"] == sorted(
        set(preparation.COMMONS_LICENSES.values())
    )


def test_commons_api_query_cache_avoids_repeat_requests(tmp_path: Path):
    import json

    class Response:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            assert chunk_size == 64 * 1024
            yield json.dumps({"query": {"marker": "cached"}}).encode("utf-8")

        def close(self):
            pass

    class Session:
        calls = 0

        def get(self, *_args, **_kwargs):
            self.calls += 1
            assert _kwargs["timeout"] == preparation.COMMONS_REQUEST_TIMEOUT
            assert _kwargs["stream"] is True
            return Response()

    session = Session()
    budget = preparation.CommonsScanBudget(max_api_requests=2)
    params = {"action": "query", "cmtitle": "Category:Cars"}
    first = preparation._commons_cached_query(session, params, tmp_path, budget)
    second = preparation._commons_cached_query(session, params, tmp_path, budget)
    assert first == second
    assert session.calls == 1
    assert budget.api_requests == 1
    assert budget.cache_hits == 1


def test_commons_api_requests_obey_hard_request_cap(tmp_path: Path):
    class Session:
        def get(self, *_args, **_kwargs):
            pytest.fail("request cap must block before network access")

    budget = preparation.CommonsScanBudget(max_api_requests=0)
    with pytest.raises(preparation.PreparationError, match="request limit of 0"):
        preparation._commons_cached_query(
            Session(),
            {"action": "query", "cmtitle": "Category:Cars"},
            tmp_path,
            budget,
        )


def test_commons_cached_query_obeys_elapsed_time_limit(tmp_path: Path):
    class Session:
        def get(self, *_args, **_kwargs):
            pytest.fail("elapsed-time limit must block before network access")

    budget = preparation.CommonsScanBudget(max_elapsed_seconds=-1)
    with pytest.raises(preparation.PreparationError, match="elapsed-time limit"):
        preparation._commons_cached_query(
            Session(),
            {"action": "query", "cmtitle": "Category:Cars"},
            tmp_path,
            budget,
        )


def test_completed_commons_roots_resume_without_api_requests(tmp_path: Path):
    completed = {
        f"{category}:Category:{root}"
        for category, roots in preparation.COMMONS_CATEGORY_ROOTS.items()
        for root in roots
    }
    state_path = tmp_path / "commons-scan-state.json"
    preparation._write_commons_state(state_path, {}, completed)

    class Session:
        def get(self, *_args, **_kwargs):
            pytest.fail("completed roots must be skipped from the resume state")

    records, counts, scan = preparation._commons_metadata(
        Session(),
        tmp_path,
        budget=preparation.CommonsScanBudget(max_api_requests=0),
    )
    assert records == {}
    assert set(counts.values()) == {0}
    assert scan["complete"] is True
    assert scan["api_requests"] == 0


def test_commons_preflight_reports_partial_counts_when_request_cap_stops_scan(
    tmp_path: Path,
):
    class Session:
        def get(self, *_args, **_kwargs):
            pytest.fail("zero request cap must not access the network")

    _, counts, scan = preparation._commons_metadata(
        Session(),
        tmp_path,
        budget=preparation.CommonsScanBudget(max_api_requests=0),
    )
    assert set(counts.values()) == {0}
    assert scan["complete"] is False
    assert scan["quotas_met"] is False
    assert scan["stop_reason"] == "API request limit of 0 reached"


def test_commons_check_only_reports_shortage_without_downloading_or_claiming_quotas(
    monkeypatch, tmp_path: Path
):
    counts = {f"{split}:{category}": 0 for split in SPLITS for category in REQUIRED_NON_CXR_CATEGORIES}
    monkeypatch.setattr(
        preparation,
        "_commons_metadata",
        lambda _session, _cache_dir: (
            {},
            counts,
            {
                "complete": False,
                "quotas_met": False,
                "stop_reason": "API request limit reached",
            },
        ),
    )
    monkeypatch.setattr(
        preparation,
        "_collect_taix_rows",
        lambda *_args: pytest.fail("CXR metadata need not be queried for failed photo quotas"),
    )
    monkeypatch.setattr(
        preparation,
        "_download_selected",
        lambda *_args: pytest.fail("check-only must not download images"),
    )
    report = preparation.prepare_dataset(
        tmp_path / "preflight",
        photo_source="commons",
        check_only=True,
    )
    assert report["quota_status"] == "not_met"
    assert report["commons_scan"]["complete"] is False
    assert report["license_filtered_candidates"] == counts
    assert report["images_downloaded"] == 0


def test_default_negative_targets_preserve_calibration_and_test_minimums():
    assert {
        split: sum(preparation.NEGATIVE_TARGETS_BY_SPLIT[split].values())
        for split in SPLITS
    } == {"train": 1000, "validation": 1000, "calibration": 3000, "test": 3000}


def test_license_review_requires_exact_sample_ids_and_per_image_review(tmp_path: Path):
    rows = [
        {
            "image_id": "sample-1",
            "landing_url": "https://source.example/sample-1",
        }
    ]
    review_path = tmp_path / "review.csv"
    with pytest.raises(preparation.PreparationError, match="not been independently reviewed"):
        preparation._validate_license_review(review_path, rows)

    review = pd.DataFrame(
        [
            {
                "image_id": "sample-1",
                "verified_license": preparation.OPEN_IMAGES_LICENSE,
                "reviewed_landing_url": "https://source.example/sample-1",
                "reviewed_at": "2026-10-25T00:00:00Z",
                "reviewer": "test reviewer",
            }
        ]
    )
    review.to_csv(review_path, index=False)
    result = preparation._validate_license_review(review_path, rows)
    assert result["reviewed_image_count"] == 1
    assert "does not independently prove" in result["meaning"]

    review.loc[0, "reviewed_landing_url"] = "https://source.example/other"
    review.to_csv(review_path, index=False)
    with pytest.raises(preparation.PreparationError, match="was not reviewed exactly"):
        preparation._validate_license_review(review_path, rows)


def test_review_template_contains_context_needed_for_each_sample(tmp_path: Path):
    path = tmp_path / "review-template.csv"
    preparation._write_license_review_template(
        [
            {
                "image_id": "sample-1",
                "split": "test",
                "category": "cars",
                "original_url": "https://images.example/sample-1.jpg",
                "landing_url": "https://source.example/sample-1",
                "license": preparation.OPEN_IMAGES_LICENSE,
                "author": "sample author",
                "title": "sample title",
            }
        ],
        path,
    )
    template = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert template.loc[0, "category"] == "cars"
    assert template.loc[0, "split"] == "test"
    assert template.loc[0, "verified_license"] == ""


def test_taix_val_patients_are_assigned_to_exactly_one_of_validation_or_calibration():
    for patient in ("patient-a", "patient-b", "patient-c", "patient-d"):
        split = preparation._taix_split(patient, "val")
        assert split in {"validation", "calibration"}
        assert preparation._taix_split(patient, "val") == split
    assert preparation._taix_split("patient", "train") == "train"
    assert preparation._taix_split("patient", "test") == "test"


def test_taix_manifest_records_stable_source_not_expiring_image_url(
    monkeypatch, tmp_path: Path
):
    signed_url = "https://temporary.example/image?token=short-lived-secret"
    stable_source = f"https://huggingface.co/datasets/{preparation.TAIX_DATASET}"
    image = tmp_path / "sample.png"

    def save_sample(_session, received_url, target):
        assert received_url == signed_url
        target.parent.mkdir(parents=True, exist_ok=True)
        image_bytes = b"verified synthetic test bytes"
        target.write_bytes(image_bytes)
        return len(image_bytes), ".png"

    monkeypatch.setattr(preparation, "_fetch_image", save_sample)
    rows, _, errors = preparation._download_selected(
        None,
        [
            {
                "uid": "uid-1",
                "patient_id": "patient-1",
                "split": "train",
                "image_url": signed_url,
                "license": preparation.TAIX_LICENSE,
                "attribution": preparation.TAIX_ATTRIBUTION,
            }
        ],
        tmp_path / "dataset",
    )
    assert not errors
    assert rows[0]["source_url"] == stable_source
    assert rows[0]["source_landing_url"] == stable_source
    assert rows[0]["source_title"] == preparation.TAIX_SOURCE_TITLE
    assert signed_url not in str(rows)


def test_corrupt_download_is_removed(monkeypatch, tmp_path: Path):
    class Response:
        headers = {"Content-Length": "5"}
        content = b"nope!"

    monkeypatch.setattr(preparation, "_download", lambda *_args, **_kwargs: Response())
    target = tmp_path / "images" / "bad.bin"
    with pytest.raises(preparation.PreparationError, match="Corrupt or unsupported"):
        preparation._fetch_image(None, "https://images.example/bad.png", target)
    assert not target.exists()


def test_report_refuses_to_publish_when_required_counts_are_missing():
    rows = [{"split": "test", "label": "non_cxr", "category": "cars"}]
    with pytest.raises(preparation.PreparationError, match="no manifest was published"):
        preparation._verify_downloaded_counts(
            rows,
            {"train": 4, "validation": 4, "calibration": 4, "test": 4},
            {"train": 1, "validation": 1, "calibration": 1, "test": 1},
            errors=["synthetic download failure"],
        )
