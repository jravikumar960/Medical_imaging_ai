"""Prepare a small, license-attributed CXR/photo modality dataset.

TAIX-Ray is retrieved through its public Hugging Face dataset viewer, not as
full Parquet shards. Negative examples can come from Open Images (which requires
a completed per-image review file) or Wikimedia Commons (strict CC BY 2.0/2.5/
3.0/4.0 and CC0 per-file API metadata allowlist). This script prepares research
data only and does not enable application inference.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import html
import io
import json
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modality_training import (
    MEDICAL_MANIFEST_REQUIRED_COLUMNS,
    MEDICAL_NON_CXR_CATEGORIES,
    REQUIRED_NON_CXR_CATEGORIES,
    SPLITS,
    validate_manifest,
)

SEED = 20261025
TAIX_DATASET = "TLAIM/TAIX-Ray"
TAIX_REVISION = "7f43720e2c7ab6400608880c0cb8b950ac0e5afb"
TAIX_LICENSE = "CC BY 4.0"
TAIX_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
TAIX_SOURCE_TITLE = (
    "TAIX-Ray: A comprehensive bedside chest radiography dataset with structured, "
    "itemized and graded radiologic reports"
)
TAIX_ATTRIBUTION = (
    f"Truhn et al., {TAIX_SOURCE_TITLE}, Scientific Data 13:632 (2026), "
    "https://doi.org/10.1038/s41597-026-07271-7"
)
TAIX_ROWS_API = "https://datasets-server.huggingface.co/rows"
OPEN_IMAGES_BASE = "https://storage.googleapis.com/openimages/2018_04"
OPEN_IMAGES_LICENSE = "https://creativecommons.org/licenses/by/2.0/"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
COMMONS_LICENSES = {
    f"{scheme}://creativecommons.org/licenses/by/{version}": f"CC BY {version}"
    for scheme in ("http", "https")
    for version in ("2.0", "2.5", "3.0", "4.0")
} | {
    "https://creativecommons.org/publicdomain/zero/1.0": "CC0",
    "http://creativecommons.org/publicdomain/zero/1.0": "CC0",
    "https://creativecommons.org/publicdomain/zero/1.0/deed.en": "CC0",
    "http://creativecommons.org/publicdomain/zero/1.0/deed.en": "CC0",
}
COMMONS_CATEGORY_ROOTS: dict[str, tuple[str, ...]] = {
    "cars": ("Automobiles", "Cars"),
    "buildings": ("Buildings",),
    "people": ("People",),
    "everyday_objects": ("Household objects", "Objects", "Furniture", "Tools"),
}
COMMONS_CATEGORY_DEPTH = 3
COMMONS_MAX_CATEGORIES_PER_ROOT = 100
COMMONS_MAX_FILES_PER_CATEGORY = 10000
COMMONS_METADATA_BATCH_SIZE = 200
COMMONS_MAX_API_REQUESTS = 400
COMMONS_MAX_PAGE_VISITS = 600
COMMONS_MAX_ELAPSED_SECONDS = 300
COMMONS_REQUEST_TIMEOUT = (3, 5)
COMMONS_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
COMMONS_CACHE_TTL_SECONDS = 24 * 60 * 60
COMMONS_SCAN_STATE_VERSION = 1
COMMONS_GROUP_BUCKETS = {
    "train": (0, 1250),
    "validation": (1250, 2500),
    "calibration": (2500, 6250),
    "test": (6250, 10000),
}
MAX_IMAGE_BYTES = 25 * 1024 * 1024
OPEN_IMAGES_CATEGORIES: dict[str, tuple[str, ...]] = {
    "cars": ("Car",),
    "buildings": ("Building", "House", "Skyscraper"),
    "people": ("Person",),
    "everyday_objects": (
        "Apple",
        "Backpack",
        "Bed",
        "Book",
        "Bottle",
        "Bowl",
        "Chair",
        "Clock",
        "Cup",
        "Keyboard",
        "Lamp",
        "Laptop",
        "Mobile phone",
        "Table",
        "Telephone",
    ),
}
OPEN_IMAGES_SPLITS = ("validation", "test")
OPEN_IMAGES_GROUP_BUCKETS = {
    "train": (0, 1000),
    "validation": (1000, 2000),
    "calibration": (2000, 6000),
    "test": (6000, 10000),
}
TAIX_TARGETS = {
    "train": 400,
    "validation": 100,
    "calibration": 150,
    "test": 150,
}
NEGATIVE_TARGETS_BY_SPLIT = {
    "train": {category: 250 for category in REQUIRED_NON_CXR_CATEGORIES},
    "validation": {category: 250 for category in REQUIRED_NON_CXR_CATEGORIES},
    "calibration": {category: 750 for category in REQUIRED_NON_CXR_CATEGORIES},
    "test": {category: 750 for category in REQUIRED_NON_CXR_CATEGORIES},
}
MEDICAL_NEGATIVE_TARGETS_BY_SPLIT = {
    "train": {category: 0 for category in MEDICAL_NON_CXR_CATEGORIES},
    "validation": {category: 0 for category in MEDICAL_NON_CXR_CATEGORIES},
    "calibration": {category: 0 for category in MEDICAL_NON_CXR_CATEGORIES},
    "test": {category: 0 for category in MEDICAL_NON_CXR_CATEGORIES},
}
DEFAULT_MEDICAL_STAGING_DIRNAME = "medical-staging"


class PreparationError(RuntimeError):
    """Raised when approved data or quotas are unavailable."""


@dataclasses.dataclass
class CommonsScanBudget:
    max_api_requests: int = COMMONS_MAX_API_REQUESTS
    max_page_visits: int = COMMONS_MAX_PAGE_VISITS
    max_elapsed_seconds: int = COMMONS_MAX_ELAPSED_SECONDS
    started_at: float = dataclasses.field(default_factory=time.monotonic)
    api_requests: int = 0
    page_visits: int = 0
    cache_hits: int = 0
    stop_reason: str | None = None

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def check(self) -> None:
        if self.elapsed_seconds >= self.max_elapsed_seconds:
            self.stop_reason = (
                f"elapsed-time limit of {self.max_elapsed_seconds} seconds reached"
            )
            raise PreparationError(self.stop_reason)

    def check_request(self) -> None:
        self.check()
        if self.api_requests >= self.max_api_requests:
            self.stop_reason = (
                f"API request limit of {self.max_api_requests} reached"
            )
            raise PreparationError(self.stop_reason)

    def visit_page(self) -> None:
        self.check()
        if self.page_visits >= self.max_page_visits:
            self.stop_reason = (
                f"pagination visit limit of {self.max_page_visits} reached"
            )
            raise PreparationError(self.stop_reason)
        self.page_visits += 1


def _download(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: tuple[int, int] = (15, 90),
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(1.5 * (attempt + 1))
    if isinstance(last_error, requests.HTTPError) and last_error.response is not None:
        detail = f"HTTP {last_error.response.status_code}"
    else:
        detail = type(last_error).__name__ if last_error else "unknown error"
    raise PreparationError(f"Could not download remote source ({detail}).") from last_error


def _cached_csv(
    session: requests.Session, url: str, cache_path: Path
) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.is_file():
        response = _download(session, url)
        temporary_path = cache_path.with_suffix(cache_path.suffix + ".part")
        temporary_path.write_bytes(response.content)
        temporary_path.replace(cache_path)
    return pd.read_csv(cache_path, dtype=str, keep_default_na=False)


def _split_bucket(group_id: str, seed: int = SEED) -> int:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10000


def _openimages_split(group_id: str) -> str:
    bucket = _split_bucket(f"openimages:{group_id}")
    for split, (lower, upper) in OPEN_IMAGES_GROUP_BUCKETS.items():
        if lower <= bucket < upper:
            return split
    raise AssertionError("All deterministic buckets must map to a split.")


def _commons_split(group_id: str) -> str:
    bucket = _split_bucket(f"wikimedia-commons:{group_id}")
    for split, (lower, upper) in COMMONS_GROUP_BUCKETS.items():
        if lower <= bucket < upper:
            return split
    raise AssertionError("All deterministic buckets must map to a split.")


def _medical_split(group_id: str) -> str:
    bucket = _split_bucket(f"medical:{group_id}")
    for split, (lower, upper) in OPEN_IMAGES_GROUP_BUCKETS.items():
        if lower <= bucket < upper:
            return split
    raise AssertionError("All deterministic buckets must map to a split.")


def _human_photo_title_is_suspicious(title: str) -> bool:
    normalized = title.casefold().replace("-", " ").replace("_", " ")
    return any(
        term in normalized
        for term in ("x ray", "xray", "radiograph", "radiography", "medical scan")
    )


def _openimages_metadata(
    session: requests.Session, cache_dir: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    descriptions = _cached_csv(
        session,
        f"{OPEN_IMAGES_BASE}/class-descriptions.csv",
        cache_dir / "openimages-class-descriptions.csv",
    )
    class_id_by_name = {
        str(row[1]).strip().casefold(): str(row[0]).strip()
        for row in descriptions.itertuples(index=False, name=None)
    }
    class_ids: dict[str, set[str]] = {}
    for category, names in OPEN_IMAGES_CATEGORIES.items():
        ids = {
            class_id_by_name[name.casefold()]
            for name in names
            if name.casefold() in class_id_by_name
        }
        if not ids:
            raise PreparationError(f"Open Images classes missing for category {category}.")
        class_ids[category] = ids

    annotations_by_id: dict[str, set[str]] = {}
    relevant_ids = set().union(*class_ids.values())
    for split_name in OPEN_IMAGES_SPLITS:
        annotation_path = cache_dir / f"openimages-{split_name}-human-labels.csv"
        if not annotation_path.is_file():
            url = (
                f"{OPEN_IMAGES_BASE}/{split_name}/"
                f"{split_name}-annotations-human-imagelabels.csv"
            )
            response = _download(session, url)
            temporary_path = annotation_path.with_suffix(".csv.part")
            temporary_path.write_bytes(response.content)
            temporary_path.replace(annotation_path)
        for chunk in pd.read_csv(
            annotation_path,
            dtype={"ImageID": str, "Source": str, "LabelName": str, "Confidence": str},
            chunksize=250000,
        ):
            verified = chunk.loc[
                chunk["Source"].eq("verification")
                & chunk["Confidence"].eq("1")
                & chunk["LabelName"].isin(relevant_ids),
                ["ImageID", "LabelName"],
            ]
            for image_id, label_id in verified.itertuples(index=False, name=None):
                annotations_by_id.setdefault(str(image_id), set()).add(str(label_id))

    image_records: dict[str, dict[str, Any]] = {}
    for split_name in OPEN_IMAGES_SPLITS:
        file_name = f"openimages-{split_name}-image-metadata.csv"
        metadata_path = cache_dir / file_name
        url = f"{OPEN_IMAGES_BASE}/{split_name}/{split_name}-images-with-rotation.csv"
        metadata = _cached_csv(session, url, metadata_path)
        for record in metadata.to_dict("records"):
            image_id = record["ImageID"]
            if record["License"] != OPEN_IMAGES_LICENSE:
                continue
            profile = record["AuthorProfileURL"].strip()
            author = record["Author"].strip()
            if not profile and not author:
                continue
            if _human_photo_title_is_suspicious(record["Title"]):
                continue
            labels = annotations_by_id.get(image_id, set())
            categories = [
                category
                for category, ids in class_ids.items()
                if labels.intersection(ids)
            ]
            if not categories:
                continue
            group_id = profile or author
            image_records[image_id] = {
                "image_id": image_id,
                "group_id": group_id,
                "split": _openimages_split(group_id),
                "categories": categories,
                "original_url": record["OriginalURL"],
                "landing_url": record["OriginalLandingURL"],
                "license": OPEN_IMAGES_LICENSE,
                "author": author,
                "title": record["Title"],
                "subset": split_name,
            }

    counts: dict[str, int] = {}
    for split in SPLITS:
        for category in REQUIRED_NON_CXR_CATEGORIES:
            counts[f"{split}:{category}"] = sum(
                split == record["split"] and category in record["categories"]
                for record in image_records.values()
            )
    return image_records, counts


def _commons_license(extmetadata: dict[str, Any]) -> tuple[str, str] | None:
    license_url = str(
        (extmetadata.get("LicenseUrl") or {}).get("value") or ""
    ).strip()
    license_name = str(
        (extmetadata.get("LicenseShortName") or {}).get("value") or ""
    ).strip()
    normalized_url = license_url.rstrip("/")
    if normalized_url.endswith("/deed.en"):
        normalized_url = normalized_url[: -len("/deed.en")].rstrip("/")
    canonical_name = COMMONS_LICENSES.get(normalized_url)
    if canonical_name is None or license_name != canonical_name:
        return None
    return canonical_name, license_url


def _commons_artist(extmetadata: dict[str, Any]) -> str:
    value = str((extmetadata.get("Artist") or {}).get("value") or "")
    text = html.unescape(re.sub(r"<[^>]*>", " ", value))
    return re.sub(r"\s+", " ", text).strip()


def _add_commons_file_records(
    records: dict[str, dict[str, Any]],
    pages: Any,
    category: str,
) -> None:
    for page in pages:
        title = str(page.get("title") or "")
        if not title:
            continue
        image_info = (page.get("imageinfo") or [{}])[0]
        mime_type = str(image_info.get("mime") or "")
        if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
            continue
        metadata = image_info.get("extmetadata") or {}
        license_info = _commons_license(metadata)
        artist = _commons_artist(metadata)
        image_url = str(image_info.get("url") or "")
        description_url = str(image_info.get("descriptionurl") or "")
        if (
            license_info is None
            or not artist
            or not image_url.startswith("https://")
            or not description_url.startswith(
                "https://commons.wikimedia.org/wiki/File:"
            )
        ):
            continue
        license_name, license_url = license_info
        group_id = f"commons-artist:{artist.casefold()}"
        source_id = title.removeprefix("File:")
        record = records.setdefault(
            title,
            {
                "image_id": title,
                "commons_file_title": title,
                "categories": set(),
                "category": "",
                "group_id": group_id,
                "split": _commons_split(group_id),
                "original_url": image_url,
                "landing_url": description_url,
                "license": license_name,
                "license_url": license_url,
                "author": artist,
                "title": source_id,
                "subset": "Wikimedia Commons",
                "source": "Wikimedia Commons",
                "size_bytes": int(image_info.get("size") or 0),
            },
        )
        record["categories"].add(category)


def _write_commons_state(path: Path, records: dict[str, dict[str, Any]], completed_roots: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": COMMONS_SCAN_STATE_VERSION,
        "saved_at": time.time(),
        "completed_roots": sorted(completed_roots),
        "records": {
            title: {
                **record,
                "categories": sorted(record["categories"]),
            }
            for title, record in records.items()
        },
    }
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(path)


def _read_commons_state(path: Path) -> tuple[dict[str, dict[str, Any]], set[str]]:
    if not path.is_file():
        return {}, set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    saved_at = float(payload.get("saved_at", 0))
    if (
        payload.get("version") != COMMONS_SCAN_STATE_VERSION
        or time.time() - saved_at > COMMONS_CACHE_TTL_SECONDS
    ):
        return {}, set()
    records = payload.get("records", {})
    for record in records.values():
        record["categories"] = set(record.get("categories", []))
    return records, set(payload.get("completed_roots", []))


def _commons_cached_query(
    session: requests.Session,
    params: dict[str, Any],
    cache_dir: Path,
    budget: CommonsScanBudget,
) -> dict[str, Any]:
    budget.visit_page()
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(
        json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_path = cache_dir / f"{key}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            time.time() - float(cached.get("fetched_at", 0))
            <= COMMONS_CACHE_TTL_SECONDS
        ):
            budget.cache_hits += 1
            return cached["payload"]

    budget.check_request()
    budget.api_requests += 1
    try:
        response = session.get(
            COMMONS_API,
            params=params,
            timeout=COMMONS_REQUEST_TIMEOUT,
            stream=True,
        )
        response.raise_for_status()
        payload = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            budget.check()
            if not chunk:
                continue
            payload.extend(chunk)
            if len(payload) > COMMONS_MAX_RESPONSE_BYTES:
                raise PreparationError(
                    f"Commons API response exceeds {COMMONS_MAX_RESPONSE_BYTES} bytes."
                )
        result = json.loads(payload)
    except (requests.RequestException, ValueError) as exc:
        raise PreparationError(
            f"Commons API request failed ({type(exc).__name__}); partial metadata is cached."
        ) from exc
    finally:
        if "response" in locals():
            response.close()
    if not isinstance(result, dict) or "error" in result:
        raise PreparationError("Commons API returned an invalid or error response.")
    time.sleep(0.1)
    temporary = cache_path.with_suffix(".json.part")
    temporary.write_text(
        json.dumps({"fetched_at": time.time(), "payload": result}),
        encoding="utf-8",
    )
    temporary.replace(cache_path)
    return result


def _commons_metadata(
    session: requests.Session,
    cache_dir: Path,
    *,
    budget: CommonsScanBudget | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, int], dict[str, Any]]:
    budget = budget or CommonsScanBudget()
    state_path = cache_dir / "commons-scan-state.json"
    records, completed_roots = _read_commons_state(state_path)
    seen_files = {
        category: set() for category in REQUIRED_NON_CXR_CATEGORIES
    }
    quota_available = False
    stopped_reason: str | None = None
    try:
        _select_commons_by_split(records, NEGATIVE_TARGETS_BY_SPLIT)
    except PreparationError:
        pass
    else:
        quota_available = True
    for category, roots in COMMONS_CATEGORY_ROOTS.items():
        if quota_available:
            break
        for root in roots:
            if quota_available:
                break
            root_key = f"{category}:Category:{root}"
            if root_key in completed_roots:
                continue
            queue = [(f"Category:{root}", 0)]
            visited: set[str] = set()
            root_complete = True
            while (
                queue
                and len(visited) < COMMONS_MAX_CATEGORIES_PER_ROOT
                and len(seen_files[category]) < COMMONS_MAX_FILES_PER_CATEGORY
            ):
                category_title, depth = queue.pop(0)
                if category_title in visited:
                    continue
                visited.add(category_title)

                try:
                    continuation: dict[str, Any] = {}
                    while len(seen_files[category]) < COMMONS_MAX_FILES_PER_CATEGORY:
                        params: dict[str, Any] = {
                            "action": "query",
                            "format": "json",
                            "generator": "categorymembers",
                            "gcmtitle": category_title,
                            "gcmtype": "file",
                            "gcmnamespace": 6,
                            "gcmlimit": COMMONS_METADATA_BATCH_SIZE,
                            "prop": "imageinfo",
                            "iiprop": "url|mime|size|extmetadata",
                        }
                        params.update(continuation)
                        result = _commons_cached_query(
                            session, params, cache_dir / "pages", budget
                        )
                        pages = list(result.get("query", {}).get("pages", {}).values())
                        _add_commons_file_records(records, pages, category)
                        for page in pages:
                            title = str(page.get("title") or "")
                            if title:
                                seen_files[category].add(title)
                        if budget.page_visits % 10 == 0:
                            _write_commons_state(
                                state_path, records, completed_roots
                            )
                            print(
                                f"Commons preflight: {category} scanned "
                                f"{len(seen_files[category])} file entries; "
                                f"{sum(category in record['categories'] for record in records.values())} "
                                f"eligible records; API requests={budget.api_requests}, "
                                f"cache hits={budget.cache_hits}, "
                                f"pages={budget.page_visits}/{budget.max_page_visits}, "
                                f"elapsed={budget.elapsed_seconds:.0f}s.",
                                file=sys.stderr,
                                flush=True,
                            )
                        try:
                            _select_commons_by_split(
                                records, NEGATIVE_TARGETS_BY_SPLIT
                            )
                        except PreparationError:
                            pass
                        else:
                            quota_available = True
                            break
                        continuation = result.get("continue", {})
                        if not continuation:
                            break
                    if (
                        len(seen_files[category]) >= COMMONS_MAX_FILES_PER_CATEGORY
                        or len(visited) >= COMMONS_MAX_CATEGORIES_PER_ROOT
                    ):
                        root_complete = False
                except PreparationError as exc:
                    stopped_reason = str(exc)
                    root_complete = False
                    break
                if quota_available:
                    break

                if depth >= COMMONS_CATEGORY_DEPTH:
                    continue
                continuation = {}
                while True:
                    try:
                        result = _commons_cached_query(
                            session,
                            {
                            "action": "query",
                            "format": "json",
                            "list": "categorymembers",
                            "cmtitle": category_title,
                            "cmtype": "subcat",
                            "cmlimit": 500,
                            **continuation,
                        },
                            cache_dir / "pages",
                            budget,
                        )
                    except PreparationError as exc:
                        stopped_reason = str(exc)
                        root_complete = False
                        break
                    for member in result.get("query", {}).get("categorymembers", []):
                        child_title = str(member.get("title") or "")
                        if child_title and child_title not in visited:
                            queue.append((child_title, depth + 1))
                    continuation = result.get("continue", {})
                    if budget.page_visits % 10 == 0:
                        print(
                            f"Commons preflight: exploring {category_title}; "
                            f"pages={budget.page_visits}/{budget.max_page_visits}, "
                            f"API requests={budget.api_requests}, "
                            f"cache hits={budget.cache_hits}, "
                            f"elapsed={budget.elapsed_seconds:.0f}s.",
                            file=sys.stderr,
                            flush=True,
                        )
                    if not continuation:
                        break
                if stopped_reason:
                    break
            if root_complete:
                completed_roots.add(root_key)
            _write_commons_state(state_path, records, completed_roots)
            if quota_available:
                break
            if stopped_reason:
                break
        if quota_available:
            break
        if stopped_reason:
            break

    availability = {
        f"{split}:{category}": sum(
            record["split"] == split and category in record["categories"]
            for record in records.values()
        )
        for split in SPLITS
        for category in REQUIRED_NON_CXR_CATEGORIES
    }
    try:
        _select_commons_by_split(records, NEGATIVE_TARGETS_BY_SPLIT)
        quotas_met = True
        quota_error = None
    except PreparationError as exc:
        quotas_met = False
        quota_error = str(exc)
    scan_info = {
        "complete": not stopped_reason and (
            quota_available or all(
                f"{category}:Category:{root}" in completed_roots
                for category, roots in COMMONS_CATEGORY_ROOTS.items()
                for root in roots
            )
        ),
        "quotas_met": quotas_met,
        "quota_error": quota_error,
        "stop_reason": stopped_reason,
        "api_requests": budget.api_requests,
        "page_visits": budget.page_visits,
        "cache_hits": budget.cache_hits,
        "elapsed_seconds": round(budget.elapsed_seconds, 2),
        "completed_roots": sorted(completed_roots),
        "cache_state": str(state_path),
    }
    return records, availability, scan_info


def _select_commons_by_split(
    records: dict[str, dict[str, Any]],
    targets: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for split in sorted(SPLITS):
        for category in sorted(REQUIRED_NON_CXR_CATEGORIES):
            choices = [
                record
                for record in records.values()
                if record["split"] == split
                and category in record["categories"]
                and record["image_id"] not in used_ids
            ]
            choices.sort(
                key=lambda record: hashlib.sha256(
                    f"{SEED}:commons-select:{split}:{category}:"
                    f"{record['image_id']}".encode("utf-8")
                ).digest()
            )
            needed = targets[split][category]
            if len(choices) < needed:
                raise PreparationError(
                    f"Wikimedia Commons has {len(choices)}/{needed} distinct "
                    f"allowlisted {category} images for {split} after de-duplication."
                )
            for record in choices[:needed]:
                selected.append({**record, "category": category})
                used_ids.add(record["image_id"])
    return selected


def _commons_quota_availability(
    records: dict[str, dict[str, Any]],
    targets: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    availability: dict[str, dict[str, int]] = {}
    used_ids: set[str] = set()
    for split in sorted(SPLITS):
        availability[split] = {}
        for category in sorted(REQUIRED_NON_CXR_CATEGORIES):
            choices = [
                record
                for record in records.values()
                if record["split"] == split
                and category in record["categories"]
                and record["image_id"] not in used_ids
            ]
            choices.sort(
                key=lambda record: hashlib.sha256(
                    f"{SEED}:commons-select:{split}:{category}:"
                    f"{record['image_id']}".encode("utf-8")
                ).digest()
            )
            needed = targets[split][category]
            availability[split][category] = len(choices)
            used_ids.update(record["image_id"] for record in choices[:needed])
    return availability


def _taix_split(patient_id: str, source_split: str) -> str | None:
    if source_split == "train":
        return "train"
    if source_split == "test":
        return "test"
    if source_split in {"val", "validation"}:
        return "validation" if _split_bucket(f"taix-val:{patient_id}") < 5000 else "calibration"
    return None


def _collect_taix_rows(
    session: requests.Session, targets: dict[str, int], seed: int = SEED
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(seed)
    selected: dict[str, dict[str, Any]] = {}
    group_splits: dict[str, str] = {}
    seen_pages: set[tuple[str, int]] = set()
    source_splits = ("train", "val", "test")
    status_response = _download(
        session,
        "https://datasets-server.huggingface.co/splits",
        params={"dataset": TAIX_DATASET},
    )
    status_json = status_response.json()
    available_splits = {
        item["split"]
        for item in status_json.get("splits", [])
        if item["config"] == "default"
    }
    attempts = 0
    while any(
        sum(row["split"] == split for row in selected.values()) < target
        for split, target in targets.items()
    ):
        progress = False
        for source_split in source_splits:
            if source_split not in available_splits:
                continue
            total = {
                "train": 137593,
                "val": 34860,
                "test": 42928,
            }[source_split]
            offset = rng.randrange(0, max(1, total - 100))
            page_key = (source_split, offset)
            if page_key in seen_pages:
                continue
            seen_pages.add(page_key)
            response = _download(
                session,
                TAIX_ROWS_API,
                params={
                    "dataset": TAIX_DATASET,
                    "config": "default",
                    "split": source_split,
                    "revision": TAIX_REVISION,
                    "offset": offset,
                    "length": 100,
                },
            )
            page = response.json()
            for item in page.get("rows", []):
                row = item["row"]
                row_split = str(row.get("Split") or "").strip().casefold()
                expected_source_split = {
                    "train": "train",
                    "val": "val",
                    "test": "test",
                }[source_split]
                if row_split and row_split not in {
                    expected_source_split,
                    "validation" if source_split == "val" else expected_source_split,
                }:
                    continue
                patient_id = str(row.get("PatientID") or "").strip()
                uid = str(row.get("UID") or "").strip()
                if not patient_id or not uid or uid in selected:
                    continue
                target_split = _taix_split(patient_id, source_split)
                if target_split not in targets:
                    continue
                if sum(x["split"] == target_split for x in selected.values()) >= targets[target_split]:
                    continue
                previous = group_splits.setdefault(patient_id, target_split)
                if previous != target_split:
                    continue
                image = row.get("Image") or {}
                image_url = image.get("src")
                if not image_url:
                    continue
                selected[uid] = {
                    "uid": uid,
                    "patient_id": patient_id,
                    "split": target_split,
                    "source_split": source_split,
                    "image_url": image_url,
                    "source": TAIX_DATASET,
                    "license": TAIX_LICENSE,
                    "license_url": TAIX_LICENSE_URL,
                    "attribution": TAIX_ATTRIBUTION,
                }
                progress = True
        attempts += 1
        if not progress and attempts > 400:
            counts = {
                split: sum(row["split"] == split for row in selected.values())
                for split in targets
            }
            raise PreparationError(
                f"Could not obtain enough distinct TAIX-Ray patients after "
                f"{attempts} sampling rounds; got {counts}, need {targets}."
            )
    counts = {
        split: sum(row["split"] == split for row in selected.values())
        for split in targets
    }
    return list(selected.values()), counts


def _fetch_image(
    session: requests.Session, url: str, target: Path
) -> tuple[int, str]:
    response = _download(session, url, timeout=(15, 60))
    content_length = response.headers.get("Content-Length")
    if content_length and int(content_length) > MAX_IMAGE_BYTES:
        raise PreparationError(f"Image exceeds {MAX_IMAGE_BYTES} byte limit.")
    if len(response.content) > MAX_IMAGE_BYTES:
        raise PreparationError(f"Image exceeds {MAX_IMAGE_BYTES} byte limit.")
    try:
        with Image.open(io.BytesIO(response.content)) as image:
            image.verify()
        with Image.open(io.BytesIO(response.content)) as image:
            image.load()
            if image.width < 16 or image.height < 16:
                raise ValueError("image dimensions are too small")
            image_format = image.format or ""
            suffix = {"PNG": ".png", "JPEG": ".jpg"}.get(image_format)
            if suffix is None:
                raise ValueError(f"unsupported image format {image_format!r}")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        target.unlink(missing_ok=True)
        raise PreparationError(f"Corrupt or unsupported image: {exc}") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    return len(response.content), suffix


def _download_selected(
    session: requests.Session,
    rows: list[dict[str, Any]],
    image_root: Path,
) -> tuple[list[dict[str, Any]], int, list[str]]:
    manifest_rows: list[dict[str, Any]] = []
    total_bytes = 0
    errors: list[str] = []
    hashes: set[str] = set()
    for row in rows:
        if row.get("local_path"):
            split = str(row["split"])
            target = image_root / "images" / split / f"medical-{str(row.get('content_hash', 'image'))[:24]}.bin"
            try:
                source_path = Path(row["local_path"])
                image_bytes = source_path.read_bytes()
                if len(image_bytes) > MAX_IMAGE_BYTES:
                    raise PreparationError("medical image exceeds maximum size")
                image_hash = row.get("content_hash") or hashlib.sha256(image_bytes).hexdigest()
                if image_hash in hashes:
                    raise PreparationError("exact duplicate image content")
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.verify()
                    image_format = image.format or ""
                suffix = {"PNG": ".png", "JPEG": ".jpg", "JPG": ".jpg"}.get(
                    image_format.upper()
                )
                if suffix is None:
                    raise PreparationError(
                        f"Unsupported medical image format {image_format!r}"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(image_bytes)
                final_path = target.with_suffix(suffix)
                target.replace(final_path)
                hashes.add(image_hash)
                total_bytes += len(image_bytes)
                manifest_rows.append(
                    {
                        "image_path": str(final_path.relative_to(image_root)).replace("\\", "/"),
                        "label": "non_cxr",
                        "split": split,
                        "group_id": row["group_id"],
                        "source": row["source"],
                        "source_title": row["source_title"],
                        "category": row["category"],
                        "license": row["license"],
                        "attribution": row["attribution"],
                        "license_url": row["license_url"],
                        "source_url": row["source_url"],
                        "source_landing_url": row["landing_url"],
                        "source_id": row["image_id"],
                    }
                )
            except (PreparationError, OSError, UnidentifiedImageError, ValueError) as exc:
                target.unlink(missing_ok=True)
                errors.append(f"medical:{row.get('image_id')}: {exc}")
            continue

        split = row["split"]
        if row.get("uid"):
            image_name = f"taix-{row['uid']}"
            image_download_url = row["image_url"]
            group_id = f"taixray-patient:{row['patient_id']}"
            category = "chest_xray"
            source = TAIX_DATASET
            license_name = row["license"]
            attribution = row["attribution"]
            source_title = TAIX_SOURCE_TITLE
            landing_url = f"https://huggingface.co/datasets/{TAIX_DATASET}"
            source_url = landing_url
        elif row.get("commons_file_title"):
            image_name = "commons-" + hashlib.sha256(
                row["commons_file_title"].encode("utf-8")
            ).hexdigest()
            image_download_url = row["original_url"]
            group_id = row["group_id"]
            category = row["category"]
            source = "Wikimedia Commons"
            license_name = row["license"]
            attribution = row["author"]
            source_title = row["title"]
            landing_url = row["landing_url"]
            source_url = landing_url
        else:
            image_name = f"openimages-{row['image_id']}"
            image_download_url = row["original_url"]
            group_id = f"openimages-uploader:{row['group_id']}"
            category = row["category"]
            source = f"Open Images V6 {row['subset']}"
            license_name = row["license"]
            attribution = row["author"]
            source_title = row["title"]
            landing_url = row["landing_url"]
        target = image_root / "images" / split / f"{image_name}.bin"
        try:
            size, suffix = _fetch_image(session, image_download_url, target)
            image_bytes = target.read_bytes()
            image_hash = hashlib.sha256(image_bytes).hexdigest()
            if image_hash in hashes:
                target.unlink(missing_ok=True)
                raise PreparationError("exact duplicate image content")
            hashes.add(image_hash)
            final_path = target.with_suffix(suffix)
            target.replace(final_path)
            total_bytes += size
        except (PreparationError, requests.RequestException, OSError) as exc:
            target.unlink(missing_ok=True)
            errors.append(f"{source}:{row.get('image_id', row.get('uid'))}: {exc}")
            continue
        manifest_rows.append(
            {
                "image_path": str(final_path.relative_to(image_root)).replace("\\", "/"),
                "label": "cxr" if category == "chest_xray" else "non_cxr",
                "split": split,
                "group_id": group_id,
                "source": source,
                "source_title": source_title,
                "category": category,
                "license": license_name,
                "attribution": attribution,
                "license_url": row.get("license_url", ""),
                "source_url": source_url,
                "source_landing_url": landing_url,
                "source_id": row.get(
                    "uid", row.get("commons_file_title", row.get("image_id"))
                ),
            }
        )
    return manifest_rows, total_bytes, errors


def _check_quota(rows: list[dict[str, Any]], expected: dict[str, int]) -> None:
    counts = {
        split: sum(row["split"] == split for row in rows)
        for split in SPLITS
    }
    categories = {
        split: {
            row["category"]
            for row in rows
            if row["split"] == split and row["category"] != "chest_xray"
        }
        for split in SPLITS
    }
    if counts != expected:
        raise PreparationError(f"Selected negative split counts {counts}; expected {expected}.")
    for split in SPLITS:
        missing = REQUIRED_NON_CXR_CATEGORIES - categories[split]
        if missing:
            raise PreparationError(
                f"Split {split} lacks required negative categories: {sorted(missing)}."
            )


def _is_http_url(value: str) -> bool:
    return str(value).strip().lower().startswith(("http://", "https://"))


def _validate_medical_local_image(path: Path) -> tuple[int, int, str]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format or ""
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PreparationError(f"Corrupt or unsupported medical image {path}: {exc}") from exc
    if width < 16 or height < 16:
        raise PreparationError(
            f"Medical image dimensions are too small: {path} ({width}x{height})."
        )
    suffix = {"PNG": ".png", "JPEG": ".jpg", "JPG": ".jpg"}.get(image_format.upper())
    if suffix is None:
        raise PreparationError(f"Unsupported medical image format {image_format!r}: {path}")
    return width, height, suffix


LICENSE_REVIEW_COLUMNS = {
    "image_id",
    "verified_license",
    "reviewed_landing_url",
    "reviewed_at",
    "reviewer",
}


def _write_license_review_template(
    rows: list[dict[str, Any]], template_path: Path
) -> None:
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template = pd.DataFrame(
        [
            {
                "image_id": row["image_id"],
                "split": row["split"],
                "category": row["category"],
                "original_url": row["original_url"],
                "landing_url": row["landing_url"],
                "metadata_license": row["license"],
                "author": row["author"],
                "title": row["title"],
                "verified_license": "",
                "reviewed_landing_url": "",
                "reviewed_at": "",
                "reviewer": "",
            }
            for row in sorted(rows, key=lambda item: item["image_id"])
        ]
    )
    template.to_csv(template_path, index=False, quoting=csv.QUOTE_MINIMAL)


def _validate_license_review(
    review_path: Path, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    if not review_path.is_file():
        raise PreparationError(
            "Open Images source pages have not been independently reviewed. "
            "Complete the generated openimages-license-review-template.csv, save it "
            "as openimages-license-review.csv, and rerun."
        )
    review = pd.read_csv(review_path, dtype=str, keep_default_na=False)
    missing_columns = LICENSE_REVIEW_COLUMNS - set(review.columns)
    if missing_columns:
        raise PreparationError(
            f"License review is missing columns: {', '.join(sorted(missing_columns))}."
        )
    if review["image_id"].duplicated().any():
        raise PreparationError("License review contains duplicate image_id values.")
    expected = {row["image_id"]: row for row in rows}
    reviewed = {row["image_id"]: row for row in review.to_dict("records")}
    if set(reviewed) != set(expected):
        raise PreparationError(
            "License review image IDs do not exactly match the selected sample "
            f"(missing={len(set(expected) - set(reviewed))}, "
            f"unexpected={len(set(reviewed) - set(expected))})."
        )
    for image_id, item in expected.items():
        approval = reviewed[image_id]
        if approval["verified_license"].strip() != OPEN_IMAGES_LICENSE:
            raise PreparationError(
                f"Image {image_id} lacks an explicit review for the required CC BY 2.0 license."
            )
        if approval["reviewed_landing_url"].strip() != item["landing_url"]:
            raise PreparationError(
                f"Image {image_id} landing URL was not reviewed exactly."
            )
        if not approval["reviewer"].strip():
            raise PreparationError(f"Image {image_id} has no reviewer attribution.")
        if pd.isna(pd.to_datetime(approval["reviewed_at"], errors="coerce", utc=True)):
            raise PreparationError(f"Image {image_id} has an invalid reviewed_at value.")
    return {
        "reviewed_image_count": len(reviewed),
        "review_file": str(review_path.resolve()),
        "meaning": (
            "Human-provided per-image review record; it does not independently prove "
            "the reviewer identity or license claim."
        ),
    }


def prepare_dataset(
    output_dir: Path,
    *,
    check_only: bool = False,
    license_review_path: Path | None = None,
    photo_source: str = "openimages",
    medical_manifest_path: Path | None = None,
    medical_staging_dir: Path | None = None,
) -> dict[str, Any]:
    if photo_source not in {"openimages", "commons"}:
        raise PreparationError(f"Unsupported photo source: {photo_source!r}.")
    if photo_source == "commons" and license_review_path is not None:
        raise PreparationError(
            "--license-review applies only to Open Images; Commons licensing is "
            "filtered from per-file API metadata."
        )
    output_dir = output_dir.resolve()
    medical_staging_root = (
        Path(medical_staging_dir).resolve()
        if medical_staging_dir is not None
        else (output_dir / DEFAULT_MEDICAL_STAGING_DIRNAME).resolve()
    )
    metadata_dir = output_dir / "metadata"
    image_root = output_dir
    session = requests.Session()
    session.headers["User-Agent"] = (
        "medical-imaging-modality-research/1.0 "
        "(https://www.mediawiki.org/wiki/API:Etiquette)"
    )

    per_category_by_split = NEGATIVE_TARGETS_BY_SPLIT
    negative_targets = {
        f"{split}:{category}": target
        for split, category_targets in per_category_by_split.items()
        for category, target in category_targets.items()
    }
    review_template_path: Path | None = None
    commons_scan: dict[str, Any] | None = None
    commons_quota_availability: dict[str, dict[str, int]] | None = None
    quota_error: str | None = None
    if photo_source == "openimages":
        photo_records, photo_availability = _openimages_metadata(
            session, metadata_dir
        )
        insufficiencies = {
            key: {"available": photo_availability.get(key, 0), "needed": needed}
            for key, needed in negative_targets.items()
            if photo_availability.get(key, 0) < needed
        }
        if insufficiencies:
            raise PreparationError(
                f"Insufficient explicitly licensed Open Images candidates: {insufficiencies}"
            )
        negative_rows = _select_openimages_by_split(
            photo_records, per_category_by_split
        )
        review_template_path = output_dir / "openimages-license-review-template.csv"
        _write_license_review_template(negative_rows, review_template_path)
        license_review: dict[str, Any] | None = None
    elif photo_source == "commons":
        commons_cache_dir = metadata_dir / "commons-api-cache"
        photo_records, photo_availability, commons_scan = _commons_metadata(
            session, commons_cache_dir
        )
        commons_quota_availability = _commons_quota_availability(
            photo_records, per_category_by_split
        )
        try:
            negative_rows = _select_commons_by_split(
                photo_records, per_category_by_split
            )
        except PreparationError as exc:
            negative_rows = []
            quota_error = str(exc)
        else:
            quota_error = None
        license_review = {
            "mode": "automatic_per_file_license_allowlist",
            "allowed_licenses": sorted(set(COMMONS_LICENSES.values())),
            "review_required": False,
            "evidence": (
                "Wikimedia Commons API per-file LicenseUrl and LicenseShortName "
                "metadata; not independent legal verification or a warranty of rights."
            ),
        }
    medical_rows: list[dict[str, Any]] = []
    medical_grouping_status = "not_provided"
    medical_expected_counts = {split: 0 for split in SPLITS}
    estimated_medical_bytes = 0
    if medical_manifest_path is not None:
        medical_rows = _load_medical_manifest(medical_manifest_path, medical_staging_root)
        medical_expected_counts = {
            split: sum(row["split"] == split for row in medical_rows)
            for split in SPLITS
        }
        if medical_rows and all(row.get("grouping_verified") for row in medical_rows):
            medical_grouping_status = "verified"
        elif medical_rows and any(row.get("grouping_verified") for row in medical_rows):
            medical_grouping_status = "mixed"
        elif medical_rows:
            medical_grouping_status = "unverified"
        estimated_medical_bytes = sum(
            int(row.get("size_bytes") or 300_000) for row in medical_rows
        )
        if any(
            MEDICAL_NEGATIVE_TARGETS_BY_SPLIT[split][category] > 0
            for split in SPLITS
            for category in MEDICAL_NON_CXR_CATEGORIES
        ):
            _check_medical_quota(medical_rows, MEDICAL_NEGATIVE_TARGETS_BY_SPLIT)

    expected_negative_counts = {
        "train": 1000,
        "validation": 1000,
        "calibration": 3000,
        "test": 3000,
    }
    for split in SPLITS:
        expected_negative_counts[split] += medical_expected_counts[split]
    quotas_met = quota_error is None if photo_source == "commons" else True
    if not quotas_met and not check_only:
        shortage_detail = commons_quota_availability
        raise PreparationError(
            f"Commons photo quotas are not satisfied: {quota_error}. "
            f"Eligible count shortages by split/category: {shortage_detail}. "
            f"Scan status: {commons_scan}."
        )
    if quotas_met:
        _check_quota(negative_rows, expected_negative_counts)
        cxr_rows, cxr_counts = _collect_taix_rows(session, TAIX_TARGETS)
    else:
        cxr_rows = []
        cxr_counts = {
            split: {"planned": target, "queried": False}
            for split, target in TAIX_TARGETS.items()
        }

    estimated_photo_bytes = sum(int(row.get("size_bytes") or 300_000) for row in negative_rows)
    estimated_cxr_bytes = sum(TAIX_TARGETS.values()) * 300_000
    estimated_bytes = estimated_photo_bytes + estimated_cxr_bytes + estimated_medical_bytes
    free_bytes = shutil.disk_usage(output_dir if output_dir.exists() else output_dir.parent).free
    if free_bytes < estimated_bytes * 1.5:
        raise PreparationError(
            f"Insufficient free disk: have {free_bytes} bytes; estimated minimum "
            f"{int(estimated_bytes * 1.5)} bytes."
        )
    if check_only:
        eligibility_report = None
        if photo_source == "commons" and commons_quota_availability is not None:
            eligibility_report = {
                split: {
                    category: {
                        "discovered_eligible": photo_availability.get(
                            f"{split}:{category}", 0
                        ),
                        "available_after_unique_selection": available,
                        "required": per_category_by_split[split][category],
                        "shortage": max(
                            0, per_category_by_split[split][category] - available
                        ),
                        "assessment": (
                            "quota_met"
                            if available >= per_category_by_split[split][category]
                            else (
                                "below_quota_after_complete_scan"
                                if commons_scan and commons_scan["complete"]
                                else "provisional_shortage_scan_incomplete"
                            )
                        ),
                    }
                    for category, available in commons_quota_availability[
                        split
                    ].items()
                }
                for split in SPLITS
            }
        return {
            "check_only": True,
            "quota_status": "met" if quotas_met else "not_met",
            "quota_error": quota_error,
            "commons_scan": commons_scan,
            "quota_availability": eligibility_report,
            "output_dir": str(output_dir),
            "photo_source": photo_source,
            "license_filtered_candidates": photo_availability,
            "negative_selection_counts": expected_negative_counts,
            "taix_cxr_counts": cxr_counts,
            "estimated_minimum_download_bytes": estimated_bytes,
            "estimated_photo_image_bytes": estimated_photo_bytes,
            "estimated_cxr_image_bytes": estimated_cxr_bytes,
            "photo_image_size_basis": (
                "Exact per-file image sizes returned by Wikimedia Commons API metadata."
                if photo_source == "commons"
                else "300,000-byte planning estimate per image."
            ),
            "free_disk_bytes": free_bytes,
            "license": {
                "taix_ray": TAIX_LICENSE,
                "negative_class": (
                    sorted(set(COMMONS_LICENSES.values()))
                    if photo_source == "commons"
                    else OPEN_IMAGES_LICENSE
                ),
            },
            "license_review_template": (
                str(review_template_path) if review_template_path is not None else None
            ),
            "negative_class_license_evidence": (
                "Per-file Commons LicenseUrl and LicenseShortName metadata; "
                "all selected records must match the allowlist and include attribution."
                if photo_source == "commons"
                else "Open Images license metadata only; per-image human review is still required."
            ),
            "images_downloaded": 0,
            "medical_manifest": str(medical_manifest_path) if medical_manifest_path is not None else None,
            "medical_manifest_rows": len(medical_rows),
            "medical_category_counts": _category_counts(medical_rows),
            "medical_grouping_status": medical_grouping_status,
            "medical_group_leakage_count": 0,
            "medical_duplicate_image_count": 0,
            "estimated_medical_image_bytes": estimated_medical_bytes,
        }

    if photo_source == "openimages":
        review_path = license_review_path or output_dir / "openimages-license-review.csv"
        license_review = _validate_license_review(review_path, negative_rows)
    rows = negative_rows + cxr_rows + medical_rows
    manifest_rows, downloaded_bytes, errors = _download_selected(
        session, rows, image_root
    )
    _verify_downloaded_counts(
        manifest_rows, expected_negative_counts, TAIX_TARGETS, errors=errors
    )
    _check_quota(
        [row for row in manifest_rows if row["label"] == "non_cxr"],
        expected_negative_counts,
    )
    medical_manifest_rows = [
        row for row in manifest_rows if row["category"] in MEDICAL_NON_CXR_CATEGORIES
    ]
    if medical_manifest_rows and any(
        MEDICAL_NEGATIVE_TARGETS_BY_SPLIT[split][category] > 0
        for split in SPLITS
        for category in MEDICAL_NON_CXR_CATEGORIES
    ):
        _check_medical_quota(medical_manifest_rows, MEDICAL_NEGATIVE_TARGETS_BY_SPLIT)
    validate_manifest(pd.DataFrame(manifest_rows), image_root)
    manifest_path = output_dir / "manifest.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, quoting=csv.QUOTE_MINIMAL)
    report = {
        "check_only": False,
        "output_dir": str(output_dir),
        "photo_source": photo_source,
        "manifest": str(manifest_path),
        "images_downloaded": len(manifest_rows),
        "image_bytes": downloaded_bytes,
        "image_megabytes": round(downloaded_bytes / 1_000_000, 2),
        "split_counts": {
            split: {
                label: sum(
                    row["split"] == split and row["label"] == label
                    for row in manifest_rows
                )
                for label in ("cxr", "non_cxr")
            }
            for split in sorted(SPLITS)
        },
        "negative_category_counts": _category_counts(manifest_rows),
        "medical_manifest": str(medical_manifest_path) if medical_manifest_path is not None else None,
        "medical_grouping_status": medical_grouping_status,
        "medical_images_downloaded": len(medical_manifest_rows),
        "skipped_or_corrupt_images": len(errors),
        "errors": errors[:20],
        "manifest_disk_bytes": manifest_path.stat().st_size,
        "all_output_disk_bytes": _directory_size(output_dir),
        "license": {
            "taix_ray": TAIX_LICENSE,
            "negative_class": (
                sorted(set(COMMONS_LICENSES.values()))
                if photo_source == "commons"
                else OPEN_IMAGES_LICENSE
            ),
        },
        "negative_class_license_review": license_review,
        "status": "research_dataset_only; classifier remains disconnected from application",
    }
    (output_dir / "preparation-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _select_openimages_by_split(
    records: dict[str, dict[str, Any]],
    targets: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for split in sorted(SPLITS):
        candidates = [r for r in records.values() if r["split"] == split]
        used: set[str] = set()
        for category in sorted(REQUIRED_NON_CXR_CATEGORIES):
            needed = targets[split][category]
            choices = [
                r
                for r in candidates
                if category in r["categories"]
                and r.get("license") == OPEN_IMAGES_LICENSE
                and r.get("group_id")
                and r.get("author")
                and r.get("landing_url")
                and r.get("original_url")
            ]
            choices.sort(
                key=lambda r: hashlib.sha256(
                    f"{SEED}:{split}:{category}:{r['image_id']}".encode("utf-8")
                ).digest()
            )
            added = 0
            for record in choices:
                if record["image_id"] in used:
                    continue
                selected.append({**record, "split": split, "category": category})
                used.add(record["image_id"])
                added += 1
                if added == needed:
                    break
            if added < needed:
                raise PreparationError(
                    f"Open Images split {split} has {added}/{needed} unique licensed "
                    f"{category} candidates after cross-category de-duplication."
                )
    return selected


def _verify_downloaded_counts(
    rows: list[dict[str, Any]],
    negative_targets: dict[str, int],
    cxr_targets: dict[str, int],
    *,
    errors: list[str],
) -> None:
    deficits = {}
    for split in SPLITS:
        for label, expected in (
            ("non_cxr", negative_targets[split]),
            ("cxr", cxr_targets[split]),
        ):
            actual = sum(
                row["split"] == split and row["label"] == label for row in rows
            )
            if actual != expected:
                deficits[f"{split}:{label}"] = {"actual": actual, "expected": expected}
    if deficits:
        raise PreparationError(
            f"Download failures left incomplete splits; no manifest was published: "
            f"{deficits}. Failed/corrupt downloads: {len(errors)}; samples: {errors[:10]}"
        )


def _load_medical_manifest(manifest_path: Path, staging_dir: Path) -> list[dict[str, Any]]:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise PreparationError(f"Medical manifest does not exist: {manifest_path}")

    staging_dir = Path(staging_dir).resolve()
    try:
        frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError) as exc:
        raise PreparationError(f"Could not read medical manifest: {exc}") from exc

    missing = MEDICAL_MANIFEST_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise PreparationError(
            f"Medical manifest is missing columns: {', '.join(sorted(missing))}"
        )
    if frame.empty:
        raise PreparationError("Medical manifest contains no rows.")

    rows: list[dict[str, Any]] = []
    group_splits: dict[str, str] = {}
    content_hashes: set[str] = set()
    for row_index, raw in frame.iterrows():
        image_id = str(raw["image_id"]).strip()
        if not image_id:
            raise PreparationError(f"Medical manifest row {row_index} has an empty image_id.")

        local_value = str(raw["local_path"]).strip()
        if not local_value:
            raise PreparationError(f"Medical manifest row {row_index} has an empty local_path.")
        local_path = Path(local_value)
        if not local_path.is_absolute():
            local_path = staging_dir / local_path
        local_path = local_path.resolve()
        if staging_dir not in local_path.parents:
            raise PreparationError(
                f"Medical manifest row {row_index} local_path escapes the approved "
                f"staging directory: {local_path}"
            )
        if not local_path.is_file():
            raise PreparationError(f"Medical image does not exist: {local_path}")

        width, height, _ = _validate_medical_local_image(local_path)
        content_hash = hashlib.sha256(local_path.read_bytes()).hexdigest()
        if content_hash in content_hashes:
            raise PreparationError(
                f"Duplicate medical image content in manifest: {local_path}"
            )
        content_hashes.add(content_hash)

        group_id = str(raw["group_id"]).strip()
        if not group_id:
            raise PreparationError(f"Medical manifest row {row_index} has an empty group_id.")

        category = str(raw["category"]).strip()
        if category not in MEDICAL_NON_CXR_CATEGORIES:
            raise PreparationError(
                f"Medical manifest row {row_index} has invalid category {category!r}."
            )

        source = str(raw["source"]).strip()
        if not source:
            raise PreparationError(f"Medical manifest row {row_index} has an empty source.")

        license_name = str(raw["license"]).strip()
        if not license_name:
            raise PreparationError(f"Medical manifest row {row_index} has an empty license.")

        attribution = str(raw["attribution"]).strip()
        if not attribution:
            raise PreparationError(
                f"Medical manifest row {row_index} has an empty attribution."
            )

        license_url = str(raw["license_url"]).strip()
        source_url = str(raw["source_url"]).strip()
        landing_url = str(raw["landing_url"]).strip()
        for field_name, value in (
            ("license_url", license_url),
            ("source_url", source_url),
            ("landing_url", landing_url),
        ):
            if not value or not _is_http_url(value):
                raise PreparationError(
                    f"Medical manifest row {row_index} has an invalid {field_name}."
                )

        source_title = str(raw.get("source_title", "")).strip()
        if not source_title:
            source_title = source

        split_value = str(raw.get("split", "")).strip()
        if split_value:
            if split_value not in SPLITS:
                raise PreparationError(
                    f"Medical manifest row {row_index} has invalid split {split_value!r}."
                )
            split = split_value
        else:
            split = _medical_split(group_id)

        previous_split = group_splits.get(group_id)
        if previous_split is not None and previous_split != split:
            raise PreparationError(
                f"Group leakage: group_id {group_id!r} occurs in {previous_split!r} "
                f"and {split!r}."
            )
        group_splits[group_id] = split

        grouping_value = str(
            raw.get("grouping_verified", raw.get("patient_grouping_verified", ""))
        ).strip().lower()
        grouping_verified = grouping_value in {"true", "1", "yes", "verified"}

        rows.append(
            {
                "image_id": image_id,
                "local_path": str(local_path),
                "group_id": group_id,
                "split": split,
                "category": category,
                "source": source,
                "source_title": source_title,
                "license": license_name,
                "attribution": attribution,
                "license_url": license_url,
                "source_url": source_url,
                "landing_url": landing_url,
                "size_bytes": local_path.stat().st_size,
                "content_hash": content_hash,
                "grouping_verified": grouping_verified,
                "_width": width,
                "_height": height,
            }
        )

    return rows


load_medical_manifest = _load_medical_manifest
validate_medical_manifest = _load_medical_manifest
_validate_medical_manifest = _load_medical_manifest


def _check_medical_quota(rows: list[dict[str, Any]], targets: dict[str, dict[str, int]]) -> None:
    for split in SPLITS:
        for category, needed in targets[split].items():
            if needed <= 0:
                continue
            actual = sum(
                row["split"] == split and row["category"] == category for row in rows
            )
            if actual < needed:
                raise PreparationError(
                    f"Medical split {split} has {actual}/{needed} {category} images."
                )


def _category_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    categories = sorted(REQUIRED_NON_CXR_CATEGORIES | MEDICAL_NON_CXR_CATEGORIES)
    return {
        split: {
            category: sum(
                row["split"] == split and row["category"] == category for row in rows
            )
            for category in categories
        }
        for split in sorted(SPLITS)
    }


def _directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("R:/medical-imaging-modality-data"),
    )
    parser.add_argument(
        "--photo-source",
        choices=("openimages", "commons"),
        default="openimages",
        help=(
            "Negative-image source. Open Images requires its review CSV; Commons "
            "uses a strict per-file CC BY/CC0 metadata allowlist."
        ),
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--license-review",
        type=Path,
        help="Completed per-image review CSV; required before any image download.",
    )
    parser.add_argument(
        "--medical-manifest",
        type=Path,
        help="CSV manifest of locally staged medical-modality negatives.",
    )
    parser.add_argument(
        "--medical-staging-dir",
        type=Path,
        help=(
            "Approved staging directory containing medical manifest images. "
            "Defaults to <output-dir>/medical-staging."
        ),
    )
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                prepare_dataset(
                    args.output_dir,
                    check_only=args.check_only,
                    license_review_path=args.license_review,
                    photo_source=args.photo_source,
                    medical_manifest_path=args.medical_manifest,
                    medical_staging_dir=args.medical_staging_dir,
                ),
                indent=2,
            )
        )
    except PreparationError as exc:
        parser.exit(2, f"Data preparation blocked: {exc}\n")


if __name__ == "__main__":
    main()
