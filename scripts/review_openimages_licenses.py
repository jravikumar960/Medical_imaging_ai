#!/usr/bin/env python
"""Batch workflow for the manual Open Images CC BY 2.0 license review.

This helper never approves records. It only splits the generated review template
into smaller human-review batches, merges completed batches, and validates the
result with the existing preparation pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_modality_data import (
    LICENSE_REVIEW_COLUMNS,
    _validate_license_review,
)

TEMPLATE_COLUMNS = [
    "image_id",
    "split",
    "category",
    "original_url",
    "landing_url",
    "metadata_license",
    "author",
    "title",
    "verified_license",
    "reviewed_landing_url",
    "reviewed_at",
    "reviewer",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CSV does not exist: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def split_template(template_path: Path, output_dir: Path, batch_size: int) -> None:
    frame = _read_csv(template_path)
    missing = set(TEMPLATE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Template is missing columns: {', '.join(sorted(missing))}")
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(frame)
    for start in range(0, total, batch_size):
        batch_number = start // batch_size + 1
        target = output_dir / f"batch_{batch_number:04d}.csv"
        if target.exists():
            continue
        frame.iloc[start : start + batch_size].to_csv(
            target, index=False, quoting=1
        )
    print(f"Created up to {((total - 1) // batch_size) + 1} batches in {output_dir}")


def merge_batches(batch_dir: Path, template_path: Path, output_path: Path) -> None:
    batch_files = sorted(batch_dir.glob("batch_*.csv"))
    if not batch_files:
        raise FileNotFoundError(f"No batch_*.csv files found in {batch_dir}")
    frames = [_read_csv(path) for path in batch_files]
    merged = pd.concat(frames, ignore_index=True)
    template = _read_csv(template_path)
    if set(merged.columns) != set(TEMPLATE_COLUMNS):
        raise ValueError("Merged batch columns do not match the template schema.")
    if len(merged) != len(template):
        raise ValueError(
            f"Merged row count {len(merged)} does not match template row count {len(template)}."
        )
    if merged["image_id"].duplicated().any():
        raise ValueError("Merged batches contain duplicate image_id values.")
    if set(merged["image_id"]) != set(template["image_id"]):
        raise ValueError("Merged batch image IDs do not exactly match the template.")
    review_columns = ["verified_license", "reviewed_landing_url", "reviewed_at", "reviewer"]
    template = template.copy()
    lookup = {row["image_id"]: row for row in merged.to_dict("records")}
    for column in review_columns:
        template[column] = template["image_id"].map(
            lambda image_id: lookup[image_id][column]
        )
    template.to_csv(output_path, index=False, quoting=1)
    _validate_license_review(output_path, template.to_dict("records"))
    print(f"Wrote validated review file: {output_path}")


def status(template_path: Path, review_path: Path | None) -> None:
    template = _read_csv(template_path)
    print(f"Template rows: {len(template)}")
    if review_path is None or not review_path.is_file():
        print("Review file: not present")
        print("Completed rows: 0")
        return
    review = _read_csv(review_path)
    completed = 0
    for row in review.to_dict("records"):
        if (
            row["verified_license"].strip()
            and row["reviewed_landing_url"].strip()
            and row["reviewer"].strip()
            and row["reviewed_at"].strip()
        ):
            completed += 1
    print(f"Review file: {review_path}")
    print(f"Completed rows: {completed}/{len(review)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split", help="Split the template into review batches.")
    split_parser.add_argument("--template", type=Path, required=True)
    split_parser.add_argument("--output-dir", type=Path, required=True)
    split_parser.add_argument("--batch-size", type=int, default=100)

    merge_parser = subparsers.add_parser("merge", help="Merge completed batches and validate them.")
    merge_parser.add_argument("--batch-dir", type=Path, required=True)
    merge_parser.add_argument("--template", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)

    status_parser = subparsers.add_parser("status", help="Report review completion.")
    status_parser.add_argument("--template", type=Path, required=True)
    status_parser.add_argument("--review", type=Path)

    args = parser.parse_args()
    if args.command == "split":
        split_template(args.template, args.output_dir, args.batch_size)
    elif args.command == "merge":
        merge_batches(args.batch_dir, args.template, args.output)
    elif args.command == "status":
        status(args.template, args.review)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
