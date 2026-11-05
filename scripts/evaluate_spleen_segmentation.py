"""Offline evaluation CLI for the pretrained MONAI spleen segmentation model.

This script is separate from the production FastAPI inference API. It does not
validate CT provenance, source DICOM metadata, or signed orientation reviews,
and its results are training-set evaluation metrics only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import spleen_evaluation


def _parse_roi(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("ROI size must be three positive integers, e.g. 96,96,96")
    return tuple(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Offline evaluation of the pretrained MONAI spleen model on Task09_Spleen.")
    parser.add_argument("--dataset-root", required=True, help="Task09_Spleen dataset root containing imagesTr and labelsTr.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON/CSV reports and optional predictions.")
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "bundles" / "spleen_ct_segmentation" / "models" / "model.pt"),
        help="Local pretrained checkpoint path.",
    )
    parser.add_argument("--case-ids", help="Comma-separated case IDs to evaluate.")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N matched cases.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda:0.")
    parser.add_argument("--roi-size", type=_parse_roi, default=(96, 96, 96), help="Sliding-window ROI size.")
    parser.add_argument("--sw-batch-size", type=int, default=4, help="Sliding-window batch size.")
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding-window overlap.")
    parser.add_argument("--save-predictions", action="store_true", help="Save predicted masks as NIfTI files.")
    args = parser.parse_args(argv)

    case_ids = [value.strip() for value in args.case_ids.split(",")] if args.case_ids else None
    report = spleen_evaluation.run_evaluation(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        case_ids=case_ids,
        limit=args.limit,
        device=args.device,
        roi_size=args.roi_size,
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap,
        save_predictions=args.save_predictions,
    )

    print(f"Evaluated {report.successful_cases}/{report.evaluated_cases} cases successfully.")
    print(f"Unmatched images: {len(report.unmatched_images)}; extra labels: {len(report.extra_labels)}; mismatches: {len(report.mismatched_pairs)}.")
    print(f"Mean Dice: {report.aggregate.get('mean_dice')}")
    print(f"Mean precision: {report.aggregate.get('mean_precision')}")
    print(f"Mean recall: {report.aggregate.get('mean_recall')}")
    print(f"Reports written to {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
