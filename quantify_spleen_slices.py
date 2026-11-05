"""Convenience script: quantify per-slice metrics for spleen_10/19/56 and render worst slices.

Usage (research only):
    python quantify_spleen_slices.py --dataset-root <Task09_Spleen root> \
        --checkpoint <model.pth> --output-dir <out>

This does not overwrite any existing evaluation reports; it writes to a fresh
subdirectory (default: ``<output-dir>/spleen_slice_research``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spleen_evaluation import run_evaluation
from spleen_research_viewer import render_worst_slices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out = Path(args.output_dir) / "spleen_slice_research"
    out.mkdir(parents=True, exist_ok=True)

    report = run_evaluation(
        dataset_root=args.dataset_root,
        output_dir=out,
        checkpoint_path=args.checkpoint,
        case_ids=["spleen_10", "spleen_19", "spleen_56"],
        device=args.device,
        save_predictions=True,
        save_slice_metrics=True,
    )

    for case in report.cases:
        if case.get("status") != "success" or not case.get("prediction_path"):
            print("SKIP", case.get("case_id"), case.get("status"), case.get("error"))
            continue
        slice_json = Path(case["slice_metrics_path"]) if case.get("slice_metrics_path") else None
        image_path = case["image_path"]
        pred_path = case["prediction_path"]
        ref_path = case["label_path"]
        render_worst_slices(
            image_path=image_path,
            prediction_path=pred_path,
            reference_path=ref_path,
            slice_metrics_json=str(slice_json) if slice_json and slice_json.exists() else None,
            k=5,
            output_path=str(out / f"{case['case_id']}_worst_slices.png"),
            title=f"{case['case_id']} worst slices (research only)",
        )
        print("OK", case["case_id"], "worst slices:", [w["slice_index"] for w in (case.get("worst_slices") or [])])


if __name__ == "__main__":
    main()