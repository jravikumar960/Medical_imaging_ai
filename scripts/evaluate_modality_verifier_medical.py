#!/usr/bin/env python
"""Evaluate the production modality verifier on staged CT and MRI slices."""
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modality_verifier import load_modality_verifier


def canonical_volume(path: Path):
    image = nib.as_closest_canonical(nib.load(str(path)))
    array = np.asarray(image.dataobj, dtype=np.float32)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError(f"Invalid 3D volume: {path}")
    return image, array


def select_ct_slices(volume_shape, count=5):
    depth = volume_shape[2]
    if depth <= 0:
        raise ValueError("CT volume has no slices.")
    if count >= depth:
        return list(range(depth))
    if count <= 1:
        return [depth // 2]
    return sorted({round(i * (depth - 1) / (count - 1)) for i in range(count)})


def pil_from_array(array):
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Expected a finite 2D array.")
    return Image.fromarray(array.astype("float32"), mode="F")


def evaluate_image(verifier, image, metadata):
    result = verifier.verify(image)
    return {
        **metadata,
        "raw_score": result.score,
        "threshold": result.threshold,
        "accepted_as_cxr": bool(result.is_cxr),
        }
def group_summary(group_id, rows):
    return {
        "group_id": group_id,
        "image_count": len(rows),
        "accepted_count": sum(bool(row["accepted_as_cxr"]) for row in rows),
        "any_accepted": any(bool(row["accepted_as_cxr"]) for row in rows),
        "all_accepted": all(bool(row["accepted_as_cxr"]) for row in rows) if rows else False,
    }


def evaluate_ct(verifier, ct_path, slice_count=5):
    image, volume = canonical_volume(ct_path)
    spacing = tuple(float(v) for v in np.linalg.norm(image.affine[:3, :3], axis=0))
    group_id = "ct-study:kits_00108"
    rows = []
    for z in select_ct_slices(volume.shape, slice_count):
        array = volume[:, :, z]
        metadata = {
            "modality": "ct",
            "source_case": "C4KC-KiTS KiTS-00108",
            "source_license": "TCIA data-usage terms (collection-specific CC BY 3.0 or CC BY 4.0)",
            "group_id": group_id,
            "sequence": "ct_axial_slice",
            "slice_index": int(z),
            "shape": [int(v) for v in array.shape],
            "spacing_mm": spacing,
            "orientation": "RAS canonical",
            "intensity_min": float(array.min()),
            "intensity_max": float(array.max()),
        }
        rows.append(evaluate_image(verifier, pil_from_array(array), metadata))
    return rows, group_summary(group_id, rows)


def evaluate_mri_case(verifier, case_dir):
    case = case_dir.name
    sequence_files = {
        "t1c": case_dir / f"{case}-t1c.nii.gz",
        "t1": case_dir / f"{case}-t1n.nii.gz",
        "t2": case_dir / f"{case}-t2w.nii.gz",
        "flair": case_dir / f"{case}-t2f.nii.gz",
    }
    volumes = {}
    affines = {}
    for sequence, path in sequence_files.items():
        if not path.is_file():
            return [], None, f"missing sequence file: {path}"
        image, array = canonical_volume(path)
        volumes[sequence] = (image, array)
        affines[sequence] = image.affine
    reference_affine = affines["flair"]
    for sequence, (image, _array) in volumes.items():
        if not np.allclose(image.affine, reference_affine, atol=1e-4, rtol=1e-5):
            return [], None, f"sequence geometry mismatch: {sequence}"
    if len({volumes[s][1].shape for s in volumes}) != 1:
        return [], None, "sequence shapes differ"

    group_id = f"mri-study:{case}"
    rows = []
    for sequence in ("t1c", "t1", "t2", "flair"):
        image, volume = volumes[sequence]
        z = volume.shape[2] // 2
        array = volume[:, :, z]
        metadata = {
            "modality": "mri",
            "source_case": case,
            "source_license": "CC BY-NC-SA 4.0 (BraTS-2024-Complete, Spirit-26)",
            "group_id": group_id,
            "sequence": sequence,
            "source_sequence_file": sequence_files[sequence].name,
            "slice_index": int(z),
            "shape": [int(v) for v in array.shape],
            "volume_shape": [int(v) for v in volume.shape],
            "spacing_mm": tuple(float(v) for v in np.linalg.norm(image.affine[:3, :3], axis=0)),
            "orientation": "RAS canonical",
            "intensity_min": float(array.min()),
            "intensity_max": float(array.max()),
        }
        rows.append(evaluate_image(verifier, pil_from_array(array), metadata))
    return rows, group_summary(group_id, rows), None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct-path", type=Path, default=Path("outputs/ct_inference/kits_0108/kits_0108.nii.gz"))
    parser.add_argument("--mri-root", type=Path, default=Path("outputs/mri_inference"))
    parser.add_argument("--output", type=Path, default=Path("outputs/modality_verifier_medical_evaluation.json"))
    parser.add_argument("--ct-slices", type=int, default=5)
    args = parser.parse_args()

    verifier = load_modality_verifier()
    if verifier is None:
        raise SystemExit("No production modality verifier checkpoint is available.")

    ct_rows, ct_group = evaluate_ct(verifier, args.ct_path, args.ct_slices)

    mri_rows = []
    mri_groups = []
    mri_skipped = []
    for case_dir in sorted(args.mri_root.iterdir()):
        if not case_dir.is_dir() or not case_dir.name.startswith("BraTS-"):
            continue
        rows, group, reason = evaluate_mri_case(verifier, case_dir)
        if reason:
            mri_skipped.append({"case": case_dir.name, "reason": reason})
            continue
        mri_rows.extend(rows)
        mri_groups.append(group)

    all_rows = ct_rows + mri_rows
    all_groups = [ct_group] + mri_groups
    image_accepted = sum(bool(row["accepted_as_cxr"]) for row in all_rows)
    group_accepted = sum(bool(group["any_accepted"]) for group in all_groups)

    report = {
        "verifier": {
            "checkpoint": "models/modality_classifier.pt",
            "threshold": verifier.threshold,
            "score_type": "softmax probability for class cxr",
        },
        "ct": {
            "source_case": "C4KC-KiTS KiTS-00108",
            "source_license": "TCIA data-usage terms (collection-specific CC BY 3.0 or CC BY 4.0)",
            "group": ct_group,
            "images": ct_rows,
        },
        "mri": {
            "source_dataset": "Spirit-26/BraTS-2024-Complete",
            "source_license": "CC BY-NC-SA 4.0",
            "groups": mri_groups,
            "images": mri_rows,
            "skipped": mri_skipped,
        },
        "summary": {
            "image_count": len(all_rows),
            "accepted_image_count": image_accepted,
            "image_false_acceptance_rate": image_accepted / len(all_rows) if all_rows else None,
            "group_count": len(all_groups),
            "accepted_group_count": group_accepted,
            "group_false_acceptance_rate": group_accepted / len(all_groups) if all_groups else None,
            "note": "Slices and sequences from the same study are correlated; group-level results are reported separately from image-level results.",
        },
        "limitations": [
            "This evaluates a natural-image-versus-CXR verifier on medical images.",
            "It does not establish rejection performance for all CT or MRI studies.",
            "Correlated slices and sequences are not independent patients or studies.",
            "No ground truth is required or used for modality acceptance.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Images: {len(all_rows)}, accepted: {image_accepted}")
    print(f"Groups: {len(all_groups)}, groups with any acceptance: {group_accepted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
