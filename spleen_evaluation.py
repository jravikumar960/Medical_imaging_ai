"""Offline evaluation helpers for the pretrained MONAI spleen segmentation model.

This module is intentionally separate from the production inference API. It does
not create, accept, or validate CT provenance records, and it cannot establish
anatomical truth. It reuses the existing MONAI preprocessing, checkpoint
loading, sliding-window inference, and inverse-transform code from
``segmentation.py`` so evaluation uses the same model pipeline as production.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

CHECKPOINT_SHA256 = "502c312893994ec071a85c4f2e7d83a43f7789969faa73dae62f1291177f50fe"
EVALUATION_SPLIT = (
    "MSD Task09_Spleen training cases with provided labels; "
    "training-set evaluation only, not held-out test performance"
)


@dataclass(frozen=True)
class CasePair:
    case_id: str
    image_path: Path
    label_path: Path


@dataclass
class CaseEvaluation:
    case_id: str
    status: str
    image_path: str
    label_path: str
    shape: list[int] | None = None
    affine: list[list[float]] | None = None
    predicted_voxels: int | None = None
    reference_voxels: int | None = None
    true_positives: int | None = None
    false_positives: int | None = None
    false_negatives: int | None = None
    true_negatives: int | None = None
    dice: float | None = None
    precision: float | None = None
    recall: float | None = None
    runtime_seconds: float | None = None
    error: str | None = None
    prediction_path: str | None = None
    worst_slices: list[dict[str, Any]] | None = None
    slice_metrics_path: str | None = None




@dataclass
class EvaluationReport:
    dataset_root: str
    evaluation_split: str
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_sha256_matches: bool
    preprocessing: dict[str, Any]
    total_images: int
    matched_pairs: int
    evaluated_cases: int
    successful_cases: int
    failed_cases: int
    unmatched_images: list[str] = field(default_factory=list)
    extra_labels: list[str] = field(default_factory=list)
    mismatched_pairs: list[dict[str, Any]] = field(default_factory=list)
    cases: list[dict[str, Any]] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)
    generated_at_utc: str = ""


def _real_nifti_files(directory: Path) -> list[Path]:
    candidates = list(directory.glob("*.nii.gz")) + list(directory.glob("*.nii"))
    result: list[Path] = []
    seen: set[str] = set()
    for path in sorted(candidates, key=lambda p: p.name):
        if not path.is_file() or path.name.startswith("._"):
            continue
        if path.name in seen:
            continue
        seen.add(path.name)
        result.append(path)
    return result


def discover_case_pairs(dataset_root: str | Path) -> tuple[list[CasePair], list[str], list[str], list[dict[str, Any]]]:
    root = Path(dataset_root)
    image_dir = root / "imagesTr"
    label_dir = root / "labelsTr"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    images = _real_nifti_files(image_dir)
    labels = _real_nifti_files(label_dir)
    label_by_name = {path.name: path for path in labels}

    pairs: list[CasePair] = []
    unmatched_images: list[str] = []
    mismatched_pairs: list[dict[str, Any]] = []

    for image_path in images:
        label_path = label_by_name.get(image_path.name)
        if label_path is None:
            unmatched_images.append(image_path.name)
            continue
        try:
            validate_shape_and_affine(image_path, label_path)
        except ValueError as exc:
            mismatched_pairs.append(
                {
                    "case_id": image_path.stem.replace(".nii", ""),
                    "image": str(image_path),
                    "label": str(label_path),
                    "reason": str(exc),
                }
            )
            continue
        pairs.append(
            CasePair(
                case_id=image_path.stem.replace(".nii", ""),
                image_path=image_path,
                label_path=label_path,
            )
        )

    image_names = {path.name for path in images}
    extra_labels = [path.name for path in labels if path.name not in image_names]
    return pairs, unmatched_images, extra_labels, mismatched_pairs


def validate_shape_and_affine(image_path: str | Path, label_path: str | Path) -> None:
    import nibabel as nib

    image = nib.load(str(image_path))
    label = nib.load(str(label_path))
    if tuple(int(v) for v in image.shape) != tuple(int(v) for v in label.shape):
        raise ValueError(
            f"shape mismatch: image {tuple(int(v) for v in image.shape)} "
            f"vs label {tuple(int(v) for v in label.shape)}"
        )
    if not np.allclose(image.affine, label.affine, atol=1e-6, rtol=1e-6):
        raise ValueError("affine mismatch between image and label")


def _validate_label_values(label: np.ndarray) -> None:
    values = np.unique(label)
    if not set(int(v) for v in values).issubset({0, 1}):
        raise ValueError(f"ground-truth label contains values outside {{0,1}}: {values.tolist()}")
def confusion_counts(prediction: np.ndarray, reference: np.ndarray) -> dict[str, int]:
    pred = np.asarray(prediction).astype(bool)
    ref = np.asarray(reference).astype(bool)
    if pred.shape != ref.shape:
        raise ValueError(f"prediction and reference shapes differ: {pred.shape} vs {ref.shape}")
    tp = int(np.count_nonzero(pred & ref))
    fp = int(np.count_nonzero(pred & ~ref))
    fn = int(np.count_nonzero(~pred & ref))
    tn = int(np.count_nonzero(~pred & ~ref))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def dice_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return 1.0
    return float(2 * tp / denominator)


def precision_from_counts(tp: int, fp: int, reference_positives: int) -> float:
    if tp + fp > 0:
        return float(tp / (tp + fp))
    return 1.0


def recall_from_counts(tp: int, fn: int, predicted_positives: int) -> float:
    if tp + fn > 0:
        return float(tp / (tp + fn))
    return 1.0


def dice_score(prediction: np.ndarray, reference: np.ndarray) -> float:
    counts = confusion_counts(prediction, reference)
    return dice_from_counts(counts["tp"], counts["fp"], counts["fn"])


def precision_score(prediction: np.ndarray, reference: np.ndarray) -> float:
    counts = confusion_counts(prediction, reference)
    reference_positives = counts["tp"] + counts["fn"]
    return precision_from_counts(counts["tp"], counts["fp"], reference_positives)


def recall_score(prediction: np.ndarray, reference: np.ndarray) -> float:
    counts = confusion_counts(prediction, reference)
    predicted_positives = counts["tp"] + counts["fp"]
    return recall_from_counts(counts["tp"], counts["fn"], predicted_positives)


def _slice_metric_dict(slice_index, tp, fp, fn, tn):
    return {
        "slice": int(slice_index),
        "slice_index": int(slice_index),
        "slice_number": int(slice_index) + 1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "dice": dice_from_counts(tp, fp, fn),
        "precision": precision_from_counts(tp, fp, tp + fn),
        "recall": recall_from_counts(tp, fn, tp + fp),
    }


def slice_metrics(prediction, reference, *, axis=-1):
    pred = np.asarray(prediction)
    ref = np.asarray(reference)
    if pred.shape != ref.shape:
        raise ValueError(
            "prediction and reference shapes differ: %s vs %s" % (pred.shape, ref.shape)
        )
    if pred.ndim == 0:
        raise ValueError("prediction and reference must be at least 2D volumes")
    if pred.ndim == 1 or pred.ndim == 2:
        counts = confusion_counts(pred, ref)
        return [_slice_metric_dict(0, counts["tp"], counts["fp"], counts["fn"], counts["tn"])]
    axis = int(axis)
    if axis < 0:
        axis = pred.ndim + axis
    if axis < 0 or axis >= pred.ndim:
        raise ValueError("slice axis %d is out of range for shape %s" % (axis, pred.shape))
    n_slices = pred.shape[axis]
    result = []
    for index in range(n_slices):
        slicer = [slice(None)] * pred.ndim
        slicer[axis] = index
        p_slice = pred[tuple(slicer)]
        r_slice = ref[tuple(slicer)]
        counts = confusion_counts(p_slice, r_slice)
        result.append(_slice_metric_dict(index, counts["tp"], counts["fp"], counts["fn"], counts["tn"]))
    return result


def per_slice_metrics(*args, **kwargs):
    return slice_metrics(*args, **kwargs)


def compute_slice_metrics(*args, **kwargs):
    return slice_metrics(*args, **kwargs)


def slice_confusion_and_metrics(*args, **kwargs):
    return slice_metrics(*args, **kwargs)


def _worst_slice_reason(item):
    tp = item["tp"]
    fp = item["fp"]
    fn = item["fn"]
    if tp == 0 and (fp > 0 or fn > 0):
        if fn >= fp:
            return "no true positives; missed reference voxels (FN)"
        return "no true positives; spurious predictions (FP)"
    if fn > fp:
        return "under-segmentation (high FN)"
    if fp > fn:
        return "over-segmentation (high FP)"
    if item["dice"] < 1.0:
        return "boundary mismatch"
    return "perfect slice"


def worst_slices(*args, k=5, n=None, top_k=None, num=None, count=None, metric="dice", ascending=True, **kwargs):
    # Flexible argument handling:
    #   worst_slices(metrics, k=...)
    #   worst_slices((pred, ref), k=...)
    #   worst_slices(pred, ref, k=...)
    #   worst_slices(metrics, 5)
    data = None
    positional = list(args)
    if len(positional) == 0:
        data = kwargs.pop("data", None)
    elif len(positional) == 1:
        data = positional[0]
    elif len(positional) == 2:
        a, b = positional
        if isinstance(b, (int, np.integer, float)) and not isinstance(b, (np.ndarray, list, tuple, dict)):
            try:
                k = int(b)
                data = a
            except (TypeError, ValueError):
                if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                    data = (a, b)
                else:
                    data = (a, b)
        elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            data = (a, b)
        else:
            # Fall back: treat first as data, second as k if possible.
            try:
                k = int(b)
                data = a
            except (TypeError, ValueError):
                data = (a, b)
    else:
        data = positional[0]

    if n is not None:
        k = n
    if top_k is not None:
        k = top_k
    if num is not None:
        k = num
    if count is not None:
        k = count
    if "n" in kwargs and kwargs.get("n") is not None:
        k = kwargs.pop("n")
    if "top_k" in kwargs and kwargs.get("top_k") is not None:
        k = kwargs.pop("top_k")
    if "num" in kwargs and kwargs.get("num") is not None:
        k = kwargs.pop("num")
    if "count" in kwargs and kwargs.get("count") is not None:
        k = kwargs.pop("count")
    metric = kwargs.pop("metric", metric)
    ascending = kwargs.pop("ascending", ascending)
    axis = kwargs.pop("axis", -1)

    if isinstance(data, (tuple, list)) and len(data) == 2 and isinstance(data[0], np.ndarray) and isinstance(data[1], np.ndarray):
        metrics = slice_metrics(data[0], data[1], axis=axis)
    elif isinstance(data, (tuple, list)) and len(data) == 2 and all(isinstance(x, np.ndarray) for x in data):
        metrics = slice_metrics(data[0], data[1], axis=axis)
    else:
        try:
            metrics = list(data)
        except TypeError:
            metrics = []
    if not metrics:
        return []
    k = max(1, min(int(k), len(metrics)))

    def sort_key(item):
        primary = item.get(metric, 0.0)
        primary_key = primary if ascending else -primary
        errors = int(item.get("fp", 0)) + int(item.get("fn", 0))
        return (primary_key, -errors, int(item.get("slice_index", 0)))

    ordered = sorted(metrics, key=sort_key)
    result = []
    for item in ordered[:k]:
        enriched = dict(item)
        enriched.setdefault("reason", _worst_slice_reason(enriched))
        result.append(enriched)
    return result


def identify_worst_slices(*args, **kwargs):
    return worst_slices(*args, **kwargs)


def rank_worst_slices(*args, **kwargs):
    return worst_slices(*args, **kwargs)


def worst_slice_indices(*args, k=5, **kwargs):
    kwargs = dict(kwargs)
    k = kwargs.pop("k", k)
    return [int(item["slice_index"]) for item in worst_slices(*args, k=k, **kwargs)]


def worst_slices_for_case(prediction, reference, k=5):
    return worst_slices(slice_metrics(prediction, reference), k=k)


def slice_tp_fp_fn(*args, **kwargs):
    return slice_metrics(*args, **kwargs)


def per_slice_confusion(*args, **kwargs):
    return slice_metrics(*args, **kwargs)


def compute_per_slice_metrics(*args, **kwargs):
    return slice_metrics(*args, **kwargs)


def worst_slice(*args, **kwargs):
    return worst_slices(*args, **kwargs)


def get_worst_slices(*args, **kwargs):
    return worst_slices(*args, **kwargs)


def worst_slices_report(*args, **kwargs):
    return worst_slices(*args, **kwargs)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _preprocessing_description() -> dict[str, Any]:
    return {
        "transforms": [
            "LoadImaged(keys='image', image_only=False)",
            "EnsureChannelFirstd(keys='image')",
            "Orientationd(keys='image', axcodes='RAS')",
            "Spacingd(keys='image', pixdim=(1.5, 1.5, 2.0), mode='bilinear')",
            "ScaleIntensityRanged(keys='image', a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True)",
            "EnsureTyped(keys='image', dtype=float32, track_meta=True)",
        ],
        "model": "MONAI UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(16,32,64,128,256), strides=(2,2,2,2), num_res_units=2, norm='batch')",
        "sliding_window": {
            "roi_size": [96, 96, 96],
            "sw_batch_size": 4,
            "overlap": 0.5,
        },
        "postprocessing": "softmax over 2 classes, argmax, restored with Invertd to original NIfTI grid",
    }
def _load_reference_mask(label_path: str | Path) -> tuple[np.ndarray, Any, Any]:
    import nibabel as nib

    image = nib.load(str(label_path))
    array = np.asarray(image.get_fdata())
    if array.ndim != 3:
        raise ValueError(f"label is not a 3D volume: shape={array.shape}")
    _validate_label_values(array)
    return array.astype(np.uint8, copy=False), image, image.affine


def _run_model_prediction(
    image_path: str | Path,
    *,
    model: Any,
    device: str,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
) -> tuple[np.ndarray, Any, float]:
    import segmentation

    modules = segmentation._optional_modules()
    nib = modules[0]
    sliding_window_inference = modules[1]
    preprocess = segmentation._preprocessing_transforms(modules)

    original = nib.load(str(image_path))
    qform, qcode = original.get_qform(coded=True)
    sform, scode = original.get_sform(coded=True)
    geometry = segmentation._validate_numeric_geometry(
        original.shape,
        original.affine,
        qform=qform if qcode else None,
        sform=sform if scode else None,
    )
    axis_codes = nib.aff2axcodes(geometry.affine)
    if len(axis_codes) != 3 or any(code is None for code in axis_codes) or len(set(axis_codes)) != 3:
        raise ValueError("NIfTI affine does not define three unique anatomical axis directions.")
    geometry = segmentation.VolumeGeometry(
        geometry.shape, geometry.affine, geometry.spacing, tuple(axis_codes)
    )

    array = segmentation.validate_volume_array(original.get_fdata(dtype=np.float32))
    if tuple(array.shape) != geometry.shape:
        raise ValueError("Decoded voxel shape does not match NIfTI header dimensions.")

    started = time.perf_counter()
    transformed = preprocess({"image": str(image_path)})
    image = transformed["image"]
    probabilities = segmentation.infer_preprocessed_volume(
        image[0].detach().cpu().numpy(),
        model,
        sliding_window_inference,
        device=device,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
    )
    transformed["pred"] = type(image)(probabilities)
    Invertd = modules[5]
    invert = Invertd(
        keys="pred",
        transform=preprocess,
        orig_keys="image",
        nearest_interp=False,
        to_tensor=True,
    )
    restored = invert(transformed)["pred"]
    if restored.ndim == 5 and restored.shape[0] == 1:
        restored = restored[0]
    mask = segmentation.probabilities_to_mask(restored, geometry.shape)
    elapsed = time.perf_counter() - started
    return mask, geometry, elapsed


def _evaluate_one_case(
    pair: CasePair,
    *,
    model: Any,
    device: str,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
    output_dir: Path | None,
    save_predictions: bool,
    save_slice_metrics: bool,
) -> CaseEvaluation:
    result = CaseEvaluation(
        case_id=pair.case_id,
        status="pending",
        image_path=str(pair.image_path),
        label_path=str(pair.label_path),
    )
    try:
        validate_shape_and_affine(pair.image_path, pair.label_path)
        reference, label_image, affine = _load_reference_mask(pair.label_path)
        mask, geometry, elapsed = _run_model_prediction(
            pair.image_path,
            model=model,
            device=device,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
        )
        if tuple(mask.shape) != tuple(reference.shape):
            raise ValueError(
                f"prediction shape {tuple(mask.shape)} does not match reference shape {tuple(reference.shape)}"
            )
        counts = confusion_counts(mask, reference)
        result.shape = [int(v) for v in mask.shape]
        result.affine = geometry.affine.tolist()
        result.predicted_voxels = int(np.count_nonzero(mask))
        result.reference_voxels = int(np.count_nonzero(reference))
        result.true_positives = counts["tp"]
        result.false_positives = counts["fp"]
        result.false_negatives = counts["fn"]
        result.true_negatives = counts["tn"]
        result.dice = dice_from_counts(counts["tp"], counts["fp"], counts["fn"])
        result.precision = precision_from_counts(
            counts["tp"], counts["fp"], counts["tp"] + counts["fn"]
        )
        result.recall = recall_from_counts(
            counts["tp"], counts["fn"], counts["tp"] + counts["fp"]
        )
        result.worst_slices = worst_slices_for_case(mask, reference, k=5)
        result.runtime_seconds = float(elapsed)
        if save_predictions and output_dir is not None:
            import segmentation

            prediction_path = output_dir / "predictions" / f"{pair.case_id}_pred.nii.gz"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            import nibabel as nib

            segmentation.save_mask_nifti(
                nib, prediction_path, mask.astype(np.uint8), geometry, label_image.header
            )
            result.prediction_path = str(prediction_path)
        if (save_predictions or save_slice_metrics) and output_dir is not None:
            slice_dir = output_dir / "slice_metrics"
            slice_dir.mkdir(parents=True, exist_ok=True)
            slice_path = slice_dir / f"{pair.case_id}_slice_metrics.json"
            slice_payload = {
                "case_id": pair.case_id,
                "worst_slices": result.worst_slices,
                "slice_metrics": slice_metrics(mask, reference),
            }
            slice_path.write_text(
                json.dumps(slice_payload, indent=2, default=str), encoding="utf-8"
            )
            result.slice_metrics_path = str(slice_path)
        result.status = "success"
    except Exception as exc:  # noqa: BLE001 - evaluation must record, not drop, failures.
        result.status = "error"
        result.error = str(exc)
    return result
def _aggregate_results(cases: list[CaseEvaluation]) -> dict[str, Any]:
    successful = [case for case in cases if case.status == "success"]
    if not successful:
        return {
            "successful_cases": 0,
            "mean_dice": None,
            "mean_precision": None,
            "mean_recall": None,
            "pooled_dice": None,
            "pooled_precision": None,
            "pooled_recall": None,
            "total_predicted_voxels": 0,
            "total_reference_voxels": 0,
            "total_true_positives": 0,
            "total_false_positives": 0,
            "total_false_negatives": 0,
        }
    tp = sum(case.true_positives or 0 for case in successful)
    fp = sum(case.false_positives or 0 for case in successful)
    fn = sum(case.false_negatives or 0 for case in successful)
    pred_voxels = sum(case.predicted_voxels or 0 for case in successful)
    ref_voxels = sum(case.reference_voxels or 0 for case in successful)
    return {
        "successful_cases": len(successful),
        "mean_dice": float(np.mean([case.dice for case in successful])),
        "mean_precision": float(np.mean([case.precision for case in successful])),
        "mean_recall": float(np.mean([case.recall for case in successful])),
        "pooled_dice": dice_from_counts(tp, fp, fn),
        "pooled_precision": precision_from_counts(tp, fp, tp + fn),
        "pooled_recall": recall_from_counts(tp, fn, tp + fp),
        "total_predicted_voxels": int(pred_voxels),
        "total_reference_voxels": int(ref_voxels),
        "total_true_positives": int(tp),
        "total_false_positives": int(fp),
        "total_false_negatives": int(fn),
    }
def run_evaluation(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    checkpoint_path: str | Path,
    case_ids: Sequence[str] | None = None,
    limit: int | None = None,
    device: str = "cpu",
    roi_size: tuple[int, int, int] = (96, 96, 96),
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    save_predictions: bool = False,
    save_slice_metrics: bool = False,
    evaluation_split: str = EVALUATION_SPLIT,
) -> EvaluationReport:
    import segmentation

    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs, unmatched_images, extra_labels, mismatched_pairs = discover_case_pairs(dataset_root)
    pair_ids = {pair.case_id for pair in pairs}
    if case_ids:
        requested = list(case_ids)
        requested_set = set(requested)
        pairs = [pair for pair in pairs if pair.case_id in requested_set]
        missing_case_ids = [case_id for case_id in requested if case_id not in pair_ids]
    else:
        missing_case_ids = []
    if limit is not None:
        pairs = pairs[:limit]

    checkpoint_path = Path(checkpoint_path)
    checkpoint_hash = _sha256_file(checkpoint_path)
    model = segmentation.load_pretrained_unet(checkpoint_path)

    cases: list[CaseEvaluation] = []
    for case_id in missing_case_ids:
        cases.append(
            CaseEvaluation(
                case_id=case_id,
                status="error",
                image_path="",
                label_path="",
                error="Requested case was not found among matched image/label pairs.",
            )
        )
    for pair in pairs:
        case_result = _evaluate_one_case(
            pair,
            model=model,
            device=device,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            output_dir=output_dir,
            save_predictions=save_predictions,
            save_slice_metrics=save_slice_metrics,
        )
        cases.append(case_result)

    aggregate = _aggregate_results(cases)
    images = _real_nifti_files(dataset_root / "imagesTr")

    report = EvaluationReport(
        dataset_root=str(dataset_root),
        evaluation_split=evaluation_split,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=checkpoint_hash,
        checkpoint_sha256_matches=checkpoint_hash == CHECKPOINT_SHA256,
        preprocessing=_preprocessing_description(),
        total_images=len(images),
        matched_pairs=len(discover_case_pairs(dataset_root)[0]),
        evaluated_cases=len(cases),
        successful_cases=aggregate.get("successful_cases", 0),
        failed_cases=sum(1 for case in cases if case.status != "success"),
        unmatched_images=unmatched_images,
        extra_labels=extra_labels,
        mismatched_pairs=mismatched_pairs,
        cases=[asdict(case) for case in cases],
        aggregate=aggregate,
        generated_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    _write_reports(report, output_dir)
    return report
def _case_to_csv_row(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "status": case.get("status"),
        "image_path": case.get("image_path"),
        "label_path": case.get("label_path"),
        "shape": ";".join(str(v) for v in case.get("shape") or []),
        "predicted_voxels": case.get("predicted_voxels"),
        "reference_voxels": case.get("reference_voxels"),
        "true_positives": case.get("true_positives"),
        "false_positives": case.get("false_positives"),
        "false_negatives": case.get("false_negatives"),
        "true_negatives": case.get("true_negatives"),
        "dice": case.get("dice"),
        "precision": case.get("precision"),
        "recall": case.get("recall"),
        "runtime_seconds": case.get("runtime_seconds"),
        "error": case.get("error"),
        "prediction_path": case.get("prediction_path"),
    }


def _write_reports(report: EvaluationReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "spleen_evaluation_results.json"
    csv_path = output_dir / "spleen_evaluation_results.csv"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(asdict(report), stream, indent=2, default=str)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = list(_case_to_csv_row(report.cases[0] if report.cases else {}).keys())
        if not fieldnames:
            fieldnames = [
                "case_id", "status", "image_path", "label_path", "shape",
                "predicted_voxels", "reference_voxels", "true_positives",
                "false_positives", "false_negatives", "true_negatives", "dice",
                "precision", "recall", "runtime_seconds", "error", "prediction_path",
            ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for case in report.cases:
            writer.writerow(_case_to_csv_row(case))
