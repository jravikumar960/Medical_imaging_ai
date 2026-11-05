"""Research-only training and evaluation for a CXR-vs-photo modality gate.

This module does not enable inference in the application. Dataset provenance,
licenses, labels, and split construction must be verified by the dataset user.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

LABEL_TO_INDEX = {"non_cxr": 0, "cxr": 1}
MAX_CALIBRATED_THRESHOLD = 0.99
SPLITS = {"train", "validation", "calibration", "test"}
MANIFEST_COLUMNS = {
    "image_path",
    "label",
    "split",
    "group_id",
    "source",
    "category",
    "license",
    "attribution",
}
IMAGE_SIZE = 224
REQUIRED_NON_CXR_CATEGORIES = {"cars", "buildings", "people", "everyday_objects"}
MEDICAL_NON_CXR_CATEGORIES = {"ct", "mri", "ultrasound", "non_chest_xray"}
ALL_NON_CXR_CATEGORIES = REQUIRED_NON_CXR_CATEGORIES | MEDICAL_NON_CXR_CATEGORIES
MEDICAL_MANIFEST_REQUIRED_COLUMNS = {
    "image_id",
    "local_path",
    "group_id",
    "category",
    "source",
    "license",
    "attribution",
    "license_url",
    "source_url",
    "landing_url",
}
MEDICAL_NEGATIVE_TARGETS_BY_SPLIT = {
    "train": {category: 0 for category in MEDICAL_NON_CXR_CATEGORIES},
    "validation": {category: 0 for category in MEDICAL_NON_CXR_CATEGORIES},
    "calibration": {category: 0 for category in MEDICAL_NON_CXR_CATEGORIES},
    "test": {category: 0 for category in MEDICAL_NON_CXR_CATEGORIES},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest: pd.DataFrame, image_root: Path) -> pd.DataFrame:
    """Check source metadata, files, exact duplicates, classes, and group isolation."""
    missing = MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
    if manifest.empty:
        raise ValueError("Manifest contains no rows.")

    checked = manifest.copy()
    for column in (
        "label",
        "split",
        "group_id",
        "source",
        "category",
        "license",
        "attribution",
    ):
        checked[column] = checked[column].astype("string").str.strip()
        if checked[column].isna().any() or checked[column].eq("").any():
            raise ValueError(f"Manifest column {column!r} must be non-empty for every row.")

    invalid_labels = sorted(set(checked["label"]) - set(LABEL_TO_INDEX))
    if invalid_labels:
        raise ValueError(f"Unsupported labels: {', '.join(invalid_labels)}")
    invalid_splits = sorted(set(checked["split"]) - SPLITS)
    if invalid_splits:
        raise ValueError(f"Unsupported splits: {', '.join(invalid_splits)}")
    if set(checked["split"]) != SPLITS:
        raise ValueError(f"Manifest must contain all splits: {', '.join(sorted(SPLITS))}.")
    if set(checked.loc[checked["label"] == "cxr", "category"]) != {"chest_xray"}:
        raise ValueError("CXR rows must use category 'chest_xray'.")

    root = image_root.resolve()
    resolved_paths: list[Path] = []
    for row_index, value in checked["image_path"].items():
        if pd.isna(value) or not str(value).strip():
            raise ValueError(f"image_path is empty at row {row_index}.")
        path = (root / str(value)).resolve()
        if path == root or root not in path.parents:
            raise ValueError(f"image_path at row {row_index} escapes --image-root.")
        if not path.is_file():
            raise FileNotFoundError(f"Image for manifest row {row_index} does not exist: {path}")
        resolved_paths.append(path)
    checked["_resolved_path"] = resolved_paths

    split_by_group: dict[str, str] = {}
    split_by_hash: dict[str, str] = {}
    seen_paths: set[Path] = set()
    for row_index, row in checked.iterrows():
        path = row["_resolved_path"]
        if path in seen_paths:
            raise ValueError(f"Duplicate image_path in manifest: {row['image_path']}")
        seen_paths.add(path)

        group = str(row["group_id"])
        split = str(row["split"])
        previous_split = split_by_group.setdefault(group, split)
        if previous_split != split:
            raise ValueError(
                f"Group leakage: group_id {group!r} occurs in {previous_split!r} and {split!r}."
            )

        image_hash = _sha256(path)
        previous_hash_split = split_by_hash.get(image_hash)
        if previous_hash_split is not None:
            raise ValueError(
                f"Exact image duplicate occurs in {previous_hash_split!r} and {split!r}."
            )
        split_by_hash[image_hash] = split

    for split in SPLITS:
        labels = set(checked.loc[checked["split"] == split, "label"])
        if labels != set(LABEL_TO_INDEX):
            raise ValueError(f"Split {split!r} must contain both cxr and non_cxr images.")
        negative_categories = set(
            checked.loc[
                (checked["split"] == split) & (checked["label"] == "non_cxr"),
                "category",
            ]
        )
        missing_categories = REQUIRED_NON_CXR_CATEGORIES - negative_categories
        if missing_categories:
            raise ValueError(
                f"Split {split!r} is missing non-CXR categories: "
                f"{', '.join(sorted(missing_categories))}."
            )
    return checked


def _pil_rgb(image: Image.Image) -> Image.Image:
    pixels = np.asarray(image)
    if pixels.ndim == 2 and pixels.dtype.itemsize > 1:
        values = pixels.astype(np.float32)
        low, high = np.percentile(values, (0.5, 99.5))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError("16-bit grayscale image has no usable intensity range.")
        gray = np.clip((values - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
        return Image.fromarray(gray).convert("RGB")
    return image.convert("RGB")


def _transform(training: bool) -> transforms.Compose:
    operations: list[Any] = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if training:
        operations.append(
            transforms.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.95, 1.05))
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
        ]
    )
    return transforms.Compose(operations)


class ModalityImageDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, transform: transforms.Compose):
        self.rows = rows.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows.iloc[index]
        try:
            with Image.open(row["_resolved_path"]) as image:
                tensor = self.transform(_pil_rgb(image))
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise ValueError(
                f"Could not decode manifest image {row['image_path']!r}: {exc}"
            ) from exc
        return tensor, LABEL_TO_INDEX[str(row["label"])]


def pil_rgb(image: Image.Image) -> Image.Image:
    """Public alias for the training-time RGB conversion helper."""
    return _pil_rgb(image)


def build_transform(training: bool = False) -> transforms.Compose:
    """Public alias for the training/inference transform."""
    return _transform(training)
def build_model(pretrained_backbone: str = "none") -> nn.Module:
    """Build the small binary modality classifier.

    The default remains randomly initialized so importing or testing this module
    never downloads weights. ImageNet weights are opt-in for training.
    """
    if pretrained_backbone not in {"none", "imagenet1k_v1"}:
        raise ValueError(f"Unsupported pretrained backbone: {pretrained_backbone!r}")
    weights = None
    if pretrained_backbone == "imagenet1k_v1":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_small(weights=weights)
    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, len(LABEL_TO_INDEX))
    return model


def _predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    targets: list[int] = []
    scores: list[float] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            if logits.ndim != 2 or logits.shape != (len(labels), 2):
                raise ValueError(f"Model returned invalid logits shape {tuple(logits.shape)}.")
            if not torch.isfinite(logits).all():
                raise ValueError("Model returned non-finite logits.")
            probabilities = torch.softmax(logits, dim=1)[:, LABEL_TO_INDEX["cxr"]]
            targets.extend(labels.tolist())
            scores.extend(probabilities.cpu().tolist())
    return np.asarray(targets, dtype=np.int8), np.asarray(scores, dtype=np.float64)


def select_threshold(
    targets: np.ndarray, scores: np.ndarray, max_false_accept_rate: float
) -> float:
    """Maximize CXR recall on calibration while meeting its non-CXR FAR limit."""
    if not 0 < max_false_accept_rate < 1:
        raise ValueError("max_false_accept_rate must be strictly between 0 and 1.")
    non_cxr_count = int(np.sum(targets == LABEL_TO_INDEX["non_cxr"]))
    minimum_count = math.ceil(3.0 / max_false_accept_rate)
    if non_cxr_count < minimum_count:
        raise ValueError(
            f"Calibration needs at least {minimum_count} non-CXR images for the requested "
            f"FAR target; found {non_cxr_count}."
        )
    if targets.ndim != 1 or scores.ndim != 1 or len(targets) != len(scores):
        raise ValueError("Calibration targets and scores must be aligned one-dimensional arrays.")
    if not np.isfinite(scores).all():
        raise ValueError("Calibration scores must be finite and align with targets.")
    if set(np.unique(targets)) != set(LABEL_TO_INDEX.values()):
        raise ValueError("Calibration split must contain both classes.")

    candidates = np.concatenate(([np.nextafter(1.0, np.inf)], np.unique(scores)))
    eligible: list[tuple[int, float]] = []
    for threshold in candidates:
        predicted_cxr = scores >= threshold
        false_accepts = int(np.sum(predicted_cxr & (targets == 0)))
        far = false_accepts / non_cxr_count
        if far <= max_false_accept_rate:
            true_accepts = int(np.sum(predicted_cxr & (targets == 1)))
            eligible.append((true_accepts, float(threshold)))
    if not eligible:
        raise ValueError("No calibration threshold satisfies the requested FAR target.")
    threshold = max(eligible, key=lambda item: (item[0], item[1]))[1]
    buffered_threshold = min(threshold, MAX_CALIBRATED_THRESHOLD)
    if buffered_threshold < threshold:
        false_accepts = int(
            np.sum((scores >= buffered_threshold) & (targets == LABEL_TO_INDEX["non_cxr"]))
        )
        if false_accepts / non_cxr_count <= max_false_accept_rate:
            return buffered_threshold
    return threshold


def classification_metrics(
    targets: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    predicted = scores >= threshold
    true_cxr = targets == LABEL_TO_INDEX["cxr"]
    true_non_cxr = ~true_cxr
    true_positive = int(np.sum(predicted & true_cxr))
    false_negative = int(np.sum(~predicted & true_cxr))
    false_positive = int(np.sum(predicted & true_non_cxr))
    true_negative = int(np.sum(~predicted & true_non_cxr))
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    non_cxr_denominator = true_negative + false_positive
    cxr_denominator = true_positive + false_negative
    return {
        "n": int(len(targets)),
        "confusion_matrix": [[true_negative, false_positive], [false_negative, true_positive]],
        "false_accept_rate_non_cxr": (
            false_positive / non_cxr_denominator if non_cxr_denominator else None
        ),
        "false_reject_rate_cxr": (
            false_negative / cxr_denominator if cxr_denominator else None
        ),
        "precision_cxr": (
            true_positive / precision_denominator if precision_denominator else None
        ),
        "recall_cxr": true_positive / recall_denominator if recall_denominator else None,
    }


def _make_loader(rows, *, training: bool, batch_size: int, workers: int) -> DataLoader:
    dataset = ModalityImageDataset(rows, _transform(training))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_and_evaluate(
    manifest_path: Path,
    image_root: Path,
    output_dir: Path,
    *,
    epochs: int = 8,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    workers: int = 0,
    max_false_accept_rate: float = 0.001,
    seed: int = 42,
    pretrained_backbone: str = "none",
) -> dict[str, Any]:
    manifest = validate_manifest(pd.read_csv(manifest_path), image_root)
    minimum_test_negatives = math.ceil(3.0 / max_false_accept_rate)
    test_negative_count = int(
        (
            (manifest["split"] == "test")
            & (manifest["label"] == "non_cxr")
        ).sum()
    )
    if test_negative_count < minimum_test_negatives:
        raise ValueError(
            f"Held-out test needs at least {minimum_test_negatives} non-CXR images for the "
            f"requested FAR target; found {test_negative_count}."
        )
    torch.manual_seed(seed)
    np.random.seed(seed)
    loaders = {
        split: _make_loader(
            manifest.loc[manifest["split"] == split],
            training=(split == "train"),
            batch_size=batch_size,
            workers=workers,
        )
        for split in SPLITS
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained_backbone).to(device)
    counts = manifest.loc[manifest["split"] == "train", "label"].value_counts()
    class_weights = torch.tensor(
        [
            len(manifest.loc[manifest["split"] == "train"]) / (2 * counts[label])
            for label in ("non_cxr", "cxr")
        ],
        dtype=torch.float32,
        device=device,
    )
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_validation_loss = float("inf")
    best_state = None
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            total += len(labels)

        model.eval()
        validation_loss = 0.0
        validation_total = 0
        with torch.inference_mode():
            for images, labels in loaders["validation"]:
                logits = model(images.to(device))
                loss = loss_function(logits, labels.to(device))
                validation_loss += float(loss) * len(labels)
                validation_total += len(labels)
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / total,
            "validation_loss": validation_loss / validation_total,
        }
        history.append(epoch_record)
        if epoch_record["validation_loss"] < best_validation_loss:
            best_validation_loss = epoch_record["validation_loss"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    calibration_targets, calibration_scores = _predict(model, loaders["calibration"], device)
    threshold = select_threshold(
        calibration_targets, calibration_scores, max_false_accept_rate
    )
    test_targets, test_scores = _predict(model, loaders["test"], device)
    test_categories = manifest.loc[
        manifest["split"] == "test", "category"
    ].reset_index(drop=True)
    test_sources = manifest.loc[
        manifest["split"] == "test", "source"
    ].reset_index(drop=True)
    test_metrics_by_category = {}
    for category in sorted(test_categories.unique()):
        category_mask = test_categories.eq(category).to_numpy()
        test_metrics_by_category[str(category)] = classification_metrics(
            test_targets[category_mask], test_scores[category_mask], threshold
        )
    test_metrics_by_source = {}
    for source in sorted(test_sources.unique()):
        source_mask = test_sources.eq(source).to_numpy()
        test_metrics_by_source[str(source)] = classification_metrics(
            test_targets[source_mask], test_scores[source_mask], threshold
        )
    report = {
        "status": "research_only_not_connected_to_application_inference",
        "architecture": "torchvision.mobilenet_v3_small",
        "pretrained_weights": pretrained_backbone != "none",
        "pretrained_backbone": pretrained_backbone,
        "input_size": IMAGE_SIZE,
        "positive_label": "cxr",
        "negative_label": "non_cxr",
        "threshold_source": "calibration split only",
        "threshold": threshold,
        "requested_max_calibration_false_accept_rate": max_false_accept_rate,
        "calibration": classification_metrics(
            calibration_targets, calibration_scores, threshold
        ),
        "held_out_test": classification_metrics(test_targets, test_scores, threshold),
        "held_out_test_by_category": test_metrics_by_category,
        "held_out_test_by_source": test_metrics_by_source,
        "test_source_counts": {
            str(source): int(count)
            for source, count in (
                manifest.loc[manifest["split"] == "test", "source"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "training_history": history,
        "limitations": [
            "Research-only metrics do not establish deployment safety or clinical validity.",
            "The model is not loaded by the application and does not change its fail-closed gate.",
            "Split integrity depends on trustworthy source, group, license, and attribution metadata.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "threshold": threshold,
            "labels": LABEL_TO_INDEX,
            "architecture": "torchvision.mobilenet_v3_small",
            "input_size": IMAGE_SIZE,
            "preprocessing": {
                "resize": [IMAGE_SIZE, IMAGE_SIZE],
                "color": "RGB",
                "mean": [0.5, 0.5, 0.5],
                "std": [0.25, 0.25, 0.25],
            },
            "training": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
                "pretrained_backbone": pretrained_backbone,
                "max_false_accept_rate": max_false_accept_rate,
            },
            "dataset_provenance": {
                "manifest_path": str(Path(manifest_path)),
                "manifest_sha256": _sha256(Path(manifest_path)),
                "image_root": str(Path(image_root)),
            },
            "evaluation": {
                "calibration": classification_metrics(
                    calibration_targets, calibration_scores, threshold
                ),
                "held_out_test": classification_metrics(test_targets, test_scores, threshold),
            },
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        output_dir / "modality_classifier.pt",
    )
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-false-accept-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pretrained-backbone",
        choices=("none", "imagenet1k_v1"),
        default="none",
    )
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.workers < 0
        or args.learning_rate <= 0
        or not 0 < args.max_false_accept_rate < 1
    ):
        parser.error(
            "epochs and batch-size must be positive; workers non-negative; learning rate "
            "positive; FAR target strictly between 0 and 1."
        )
    report = train_and_evaluate(
        args.manifest,
        args.image_root,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        workers=args.workers,
        max_false_accept_rate=args.max_false_accept_rate,
        seed=args.seed,
        pretrained_backbone=args.pretrained_backbone,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
