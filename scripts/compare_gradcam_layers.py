"""Comprehensive Grad-CAM Layer Comparison Experiment.
Compares denseblock3 (model.features.denseblock3.denselayer12.conv2) vs.
denseblock4 (model.features.denseblock4.denselayer16.conv2) across representative chest X-rays
and target findings, calculating border mass, lung mask mass, sparsity, deletion/insertion AUC,
and generating side-by-side visual artifacts and reproducibility reports.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torchxrayvision as xrv
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from imaging import prepare_image, gradcam_detailed, overlay_heatmap, gradcam_layer
from xray_gradcam_localization import border_mask, mass_fraction, estimate_unvalidated_lung_mask

OUT_DIR = ROOT / "outputs" / "gradcam_layer_comparison"
VIS_DIR = OUT_DIR / "visuals"
REP_DIR = OUT_DIR / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)
REP_DIR.mkdir(parents=True, exist_ok=True)

PATCH_GRID = 14
PATCH_SIZE = 224 // PATCH_GRID  # 16
DELETION_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
RANDOM_SEEDS = [0, 1, 2]

TARGET_FINDINGS = [
    "Infiltration",
    "Pneumonia",
    "Lung Opacity",
    "Cardiomegaly",
    "Atelectasis",
    "Fibrosis",
]


def compute_file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def hoyer_sparsity(cam: np.ndarray) -> float:
    flat = np.asarray(cam, dtype=np.float64).ravel()
    n = flat.size
    if n <= 1:
        return 0.0
    l1 = float(np.abs(flat).sum())
    l2 = float(np.sqrt((flat ** 2).sum()))
    if l2 == 0:
        return 0.0
    s = (np.sqrt(n) - (l1 / l2)) / (np.sqrt(n) - 1.0)
    return float(np.clip(s, 0.0, 1.0))


def gini_coefficient(cam: np.ndarray) -> float:
    flat = np.asarray(cam, dtype=np.float64).ravel()
    flat = np.clip(flat, 0, None)
    if flat.sum() == 0:
        return 0.0
    sorted_flat = np.sort(flat)
    n = sorted_flat.size
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * sorted_flat) / (n * np.sum(sorted_flat))) - ((n + 1.0) / n))


def patch_scores_from_cam(cam: np.ndarray, grid: int = PATCH_GRID, patch_size: int = PATCH_SIZE):
    scores = np.zeros((grid, grid), dtype=np.float64)
    for i in range(grid):
        for j in range(grid):
            y0, y1 = i * patch_size, (i + 1) * patch_size
            x0, x1 = j * patch_size, (j + 1) * patch_size
            scores[i, j] = float(cam[y0:y1, x0:x1].mean())
    return scores


def mask_patches(pil_224: Image.Image, patch_mask: np.ndarray, fill_value: float) -> Image.Image:
    arr = np.asarray(pil_224.convert("L"), dtype=np.float32).copy()
    h, w = arr.shape
    grid = patch_mask.shape[0]
    ph, pw = h // grid, w // grid
    for i in range(grid):
        for j in range(grid):
            if patch_mask[i, j]:
                arr[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw] = fill_value
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")


def insert_patches(pil_224: Image.Image, patch_mask: np.ndarray, fill_value: float) -> Image.Image:
    orig_arr = np.asarray(pil_224.convert("L"), dtype=np.float32)
    arr = np.full_like(orig_arr, fill_value)
    h, w = arr.shape
    grid = patch_mask.shape[0]
    ph, pw = h // grid, w // grid
    for i in range(grid):
        for j in range(grid):
            if patch_mask[i, j]:
                arr[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw] = orig_arr[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")


def evaluate_perturbation(model, pil_224, patch_scores, target_index, baseline_score, mode="deletion", fill_value=0.0):
    flat = patch_scores.ravel()
    order = np.argsort(-flat)  # descending
    curve = []
    for frac in DELETION_LEVELS:
        n = int(round(frac * flat.size))
        mask = np.zeros_like(flat, dtype=bool)
        if n > 0:
            mask[order[:n]] = True
        mask_2d = mask.reshape((PATCH_GRID, PATCH_GRID))
        if mode == "deletion":
            perturbed = mask_patches(pil_224, mask_2d, fill_value)
        else:
            perturbed = insert_patches(pil_224, mask_2d, fill_value)

        arr224 = np.asarray(perturbed, dtype=np.float32)
        norm = (arr224 / 255.0 - 0.5) * 2048.0
        ptensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)
        with torch.inference_mode():
            score = float(model(ptensor)[0, target_index].detach().cpu())
        curve.append({
            "fraction": float(frac),
            "score": round(score, 6),
            "change": round(baseline_score - score if mode == "deletion" else score - baseline_score, 6),
        })
    return curve


def compute_auc(curve) -> float:
    fractions = [c["fraction"] for c in curve]
    scores = [c["score"] for c in curve]
    if len(fractions) < 2:
        return 0.0
    return float(np.trapezoid(scores, fractions) / (fractions[-1] - fractions[0]))


def render_raw_cam(cam: np.ndarray) -> Image.Image:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cam, cmap="viridis")
    ax.axis("off")
    fig.tight_layout(pad=0)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def save_side_by_side_visual(image_label: str, finding: str, original_pil, cam3, cam4, overlay3, overlay4):
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))

    # 1. Original
    axes[0].imshow(original_pil.convert("L"), cmap="gray")
    axes[0].set_title(f"Original\n{image_label}")
    axes[0].axis("off")

    # 2. DB3 Raw CAM
    axes[1].imshow(cam3, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("DenseBlock3\nRaw CAM")
    axes[1].axis("off")

    # 3. DB3 Overlay
    buf3 = BytesIO()
    overlay3.savefig(buf3, format="png", bbox_inches="tight", pad_inches=0)
    buf3.seek(0)
    axes[2].imshow(Image.open(buf3))
    axes[2].set_title("DenseBlock3\nOverlay")
    axes[2].axis("off")

    # 4. DB4 Raw CAM
    axes[3].imshow(cam4, cmap="viridis", vmin=0, vmax=1)
    axes[3].set_title("DenseBlock4\nRaw CAM")
    axes[3].axis("off")

    # 5. DB4 Overlay
    buf4 = BytesIO()
    overlay4.savefig(buf4, format="png", bbox_inches="tight", pad_inches=0)
    buf4.seek(0)
    axes[4].imshow(Image.open(buf4))
    axes[4].set_title("DenseBlock4\nOverlay")
    axes[4].axis("off")

    fig.suptitle(f"Grad-CAM Layer Comparison — Finding: {finding}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    safe_name = f"{image_label}_{finding.replace(' ', '_')}.png"
    out_path = VIS_DIR / safe_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    print("Initializing DenseNet-121 and PSPNet segmentation model...")
    model = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
    seg_model = xrv.baseline_models.chestx_det.PSPNet().eval()

    test_images = [
        ("kaggle_normal", ROOT / "outputs" / "first_inference" / "chest_xray_kaggle_normal.jpeg"),
        ("nih_10_infiltration", ROOT / "outputs" / "xray_labeled_evaluation" / "images" / "10_0.png"),
    ]

    pathologies = list(model.pathologies)
    valid_findings = [f for f in TARGET_FINDINGS if f in pathologies]
    print(f"Evaluated findings ({len(valid_findings)}): {valid_findings}")

    results = []
    csv_rows = []

    border_16 = border_mask((224, 224), 16)

    for img_label, img_path in test_images:
        if not img_path.exists():
            print(f"Skipping {img_label}: file not found at {img_path}")
            continue
        print(f"\nProcessing image: {img_label} ({img_path.name})...")
        sha256 = compute_file_sha256(img_path)

        with Image.open(img_path) as src:
            src.load()
            orig = src.copy()

        # Preprocess
        tensor, pil_224 = prepare_image(orig)
        base_arr = np.asarray(pil_224.convert("L"), dtype=np.float32)
        fill_val = float(base_arr.mean())

        # Unvalidated lung mask via PSPNet
        with torch.inference_mode():
            seg_out = seg_model(tensor)
        seg_logits = seg_out[0].detach().cpu().numpy()
        psp_lung = (seg_logits[4] > 0) | (seg_logits[5] > 0)
        lung_mask_img = Image.fromarray((psp_lung.astype(np.uint8) * 255))
        lung_mask_224 = np.asarray(lung_mask_img.resize((224, 224), Image.Resampling.NEAREST), dtype=np.bool_)

        # Unvalidated Otsu heuristic lung mask
        otsu_lung = estimate_unvalidated_lung_mask(pil_224)

        # Baseline scores
        with torch.inference_mode():
            out = model(tensor)
        scores = out[0].detach().cpu().numpy()

        for finding in valid_findings:
            idx = pathologies.index(finding)
            baseline_score = float(scores[idx])

            # Run Grad-CAM for both layers
            norm_cam3, raw_cam3, score3 = gradcam_detailed(model, tensor, idx, layer_name="denseblock3")
            norm_cam4, raw_cam4, score4 = gradcam_detailed(model, tensor, idx, layer_name="denseblock4")

            assert score3 == score4 == baseline_score, "Prediction scores must remain invariant across layers"

            # Metrics for DB3
            b3_border = mass_fraction(norm_cam3, border_16)
            b3_center = 1.0 - b3_border
            b3_psp_lung = mass_fraction(norm_cam3, lung_mask_224)
            b3_otsu_lung = mass_fraction(norm_cam3, otsu_lung)
            b3_hoyer = hoyer_sparsity(norm_cam3)
            b3_gini = gini_coefficient(norm_cam3)

            # Metrics for DB4
            b4_border = mass_fraction(norm_cam4, border_16)
            b4_center = 1.0 - b4_border
            b4_psp_lung = mass_fraction(norm_cam4, lung_mask_224)
            b4_otsu_lung = mass_fraction(norm_cam4, otsu_lung)
            b4_hoyer = hoyer_sparsity(norm_cam4)
            b4_gini = gini_coefficient(norm_cam4)

            # Deletion & Insertion curves
            patch3 = patch_scores_from_cam(norm_cam3)
            patch4 = patch_scores_from_cam(norm_cam4)

            del3 = evaluate_perturbation(model, pil_224, patch3, idx, baseline_score, mode="deletion", fill_value=fill_val)
            del4 = evaluate_perturbation(model, pil_224, patch4, idx, baseline_score, mode="deletion", fill_value=fill_val)
            ins3 = evaluate_perturbation(model, pil_224, patch3, idx, baseline_score, mode="insertion", fill_value=fill_val)
            ins4 = evaluate_perturbation(model, pil_224, patch4, idx, baseline_score, mode="insertion", fill_value=fill_val)

            auc_del3 = compute_auc(del3)
            auc_del4 = compute_auc(del4)
            auc_ins3 = compute_auc(ins3)
            auc_ins4 = compute_auc(ins4)

            # Visual overlays
            fig3 = overlay_heatmap(pil_224, norm_cam3)
            fig4 = overlay_heatmap(pil_224, norm_cam4)
            vis_path = save_side_by_side_visual(img_label, finding, orig, norm_cam3, norm_cam4, fig3, fig4)
            plt.close(fig3)
            plt.close(fig4)

            record = {
                "image_label": img_label,
                "image_sha256": sha256,
                "finding": finding,
                "target_index": idx,
                "baseline_score": round(baseline_score, 6),
                "denseblock3": {
                    "border_16px_mass": round(b3_border, 4),
                    "center_mass": round(b3_center, 4),
                    "psp_lung_mask_mass": round(b3_psp_lung, 4),
                    "otsu_lung_mask_mass": round(b3_otsu_lung, 4),
                    "hoyer_sparsity": round(b3_hoyer, 4),
                    "gini_index": round(b3_gini, 4),
                    "deletion_auc": round(auc_del3, 4),
                    "insertion_auc": round(auc_ins3, 4),
                    "deletion_curve": del3,
                    "insertion_curve": ins3,
                },
                "denseblock4": {
                    "border_16px_mass": round(b4_border, 4),
                    "center_mass": round(b4_center, 4),
                    "psp_lung_mask_mass": round(b4_psp_lung, 4),
                    "otsu_lung_mask_mass": round(b4_otsu_lung, 4),
                    "hoyer_sparsity": round(b4_hoyer, 4),
                    "gini_index": round(b4_gini, 4),
                    "deletion_auc": round(auc_del4, 4),
                    "insertion_auc": round(auc_ins4, 4),
                    "deletion_curve": del4,
                    "insertion_curve": ins4,
                },
                "visual_artifact": str(vis_path.relative_to(ROOT)),
            }
            results.append(record)

            csv_rows.append({
                "image_label": img_label,
                "finding": finding,
                "baseline_score": round(baseline_score, 6),
                "db3_border_mass": round(b3_border, 4),
                "db4_border_mass": round(b4_border, 4),
                "db3_psp_lung_mass": round(b3_psp_lung, 4),
                "db4_psp_lung_mass": round(b4_psp_lung, 4),
                "db3_hoyer": round(b3_hoyer, 4),
                "db4_hoyer": round(b4_hoyer, 4),
                "db3_del_auc": round(auc_del3, 4),
                "db4_del_auc": round(auc_del4, 4),
                "db3_ins_auc": round(auc_ins3, 4),
                "db4_ins_auc": round(auc_ins4, 4),
            })
            print(f"  [{finding}] score={baseline_score:.3f} | DB3 border={b3_border:.3f}, del_auc={auc_del3:.3f} | DB4 border={b4_border:.3f}, del_auc={auc_del4:.3f}")

    # Save JSON report
    report_json = REP_DIR / "layer_comparison_report.json"
    with open(report_json, "w") as f:
        json.dump({
            "experiment": "DenseBlock3 vs DenseBlock4 Grad-CAM Comparison",
            "model": "densenet121-res224-all",
            "patch_grid": PATCH_GRID,
            "patch_size": PATCH_SIZE,
            "deletion_levels": DELETION_LEVELS,
            "records": results,
        }, f, indent=2)

    # Save CSV summary
    csv_path = REP_DIR / "layer_comparison_summary.csv"
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    # Save experiment metadata
    meta = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "torchxrayvision_version": xrv.__version__,
        "numpy_version": np.__version__,
        "pillow_version": PIL.__version__,
        "random_seeds": RANDOM_SEEDS,
    }
    with open(REP_DIR / "experiment_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nExperiment complete! Evaluated {len(results)} image-finding pairs.")
    print(f"JSON Report: {report_json}")
    print(f"CSV Summary: {csv_path}")
    print(f"Visuals: {VIS_DIR}")


if __name__ == "__main__":
    main()
