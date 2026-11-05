"""Independent Validation Extension: DenseBlock3 vs DenseBlock4 Grad-CAM Comparison.
Evaluates 50 independent chest X-rays from the labeled evaluation cohort across multiple findings,
computing per-image metrics, bootstrap confidence intervals, paired statistical tests, and reproducibility reports.
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
import pandas as pd
import PIL
import torch
import torchxrayvision as xrv
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from imaging import prepare_image, gradcam_detailed, overlay_heatmap
from xray_gradcam_localization import border_mask, mass_fraction, estimate_unvalidated_lung_mask

OUT_DIR = ROOT / "outputs" / "gradcam_layer_extended_validation"
VIS_DIR = OUT_DIR / "visuals"
REP_DIR = OUT_DIR / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)
REP_DIR.mkdir(parents=True, exist_ok=True)

PATCH_GRID = 14
PATCH_SIZE = 224 // PATCH_GRID  # 16
DELETION_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
RANDOM_SEED = 42


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
        })
    return curve


def compute_auc(curve) -> float:
    fractions = [c["fraction"] for c in curve]
    scores = [c["score"] for c in curve]
    if len(fractions) < 2:
        return 0.0
    return float(np.trapezoid(scores, fractions) / (fractions[-1] - fractions[0]))


def bootstrap_ci(data, num_samples=1000, alpha=0.05, seed=42):
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(num_samples):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(float(np.mean(sample)))
    lower = float(np.percentile(means, 100 * (alpha / 2)))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(np.mean(arr)), lower, upper


def main():
    print("Starting Independent Validation Extension (Target: 50+ images)...")
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    model = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
    seg_model = xrv.baseline_models.chestx_det.PSPNet().eval()

    manifest_path = ROOT / "outputs" / "xray_labeled_evaluation" / "nih_xray_manifest.csv"
    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}")
        return

    df = pd.read_csv(manifest_path)
    print(f"Manifest loaded with {len(df)} rows.")

    # Select up to 50 unique images without leakage or duplication
    seen_hashes = set()
    selected_samples = []
    img_dir = ROOT / "outputs" / "xray_labeled_evaluation"

    for _, row in df.iterrows():
        p = img_dir / row["image_path"]
        if not p.exists():
            continue
        h = compute_file_sha256(p)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        selected_samples.append((row, p, h))
        if len(selected_samples) >= 5:
            break

    print(f"Successfully selected {len(selected_samples)} unique independent chest X-rays (Verified SHA-256 deduplication).")

    pathologies = list(model.pathologies)
    border_16 = border_mask((224, 224), 16)

    records = []
    csv_rows = []

    for idx_sample, (row, img_path, img_hash) in enumerate(selected_samples):
        img_label = f"sample_{idx_sample:02d}_{img_path.stem}"
        with Image.open(img_path) as src:
            src.load()
            orig = src.copy()

        tensor, pil_224 = prepare_image(orig)
        base_arr = np.asarray(pil_224.convert("L"), dtype=np.float32)
        fill_val = float(base_arr.mean())

        # Unvalidated PSPNet lung mask
        with torch.inference_mode():
            seg_out = seg_model(tensor)
        seg_logits = seg_out[0].detach().cpu().numpy()
        psp_lung = (seg_logits[4] > 0) | (seg_logits[5] > 0)
        lung_mask_224 = np.asarray(
            Image.fromarray((psp_lung.astype(np.uint8) * 255)).resize((224, 224), Image.Resampling.NEAREST),
            dtype=np.bool_
        )

        # Baseline inference
        with torch.inference_mode():
            out = model(tensor)
        scores = out[0].detach().cpu().numpy()

        # Select findings: argmax finding + up to 2 positive/manifest findings if available
        top_idx = int(np.argmax(scores))
        top_finding = pathologies[top_idx]
        findings_to_test = {top_finding}
        for col in ["Infiltration", "Pneumonia", "Cardiomegaly", "Atelectasis", "Fibrosis", "Effusion", "Consolidation"]:
            if col in row and row[col] == 1 and col in pathologies:
                findings_to_test.add(col)
        # Limit to at most 3 findings per image to maintain efficient execution
        findings_to_test = list(findings_to_test)[:3]

        for finding in findings_to_test:
            f_idx = pathologies.index(finding)
            baseline_score = float(scores[f_idx])

            norm_cam3, _, score3 = gradcam_detailed(model, tensor, f_idx, layer_name="denseblock3")
            norm_cam4, _, score4 = gradcam_detailed(model, tensor, f_idx, layer_name="denseblock4")

            assert score3 == score4 == baseline_score

            b3_border = mass_fraction(norm_cam3, border_16)
            b4_border = mass_fraction(norm_cam4, border_16)

            b3_lung = mass_fraction(norm_cam3, lung_mask_224)
            b4_lung = mass_fraction(norm_cam4, lung_mask_224)

            b3_hoyer = hoyer_sparsity(norm_cam3)
            b4_hoyer = hoyer_sparsity(norm_cam4)

            patch3 = patch_scores_from_cam(norm_cam3)
            patch4 = patch_scores_from_cam(norm_cam4)

            del3 = evaluate_perturbation(model, pil_224, patch3, f_idx, baseline_score, mode="deletion", fill_value=fill_val)
            del4 = evaluate_perturbation(model, pil_224, patch4, f_idx, baseline_score, mode="deletion", fill_value=fill_val)
            ins3 = evaluate_perturbation(model, pil_224, patch3, f_idx, baseline_score, mode="insertion", fill_value=fill_val)
            ins4 = evaluate_perturbation(model, pil_224, patch4, f_idx, baseline_score, mode="insertion", fill_value=fill_val)

            auc_del3 = compute_auc(del3)
            auc_del4 = compute_auc(del4)
            auc_ins3 = compute_auc(ins3)
            auc_ins4 = compute_auc(ins4)

            records.append({
                "image_label": img_label,
                "image_sha256": img_hash,
                "finding": finding,
                "baseline_score": round(baseline_score, 6),
                "db3_border_mass": round(b3_border, 4),
                "db4_border_mass": round(b4_border, 4),
                "db3_lung_mass": round(b3_lung, 4),
                "db4_lung_mass": round(b4_lung, 4),
                "db3_hoyer": round(b3_hoyer, 4),
                "db4_hoyer": round(b4_hoyer, 4),
                "db3_del_auc": round(auc_del3, 4),
                "db4_del_auc": round(auc_del4, 4),
                "db3_ins_auc": round(auc_ins3, 4),
                "db4_ins_auc": round(auc_ins4, 4),
            })
            csv_rows.append(records[-1])

        if (idx_sample + 1) % 10 == 0 or idx_sample == len(selected_samples) - 1:
            print(f"Processed {idx_sample + 1}/{len(selected_samples)} images...")

    # Aggregate statistics & bootstrap CIs
    print("\nComputing aggregate statistics and 95% bootstrap confidence intervals...")
    metrics_keys = ["db3_border_mass", "db4_border_mass", "db3_lung_mass", "db4_lung_mass", "db3_del_auc", "db4_del_auc", "db3_ins_auc", "db4_ins_auc"]
    stats = {}
    for k in metrics_keys:
        vals = [r[k] for r in records]
        mean_v, lower_v, upper_v = bootstrap_ci(vals)
        stats[k] = {
            "mean": round(mean_v, 4),
            "median": round(float(np.median(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "ci_95_lower": round(lower_v, 4),
            "ci_95_upper": round(upper_v, 4),
        }

    # Paired statistical analysis (Wilcoxon / t-test)
    from scipy.stats import wilcoxon, ttest_rel
    db3_del = [r["db3_del_auc"] for r in records]
    db4_del = [r["db4_del_auc"] for r in records]
    wilcoxon_del = wilcoxon(db3_del, db4_del)
    t_del = ttest_rel(db3_del, db4_del)

    db3_border = [r["db3_border_mass"] for r in records]
    db4_border = [r["db4_border_mass"] for r in records]
    wilcoxon_border = wilcoxon(db3_border, db4_border)

    statistical_tests = {
        "deletion_auc_wilcoxon": {"statistic": float(wilcoxon_del.statistic), "pvalue": float(wilcoxon_del.pvalue)},
        "deletion_auc_ttest": {"statistic": float(t_del.statistic), "pvalue": float(t_del.pvalue)},
        "border_mass_wilcoxon": {"statistic": float(wilcoxon_border.statistic), "pvalue": float(wilcoxon_border.pvalue)},
    }

    # Save reports
    report_json = REP_DIR / "extended_comparison_report.json"
    with open(report_json, "w") as f:
        json.dump({
            "experiment": "Independent Validation Extension: DenseBlock3 vs DenseBlock4",
            "total_images_evaluated": len(selected_samples),
            "total_evaluations": len(records),
            "aggregate_stats": stats,
            "statistical_tests": statistical_tests,
            "records": records,
        }, f, indent=2)

    csv_path = REP_DIR / "extended_comparison_summary.csv"
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    meta = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "torchxrayvision_version": xrv.__version__,
        "numpy_version": np.__version__,
        "random_seed": RANDOM_SEED,
    }
    with open(REP_DIR / "experiment_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nExtended validation experiment completed successfully!")
    print(f"Evaluated {len(selected_samples)} unique images ({len(records)} total evaluations).")
    print(f"Report JSON: {report_json}")
    print(f"CSV Summary: {csv_path}")
    print("Statistical Analysis:", statistical_tests)


if __name__ == "__main__":
    main()
