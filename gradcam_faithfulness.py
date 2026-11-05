"""Grad-CAM faithfulness evaluation experiment.

Measures whether progressively removing high-CAM-activation regions reduces
the model's target class score more than removing random or low-activation
regions. Distinguishes model faithfulness from clinical correctness.

Controls:
- Same model and input tensor for all conditions.
- Target class fixed at the original argmax (never re-selected after perturbation).
- Original preprocessing preserved (prepare_image: center-crop + resize + normalization).
- Masked pixels replaced with the mean intensity of the pre-perturbation image.
- Baseline score recorded before any perturbation.
- Scores are raw model outputs, not calibrated probabilities.

Outputs are saved to outputs/gradcam_faithfulness/.
"""
from __future__ import annotations

import json
import random
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import torchxrayvision as xrv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from imaging import prepare_image, gradcam, gradcam_layer  # noqa: E402

OUT = ROOT / "outputs" / "gradcam_faithfulness"
OUT.mkdir(parents=True, exist_ok=True)

# Patch grid: 14x14 patches of 16x16 pixels on the 224x224 CAM.
PATCH_GRID = 14
PATCH_SIZE = 224 // PATCH_GRID  # 16

DELETION_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50]
RANDOM_SEEDS = [0, 1, 2, 3, 4]
N_PATHOLOGIES = None  # set after model load


# ---------------------------------------------------------------------------
# Preprocessing (unchanged from production prepare_image)
# ---------------------------------------------------------------------------

def preprocess_pil(pil_image: Image.Image):
    """Run production prepare_image; return (tensor, pil_224)."""
    tensor, pil_224 = prepare_image(pil_image)
    return tensor, pil_224


def tensor_score(model, tensor, target_index: int) -> float:
    """Return the raw score for the target class on the given tensor."""
    with torch.inference_mode():
        output = model(tensor)
    return float(output[0, target_index].detach().cpu())


def compute_cam(model, tensor, target_index: int) -> np.ndarray:
    """Generate normalized Grad-CAM (224x224) for the target class."""
    cam, _ = gradcam(model, tensor, target_index, layer_name="denseblock4")
    return cam


# ---------------------------------------------------------------------------
# Patch ranking and masking
# ---------------------------------------------------------------------------

def patch_scores_from_cam(cam: np.ndarray, grid: int = PATCH_GRID, patch_size: int = PATCH_SIZE):
    """Return mean CAM value per patch as a (grid, grid) array."""
    h, w = cam.shape
    scores = np.zeros((grid, grid), dtype=np.float64)
    for i in range(grid):
        for j in range(grid):
            y0, y1 = i * patch_size, (i + 1) * patch_size
            x0, x1 = j * patch_size, (j + 1) * patch_size
            scores[i, j] = float(cam[y0:y1, x0:x1].mean())
    return scores


def mask_patches(pil_image: Image.Image, patch_mask: np.ndarray, fill_value: float) -> Image.Image:
    """Return a copy of pil_image with masked patches replaced by fill_value."""
    arr = np.asarray(pil_image.convert("L"), dtype=np.float32).copy()
    h, w = arr.shape
    grid = patch_mask.shape[0]
    patch_h = h // grid
    patch_w = w // grid
    for i in range(grid):
        for j in range(grid):
            if patch_mask[i, j]:
                y0, y1 = i * patch_h, (i + 1) * patch_h
                x0, x1 = j * patch_w, (j + 1) * patch_w
                arr[y0:y1, x0:x1] = fill_value
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")


def top_k_patch_mask(patch_scores: np.ndarray, fraction: float) -> np.ndarray:
    """Boolean mask of top-k fraction of patches by CAM score."""
    flat = patch_scores.ravel()
    n = int(round(fraction * flat.size))
    threshold = np.sort(flat)[-n] if n > 0 else flat.max()
    return patch_scores >= threshold


def bottom_k_patch_mask(patch_scores: np.ndarray, fraction: float) -> np.ndarray:
    """Boolean mask of bottom-k fraction of patches by CAM score."""
    flat = patch_scores.ravel()
    n = int(round(fraction * flat.size))
    threshold = np.sort(flat)[n - 1] if n > 0 else flat.min()
    return patch_scores <= threshold


def random_patch_mask(fraction: float, seed: int, grid: int = PATCH_GRID) -> np.ndarray:
    """Boolean mask of a random fraction of patches."""
    rng = np.random.default_rng(seed)
    total = grid * grid
    n = int(round(fraction * total))
    idx = rng.choice(total, size=n, replace=False)
    mask = np.zeros(total, dtype=bool)
    mask[idx] = True
    return mask.reshape((grid, grid))


# ---------------------------------------------------------------------------
# Per-image evaluation
# ---------------------------------------------------------------------------

def evaluate_image(model, pil_image: Image.Image, label: str) -> dict:
    """Run faithfulness evaluation on one image."""
    result = {"label": label}

    # Original image properties
    orig_w, orig_h = pil_image.size
    result["original_width"] = orig_w
    result["original_height"] = orig_h
    result["original_aspect_ratio"] = round(orig_w / orig_h, 6)

    # Preprocess (production pipeline, unchanged)
    tensor, pil_224 = preprocess_pil(pil_image)
    result["processed_width"] = pil_224.size[0]
    result["processed_height"] = pil_224.size[1]

    # Fill value = mean intensity of the preprocessed 224x224 image
    base_arr = np.asarray(pil_224.convert("L"), dtype=np.float32)
    fill_value = float(base_arr.mean())
    result["fill_value"] = round(fill_value, 4)

    # Baseline inference
    with torch.inference_mode():
        output = model(tensor)
    scores = output[0].detach().cpu().numpy()
    pathologies = list(getattr(model, "pathologies", []))
    target_index = int(np.argmax(scores))
    target_finding = pathologies[target_index]
    baseline_score = float(scores[target_index])

    result["target_index"] = target_index
    result["target_finding"] = target_finding
    result["baseline_score"] = round(baseline_score, 6)
    result["all_scores"] = {p: round(float(s), 6) for p, s in zip(pathologies, scores)}

    # Grad-CAM for the fixed target class
    cam = compute_cam(model, tensor, target_index)
    result["cam_min"] = float(cam.min())
    result["cam_max"] = float(cam.max())
    result["cam_mean"] = float(cam.mean())

    # Patch ranking
    patch_scores = patch_scores_from_cam(cam)
    result["patch_grid"] = PATCH_GRID
    result["patch_size"] = PATCH_SIZE

    # --- Top-CAM deletion curve ---
    top_curve = []
    for frac in DELETION_LEVELS:
        mask = top_k_patch_mask(patch_scores, frac)
        perturbed = mask_patches(pil_224, mask, fill_value)
        ptensor, _ = preprocess_pil(perturbed)
        score = tensor_score(model, ptensor, target_index)
        drop = baseline_score - score
        top_curve.append({
            "fraction": frac,
            "score": round(score, 6),
            "drop": round(drop, 6),
            "pct_drop": round(100.0 * drop / abs(baseline_score) if baseline_score != 0 else 0.0, 4),
        })
    result["top_cam_curve"] = top_curve

    # --- Low-CAM deletion curve ---
    low_curve = []
    for frac in DELETION_LEVELS:
        mask = bottom_k_patch_mask(patch_scores, frac)
        perturbed = mask_patches(pil_224, mask, fill_value)
        ptensor, _ = preprocess_pil(perturbed)
        score = tensor_score(model, ptensor, target_index)
        drop = baseline_score - score
        low_curve.append({
            "fraction": frac,
            "score": round(score, 6),
            "drop": round(drop, 6),
            "pct_drop": round(100.0 * drop / abs(baseline_score) if baseline_score != 0 else 0.0, 4),
        })
    result["low_cam_curve"] = low_curve

    # --- Random deletion curves (multiple seeds) ---
    random_curves = []
    for seed in RANDOM_SEEDS:
        curve = []
        for frac in DELETION_LEVELS:
            mask = random_patch_mask(frac, seed)
            perturbed = mask_patches(pil_224, mask, fill_value)
            ptensor, _ = preprocess_pil(perturbed)
            score = tensor_score(model, ptensor, target_index)
            drop = baseline_score - score
            curve.append({
                "fraction": frac,
                "score": round(score, 6),
                "drop": round(drop, 6),
                "pct_drop": round(100.0 * drop / abs(baseline_score) if baseline_score != 0 else 0.0, 4),
            })
        random_curves.append({"seed": seed, "curve": curve})
    result["random_curves"] = random_curves

    # Aggregate random: mean drop per fraction across seeds
    agg_random = []
    for frac in DELETION_LEVELS:
        drops = [rc["curve"][i]["drop"] for i, rc in enumerate(random_curves) for _ in [0] if rc["curve"][i]["fraction"] == frac]
        # simpler: index by position
        drops = [rc["curve"][DELETION_LEVELS.index(frac)]["drop"] for rc in random_curves]
        agg_random.append({
            "fraction": frac,
            "mean_drop": round(float(np.mean(drops)), 6),
            "std_drop": round(float(np.std(drops)), 6),
        })
    result["aggregate_random"] = agg_random

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global N_PATHOLOGIES
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    N_PATHOLOGIES = len(getattr(model, "pathologies", []))
    print(f"Model loaded. Pathologies: {N_PATHOLOGIES}")

    # Select images:
    # Square high border: 1028_0.png (0.6742), 1021_0.png (0.5390)
    # Square low border: 1052_0.png (0.0734), 1006_0.png (0.1599)
    # Non-square X-ray: chest_xray_kaggle_normal.jpeg (2090x1858)
    # Non-square natural: one coco image
    test_images = [
        ("square_high_border_1028", ROOT / "outputs" / "xray_labeled_evaluation" / "images" / "1028_0.png"),
        ("square_high_border_1021", ROOT / "outputs" / "xray_labeled_evaluation" / "images" / "1021_0.png"),
        ("square_low_border_1052", ROOT / "outputs" / "xray_labeled_evaluation" / "images" / "1052_0.png"),
        ("square_low_border_1006", ROOT / "outputs" / "xray_labeled_evaluation" / "images" / "1006_0.png"),
        ("non_square_xray_kaggle", ROOT / "outputs" / "first_inference" / "chest_xray_kaggle_normal.jpeg"),
        ("non_square_coco_000133", ROOT / "outputs" / "xray_verifier_dataset" / "images" / "coco" / "000000000133.jpg"),
    ]

    all_results = []
    for label, path in test_images:
        print(f"\n=== {label} ({path.name}) ===")
        try:
            pil_image = Image.open(path)
            r = evaluate_image(model, pil_image, label)
            all_results.append(r)
            print(f"  Target: {r['target_finding']} (idx {r['target_index']}), baseline={r['baseline_score']:.4f}")
            print(f"  Top-CAM drops: {[c['drop'] for c in r['top_cam_curve']]}")
            print(f"  Random mean drops: {[c['mean_drop'] for c in r['aggregate_random']]}")
            print(f"  Low-CAM drops: {[c['drop'] for c in r['low_cam_curve']]}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save JSON report
    report = {
        "patch_grid": PATCH_GRID,
        "patch_size": PATCH_SIZE,
        "deletion_levels": DELETION_LEVELS,
        "random_seeds": RANDOM_SEEDS,
        "fill_method": "mean_intensity_of_preprocessed_image",
        "target_selection": "original_argmax_fixed",
        "preprocessing": "prepare_image unchanged (center-crop + resize + normalization)",
        "images": all_results,
    }
    report_path = OUT / "faithfulness_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nJSON report saved to {report_path}")

    # Markdown report
    md = build_markdown(report)
    md_path = OUT / "faithfulness_report.md"
    with open(md_path, "w") as fh:
        fh.write(md)
    print(f"Markdown report saved to {md_path}")

    # Summary
    print("\n===== SUMMARY =====")
    for r in all_results:
        top_50 = r["top_cam_curve"][-1]["drop"]
        rand_50 = r["aggregate_random"][-1]["mean_drop"]
        low_50 = r["low_cam_curve"][-1]["drop"]
        print(f"{r['label']}: target={r['target_finding']} baseline={r['baseline_score']:.4f} | "
              f"top50_drop={top_50:+.4f} rand50_drop={rand_50:+.4f} low50_drop={low_50:+.4f} | "
              f"top>rand={top_50 > rand_50}")


def build_markdown(report: dict) -> str:
    lines = []
    lines.append("# Grad-CAM Faithfulness Evaluation Report")
    lines.append("")
    lines.append("## Experiment Design")
    lines.append("")
    lines.append("- **Purpose:** Evaluate whether Grad-CAM highlights regions that materially influence the model's predicted score.")
    lines.append("- **Model faithfulness vs. clinical correctness:** This experiment measures whether removing high-activation regions changes the model's output score. It does NOT assess whether the highlighted regions correspond to actual pathology.")
    lines.append(f"- **Patch grid:** {report['patch_grid']}x{report['patch_grid']} patches of {report['patch_size']}x{report['patch_size']} pixels on the 224x224 CAM.")
    lines.append(f"- **Deletion levels:** {report['deletion_levels']}")
    lines.append(f"- **Random seeds:** {report['random_seeds']}")
    lines.append(f"- **Masked pixel replacement:** {report['fill_method']}")
    lines.append(f"- **Target class:** {report['target_selection']} (fixed throughout, never re-selected after perturbation).")
    lines.append(f"- **Preprocessing:** {report['preprocessing']} (unchanged from production `imaging.py`).")
    lines.append("- **Scores:** Raw model outputs, not calibrated probabilities.")
    lines.append("")
    lines.append("## Controls")
    lines.append("- Same model and input tensor for all conditions.")
    lines.append("- Target class fixed at the original argmax.")
    lines.append("- Original preprocessing preserved.")
    lines.append("- Baseline score recorded before perturbation.")
    lines.append("- Multiple random seeds for the random baseline.")
    lines.append("")
    lines.append("## Per-Image Results")
    lines.append("")
    for r in report["images"]:
        lines.append(f"### {r['label']}")
        lines.append(f"- Original size: {r['original_width']}x{r['original_height']} (ratio {r['original_aspect_ratio']})")
        lines.append(f"- Processed size: {r['processed_width']}x{r['processed_height']}")
        lines.append(f"- Target: {r['target_finding']} (index {r['target_index']})")
        lines.append(f"- Baseline score: {r['baseline_score']:.4f}")
        lines.append(f"- CAM range: [{r['cam_min']:.6f}, {r['cam_max']:.6f}], mean {r['cam_mean']:.6f}")
        lines.append("")
        lines.append("| Fraction | Top-CAM score | Top-CAM drop | Random mean drop | Random std | Low-CAM score | Low-CAM drop |")
        lines.append("|----------|---------------|--------------|------------------|------------|---------------|--------------|")
        for i, frac in enumerate(report["deletion_levels"]):
            tc = r["top_cam_curve"][i]
            ar = r["aggregate_random"][i]
            lc = r["low_cam_curve"][i]
            lines.append(f"| {frac:.0%} | {tc['score']:.4f} | {tc['drop']:+.4f} | {ar['mean_drop']:+.4f} | {ar['std_drop']:.4f} | {lc['score']:.4f} | {lc['drop']:+.4f} |")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- If top-CAM deletion reduces the target score more than random deletion, Grad-CAM is highlighting regions the model relies on (faithfulness).")
    lines.append("- If low-CAM deletion has little effect while top-CAM deletion has a large effect, the CAM is discriminative of influential regions.")
    lines.append("- This measures model faithfulness only; it does not establish clinical localization accuracy.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("- Small sample of images; results may not generalize.")
    lines.append("- CAM is interpolated from a 7x7 activation grid to 224x224; patch boundaries do not align with true activation granularity.")
    lines.append("- Masking with mean intensity introduces out-of-distribution pixels; score changes may partly reflect distribution shift rather than region importance.")
    lines.append("- Scores are raw model outputs, not calibrated probabilities.")
    lines.append("- NIH ChestX-ray14 labels are weak (NLP-derived); target finding may not correspond to true pathology.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
