"""Grad-CAM preprocessing audit: compare center-crop+resize vs direct resize vs pad-to-square."""
from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import torchxrayvision as xrv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imaging import prepare_image, gradcam_layer, gradcam  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "gradcam_preprocessing_audit"
OUT.mkdir(parents=True, exist_ok=True)

BORDER_WIDTH = 16
CENTRAL_MARGIN = 0.25  # central 50% region


def border_mask(shape, margin=16):
    h, w = shape
    m = np.zeros((h, w), dtype=bool)
    if margin <= 0:
        return m
    margin = min(margin, min(h, w) // 2)
    m[:margin, :] = True
    m[-margin:, :] = True
    m[:, :margin] = True
    m[:, -margin:] = True
    return m


def central_mask(shape, margin_frac=0.25):
    h, w = shape
    m = np.zeros((h, w), dtype=bool)
    y0, y1 = int(h * margin_frac), int(h * (1 - margin_frac))
    x0, x1 = int(w * margin_frac), int(w * (1 - margin_frac))
    m[y0:y1, x0:x1] = True
    return m


def mass_fraction(cam, mask):
    cam = np.asarray(cam, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    total = float(cam.sum())
    if total <= 0 or cam.shape != mask.shape:
        return 0.0
    return float(cam[mask].sum() / total)


def prep_center_crop_resize(pil_image):
    """Variant A: original prepare_image (center crop to square, then resize 224)."""
    oriented = ImageOps.exif_transpose(pil_image)
    image = oriented.convert("L")
    width, height = image.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    cropped = image.crop((left, top, left + side, top + side))
    resized = cropped.resize((224, 224), Image.Resampling.BILINEAR)
    return resized


def prep_direct_resize(pil_image):
    """Variant B: current prepare_image (direct resize 224, no crop)."""
    oriented = ImageOps.exif_transpose(pil_image)
    image = oriented.convert("L")
    resized = image.resize((224, 224), Image.Resampling.BILINEAR)
    return resized


def prep_pad_resize(pil_image):
    """Variant C: aspect-preserving resize, pad to square, then resize 224."""
    oriented = ImageOps.exif_transpose(pil_image)
    image = oriented.convert("L")
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("L", (side, side), color=0)
    left, top = (side - width) // 2, (side - height) // 2
    canvas.paste(image, (left, top))
    resized = canvas.resize((224, 224), Image.Resampling.BILINEAR)
    return resized


def tensor_from_pil(pil_image):
    pixels = np.asarray(pil_image, dtype=np.float32)
    normalized = (pixels / 255.0 - 0.5) * 2048.0
    return torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)


def compute_gradcam(model, input_tensor, target_index, layer_name="denseblock4"):
    return gradcam(model, input_tensor, target_index, layer_name=layer_name)


def load_model():
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    return model


def analyze_image(model, pil_image, label):
    results = {"label": label}

    orig_w, orig_h = pil_image.size
    results["original_width"] = orig_w
    results["original_height"] = orig_h
    results["original_aspect_ratio"] = round(orig_w / orig_h, 6)
    results["original_mode"] = pil_image.mode

    # Grayscale conversion check
    gray = ImageOps.exif_transpose(pil_image).convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    results["grayscale_min"] = float(arr.min())
    results["grayscale_max"] = float(arr.max())
    results["grayscale_mean"] = float(arr.mean())

    variants = {
        "A_center_crop_resize": prep_center_crop_resize(pil_image),
        "B_direct_resize": prep_direct_resize(pil_image),
        "C_pad_resize": prep_pad_resize(pil_image),
    }

    variants_results = {}
    for vname, vimg in variants.items():
        # Verify processed dims
        vw, vh = vimg.size
        varr = np.asarray(vimg, dtype=np.float32)

        # Tensor
        tensor = tensor_from_pil(vimg)

        # Model forward + Grad-CAM
        with torch.inference_mode():
            output = model(tensor)
        scores = output[0].detach().cpu().numpy()
        top_index = int(np.argmax(scores))
        top_finding = list(getattr(model, "pathologies", []))[top_index]
        top_score = float(scores[top_index])

        # Grad-CAM
        cam, cam_score = compute_gradcam(model, tensor, top_index)

        # Mass fractions
        bmask = border_mask(cam.shape, BORDER_WIDTH)
        cmask = central_mask(cam.shape, CENTRAL_MARGIN)
        border_mass = mass_fraction(cam, bmask)
        central_mass = mass_fraction(cam, cmask)

        # Raw CAM stats
        cam_stats = {
            "min": float(cam.min()),
            "max": float(cam.max()),
            "mean": float(cam.mean()),
            "sum": float(cam.sum()),
            "std": float(cam.std()),
        }

        variants_results[vname] = {
            "processed_width": vw,
            "processed_height": vh,
            "processed_min": float(varr.min()),
            "processed_max": float(varr.max()),
            "processed_mean": float(varr.mean()),
            "top_index": top_index,
            "top_finding": top_finding,
            "top_score": top_score,
            "cam_score": cam_score,
            "raw_cam": cam_stats,
            "border_mass_16px": border_mass,
            "central_mass_25pct": central_mass,
            "border_minus_central": round(border_mass - central_mass, 6),
            "edge_cell_mass": mass_fraction(cam, border_mask(cam.shape, 1)),  # 1px outer ring
        }

    results["variants"] = variants_results

    # Compare A vs B
    a = variants_results["A_center_crop_resize"]
    b = variants_results["B_direct_resize"]
    results["comparison_A_vs_B"] = {
        "border_mass_diff_B_minus_A": round(b["border_mass_16px"] - a["border_mass_16px"], 6),
        "central_mass_diff_B_minus_A": round(b["central_mass_25pct"] - a["central_mass_25pct"], 6),
        "top_score_diff_B_minus_A": round(b["top_score"] - a["top_score"], 6),
        "same_top_class": a["top_index"] == b["top_index"],
        "cam_max_diff_B_minus_A": round(b["raw_cam"]["max"] - a["raw_cam"]["max"], 8),
    }

    # Compare A vs C
    c = variants_results["C_pad_resize"]
    results["comparison_A_vs_C"] = {
        "border_mass_diff_C_minus_A": round(c["border_mass_16px"] - a["border_mass_16px"], 6),
        "central_mass_diff_C_minus_A": round(c["central_mass_25pct"] - a["central_mass_25pct"], 6),
        "top_score_diff_C_minus_A": round(c["top_score"] - a["top_score"], 6),
        "same_top_class": a["top_index"] == c["top_index"],
    }

    return results


def main():
    model = load_model()
    print("Model loaded. Pathologies:", len(getattr(model, "pathologies", [])))

    # Select 5 square X-rays from labeled evaluation set
    eval_dir = ROOT / "outputs" / "xray_labeled_evaluation" / "images"
    square_files = sorted(list(eval_dir.glob("*.png")))[:5]

    # Select 5 non-square images: 4 coco + 1 kaggle X-ray
    coco_dir = ROOT / "outputs" / "xray_verifier_dataset" / "images" / "coco"
    coco_files = sorted(list(coco_dir.glob("*.jpg")))[:4]
    kaggle = ROOT / "outputs" / "first_inference" / "chest_xray_kaggle_normal.jpeg"

    test_images = []
    for f in square_files:
        img = Image.open(f)
        test_images.append(("square_xray", f.name, img))
    for f in coco_files:
        img = Image.open(f)
        test_images.append(("non_square_coco", f.name, img))
    img = Image.open(kaggle)
    test_images.append(("non_square_xray", kaggle.name, img))

    all_results = []
    for kind, name, pil_image in test_images:
        print(f"\n=== {kind}: {name} ({pil_image.size[0]}x{pil_image.size[1]}, {pil_image.mode}) ===")
        try:
            r = analyze_image(model, pil_image, name)
            all_results.append(r)
            for vname, v in r["variants"].items():
                print(f"  {vname}: border={v['border_mass_16px']:.4f} central={v['central_mass_25pct']:.4f} "
                      f"class={v['top_finding']} score={v['top_score']:.4f}")
            print(f"  A->B border diff: {r['comparison_A_vs_B']['border_mass_diff_B_minus_A']:+.4f} "
                  f"(B-A), same_class={r['comparison_A_vs_B']['same_top_class']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    report = {
        "border_width_px": BORDER_WIDTH,
        "central_margin_fraction": CENTRAL_MARGIN,
        "images": all_results,
    }
    report_path = OUT / "preprocessing_audit_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nReport written to {report_path}")

    # Summary
    print("\n===== SUMMARY =====")
    for r in all_results:
        label = r["label"]
        comp = r["comparison_A_vs_B"]
        print(f"{label}: orig={r['original_width']}x{r['original_height']} "
              f"ratio={r['original_aspect_ratio']:.4f} | "
              f"A border={r['variants']['A_center_crop_resize']['border_mass_16px']:.4f} "
              f"B border={r['variants']['B_direct_resize']['border_mass_16px']:.4f} "
              f"C border={r['variants']['C_pad_resize']['border_mass_16px']:.4f} | "
              f"B-A={comp['border_mass_diff_B_minus_A']:+.4f} same_class={comp['same_top_class']}")


if __name__ == "__main__":
    main()
