import json
from pathlib import Path

data = json.loads(Path("outputs/xray_gradcam/outlier_investigation/outlier_investigation.json").read_text())
records = data["outliers"]
comparison = data["comparison"]
lines = ["# Grad-CAM Outlier Investigation", "", "Outliers: 1021_0, 1028_0, 1063_0, 1106_0", ""]
for r in records:
    m = r["gradcam_mass"]
    lines.extend([
        f"## {Path(r['image']).stem}",
        "",
        f"- Original size: {r['original_size'][0]} x {r['original_size'][1]}; cropped size: {r['cropped_size'][0]} x {r['cropped_size'][1]}",
        f"- Selected label: {r['selected_label']}; selected score: {r['selected_score']:.5f}; Grad-CAM target index: {r['gradcam_target_index']}",
        f"- Lung mask area fraction: {r['lung_mask']['mask_area_fraction']:.4f}",
        f"- Lung mask component count: {r['lung_mask']['component_count']}",
        f"- Lung mask largest components: {r['lung_mask']['largest_components']}",
        f"- Lung mask touches border: {r['lung_mask']['touches_border']}",
        f"- Grad-CAM mass inside lung: {m['inside_lung']:.4f}",
        f"- Grad-CAM mass outside lung: {m['outside_lung']:.4f}",
        f"- Grad-CAM mass in 8/16/24-pixel border: {m['border_8']:.4f} / {m['border_16']:.4f} / {m['border_24']:.4f}",
        f"- Grad-CAM mass in border and lung: {m['border_and_lung']:.4f}",
        f"- Grad-CAM mass in border and non-lung: {m['border_and_non_lung']:.4f}",
        f"- Grad-CAM maximum location (y, x): {m['max_location_yx']}",
        f"- Quadrant fractions: top-left {m['quadrant_fractions']['top_left']:.4f}, top-right {m['quadrant_fractions']['top_right']:.4f}, bottom-left {m['quadrant_fractions']['bottom_left']:.4f}, bottom-right {m['quadrant_fractions']['bottom_right']:.4f}",
        "",
        "Occlusion results:",
        "",
        "| Method | Region | Score delta | Top prediction changed |",
        "|---|---|---:|---|",
    ])
    for method in ("zero", "local_mean", "gaussian_blur"):
        for region in ("lung", "border"):
            info = r["occlusion"][f"{method}_{region}"]
            lines.append(f"| {method} | {region} | {info['selected_score_delta']:.5f} | {info['top_prediction_changed']} |")
    lines.append("")
lines.extend([
    "## Comparison with the other eight images",
    "",
    f"- Outliers: mean lung mass {comparison['outliers']['mean_lung_mass_fraction']:.4f}, mean border-16 mass {comparison['outliers']['mean_border_16_fraction']:.4f}, mean lung mask area {comparison['outliers']['mean_lung_mask_area_fraction']:.4f}.",
    f"- Other eight: mean lung mass {comparison['others']['mean_lung_mass_fraction']:.4f}, mean border-16 mass {comparison['others']['mean_border_16_fraction']:.4f}, mean lung mask area {comparison['others']['mean_lung_mask_area_fraction']:.4f}.",
    "",
    "## Interpretation",
    "",
    "The outlier cases have very low estimated lung mass fraction and, in three cases, a large border mass fraction. This is consistent with Grad-CAM concentration outside the estimated lung region. The lung masks appear non-empty and aligned with the 224x224 classifier input; no empty mask or obvious preprocessing defect was found. The differences are more likely due to image content and model behavior than to a confirmed implementation defect.",
    "",
    "No clinical significance is asserted. Grad-CAM remains a research visualization.",
])
Path("outputs/xray_gradcam/outlier_investigation/outlier_investigation.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote report")
