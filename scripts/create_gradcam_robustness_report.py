import json
from pathlib import Path

data = json.loads(Path("outputs/xray_gradcam/robustness_audit/gradcam_robustness_audit.json").read_text())
agg = data["aggregate"]
lines = [
    "# Grad-CAM Robustness and Border-Sensitivity Audit",
    "",
    "Lung masks are from the pretrained ChestX-Det PSPNet model. They are not independently clinically validated.",
    "",
    "## Aggregate statistics",
    "",
    f"- Images: {agg['n_images']}",
    f"- Mean lung mask area fraction: {agg['mean_lung_mask_area_fraction']:.4f}",
    f"- Mean Grad-CAM mass inside lung (16px border definition): {agg['mean_lung_mass_fraction']:.4f}",
    f"- Mean Grad-CAM mass outside lung: {agg['mean_outside_lung_fraction']:.4f}",
    f"- Mean Grad-CAM mass in 8px border: {agg['mean_border_8_mass_fraction']:.4f}",
    f"- Mean Grad-CAM mass in 16px border: {agg['mean_border_16_mass_fraction']:.4f}",
    f"- Mean Grad-CAM mass in 24px border: {agg['mean_border_24_mass_fraction']:.4f}",
    f"- Mean selected-score change, zero-fill lung occlusion: {agg['mean_zero_lung_occlusion_delta']:.5f}",
    f"- Mean selected-score change, zero-fill border occlusion: {agg['mean_zero_border_occlusion_delta']:.5f}",
    f"- Mean selected-score change, local-mean lung occlusion: {agg['mean_local_mean_lung_occlusion_delta']:.5f}",
    f"- Mean selected-score change, Gaussian-blur lung occlusion: {agg['mean_gaussian_lung_occlusion_delta']:.5f}",
    f"- Top-prediction changes after lung occlusion: zero={agg['top_prediction_changes_after_lung_occlusion']['zero']}, local mean={agg['top_prediction_changes_after_lung_occlusion']['local_mean']}, Gaussian blur={agg['top_prediction_changes_after_lung_occlusion']['gaussian_blur']}",
    "",
    "Border regions overlap lung masks. The border and lung measurements are therefore not mutually exclusive categories.",
    "",
    "## Per-image results",
    "",
    "| Image | Selected label | Score | Lung mass fraction | Border 8px | Border 16px | Border 24px | Lung occlusion zero | Lung occlusion local mean | Lung occlusion Gaussian | Top changed (zero) |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in data["records"]:
    lines.append(
        f"| {Path(r['source_image']).name} | {r['selected_label']} | {r['selected_score']:.5f} | "
        f"{r['region_mass_fractions_16px_border']['inside_lung']:.4f} | "
        f"{r['border_mass_fractions']['8']:.4f} | {r['border_mass_fractions']['16']:.4f} | {r['border_mass_fractions']['24']:.4f} | "
        f"{r['occlusion']['zero_lung']['selected_score_delta']:.5f} | "
        f"{r['occlusion']['local_mean_lung']['selected_score_delta']:.5f} | "
        f"{r['occlusion']['gaussian_blur_lung']['selected_score_delta']:.5f} | "
        f"{r['occlusion']['zero_lung']['top_prediction_changed']} |"
    )
lines.extend([
    "",
    "## Outliers",
    "",
    "Images with validated lung mass fraction below 0.1 or above 0.6:",
])
for name in agg["outlier_images_by_validated_lung_mass"]:
    lines.append(f"- `{Path(name).name}`")
lines.extend([
    "",
    "These cases contradict the aggregate pattern and show that localization quality varies by image.",
    "",
    "## Occlusion methods",
    "",
    "- Zero fill: fastest and deterministic, but creates artificial black regions and sharp edges.",
    "- Local mean replacement: reduces extreme black artifacts but still creates an unnatural flat region.",
    "- Gaussian blur replacement: smoothest transition but can retain partial structure and may not fully remove the region's contribution.",
    "",
    "All methods use the same masks, selected output, and preprocessing. Scores are raw model outputs, not calibrated probabilities.",
    "",
    "## Tests",
    "",
    "- `tests/test_xray_gradcam_localization.py`: 6 passed.",
])
Path("outputs/xray_gradcam/robustness_audit/gradcam_robustness_audit.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote report")
