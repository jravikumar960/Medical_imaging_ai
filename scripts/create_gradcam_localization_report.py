import json
from pathlib import Path

data = json.loads(Path("outputs/xray_gradcam/localization/gradcam_localization_audit.json").read_text())
agg = data["aggregate"]
lines = [
    "# X-Ray Grad-CAM Localization Audit",
    "",
    "The lung-region masks used here are an unvalidated intensity/component heuristic. They are not clinically validated lung segmentations.",
    "",
    "## Aggregate results",
    "",
    f"- Images tested: {agg['n_images']}",
    f"- Mean Grad-CAM mass inside estimated lung region: {agg['mean_lung_mass_fraction']:.4f}",
    f"- Mean Grad-CAM mass outside estimated lung region: {agg['mean_outside_lung_fraction']:.4f}",
    f"- Mean Grad-CAM mass in 16-pixel border: {agg['mean_border_mass_fraction']:.4f}",
    f"- Mean estimated lung mask area: {agg['mean_lung_mask_area_fraction']:.4f}",
    f"- Mean selected-score change after occluding estimated lung region: {agg['mean_lung_occlusion_delta']:.5f}",
    f"- Mean selected-score change after occluding border region: {agg['mean_border_occlusion_delta']:.5f}",
    "",
    "## Per-image results",
    "",
    "| Image | Selected label | Score | Lung mass fraction | Border mass fraction | Lung occlusion delta | Border occlusion delta |",
    "|---|---|---:|---:|---:|---:|---:|",
]
for r in data["records"]:
    lines.append(
        f"| {Path(r['source_image']).name} | {r['selected_label']} | {r['selected_score']:.5f} | "
        f"{r['gradcam_mass_inside_lung_fraction']:.4f} | {r['gradcam_mass_border_fraction']:.4f} | "
        f"{r['occlusion']['lung_occlusion_score_delta']:.5f} | {r['occlusion']['border_occlusion_score_delta']:.5f} |"
    )
lines.extend([
    "",
    "## Interpretation",
    "",
    "Grad-CAM mass is not consistently concentrated inside the unvalidated estimated lung region. Several images show substantial border or non-lung mass. Occluding the estimated lung region often changes the selected score more than occluding the border, but the lung mask is only an estimate and the occlusion uses zero filling, which can create artificial edges.",
    "",
    "No clinical interpretability is claimed. Grad-CAM remains a research visualization.",
    "",
    "## Tests",
    "",
    "- `tests/test_xray_gradcam_localization.py`: 4 passed.",
])
Path("outputs/xray_gradcam/localization/localization_audit.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote report")
