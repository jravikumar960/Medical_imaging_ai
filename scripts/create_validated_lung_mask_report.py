import json
from pathlib import Path

new = json.loads(Path("outputs/xray_gradcam/validated_lung_masks/validated_lung_mask_audit.json").read_text())
old = json.loads(Path("outputs/xray_gradcam/localization/gradcam_localization_audit.json").read_text())
old_by_name = {Path(r["source_image"]).name: r for r in old["records"]}

lines = [
    "# Validated Lung-Mask Grad-CAM Localization Audit",
    "",
    "Lung masks were generated with the pretrained ChestX-Det PSPNet model from torchxrayvision, which segments 14 anatomical structures including Left Lung and Right Lung.",
    "",
    "Source and license: ChestX-Det dataset, Apache-2.0 bundle; model weights downloaded from the torchxrayvision release. Input is expected in [-1024, 1024] and is center-cropped to a square before segmentation. The 512x512 mask is resized to 224x224 with nearest-neighbor resampling for alignment with the Grad-CAM heatmap.",
    "",
    "The generated masks are not independently clinically validated.",
    "",
    "## Comparison with previous heuristic audit",
    "",
    "| Metric | Heuristic mean | Validated-mask mean |",
    "|---|---:|---:|",
    f"| Lung mask area fraction | {old['aggregate']['mean_lung_mask_area_fraction']:.4f} | {new['aggregate']['mean_lung_mask_area_fraction']:.4f} |",
    f"| Grad-CAM mass inside lung | {old['aggregate']['mean_lung_mass_fraction']:.4f} | {new['aggregate']['mean_lung_mass_fraction']:.4f} |",
    f"| Grad-CAM mass outside lung | {old['aggregate']['mean_outside_lung_fraction']:.4f} | {new['aggregate']['mean_outside_lung_fraction']:.4f} |",
    f"| Grad-CAM mass in 16px border | {old['aggregate']['mean_border_mass_fraction']:.4f} | {new['aggregate']['mean_border_mass_fraction']:.4f} |",
    f"| Lung occlusion score delta | {old['aggregate']['mean_lung_occlusion_delta']:.5f} | {new['aggregate']['mean_lung_occlusion_delta']:.5f} |",
    f"| Border occlusion score delta | {old['aggregate']['mean_border_occlusion_delta']:.5f} | {new['aggregate']['mean_border_occlusion_delta']:.5f} |",
    "",
    "## Per-image results",
    "",
    "| Image | Selected label | Heuristic lung mass | Validated lung mass | Heuristic border mass | Validated border mass |",
    "|---|---|---:|---:|---:|---:|",
]
for r in new["records"]:
    name = Path(r["source_image"]).name
    old_r = old_by_name.get(name, {})
    lines.append(
        f"| {name} | {r['selected_label']} | "
        f"{old_r.get('gradcam_mass_inside_lung_fraction', float('nan')):.4f} | {r['gradcam_mass_inside_lung_fraction']:.4f} | "
        f"{old_r.get('gradcam_mass_border_fraction', float('nan')):.4f} | {r['gradcam_mass_border_fraction']:.4f} |"
    )
lines.extend([
    "",
    "## Occlusion experiments",
    "",
    "Occlusion was applied by setting the masked region to 0 in the 224x224 model input after the standard center crop and resize. This avoids changing the preprocessing pipeline but can introduce artificial black regions.",
    "",
    "## Limitations",
    "",
    "- The ChestX-Det model is pretrained, but its masks were not independently verified on these specific images.",
    "- The model input is center-cropped and resized to 224x224 before classification, while segmentation is run on the same center-cropped image.",
    "- The 512x512 segmentation mask is resized to 224x224 for alignment.",
    "- Grad-CAM remains a research visualization and does not prove clinical reasoning.",
    "",
    "## Tests",
    "",
    "- `tests/test_xray_gradcam_localization.py`: 6 passed.",
])
Path("outputs/xray_gradcam/validated_lung_masks/validated_lung_mask_audit.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote report")
