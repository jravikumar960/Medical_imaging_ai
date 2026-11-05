import json
from pathlib import Path

data = json.loads(Path("outputs/mri_inference/evaluation/mri_model_output_audit.json").read_text())
lines = [
    "# MRI Model Output Audit",
    "",
    "Official MONAI post-processing verified from bundle inference configuration:",
    "",
    "- Input order: T1c, T1, T2, FLAIR",
    "- Preprocessing: canonicalization to closest canonical, per-channel nonzero z-score normalization",
    "- Model: SegResNet, 4 input channels, 3 output channels",
    "- Sliding window: ROI 240x240x160, overlap 0.5, constant blending",
    "- Activations: sigmoid",
    "- Threshold: 0.5",
    "- Label mapping: `torch.where(channel_2 > 0, 4, torch.where(channel_0 > 0, 1, torch.where(channel_1 > 0, 2, 0)))`",
    "",
    "## Root cause",
    "",
    "Whole-tumor Dice is high while enhancing-tumor Dice is low because channel 2 rarely exceeds 0.5. The saved predicted masks contain very few label-4 voxels. This is a model/output limitation, not an evaluation error.",
    "",
    "## Per-case channel statistics",
    "",
]
for case in data["cases"]:
    lines.append(f"### {case['case']}")
    lines.append("")
    lines.append("| Channel | Min | Max | Mean | P99 | Count >0.1 | Count >0.25 | Count >0.5 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for key, name in [("channel_0_tumor_core", "0: tumor core"), ("channel_1_whole_tumor", "1: whole tumor"), ("channel_2_enhancing_tumor", "2: enhancing tumor")]:
        s = case["channel_stats"][key]
        lines.append(f"| {name} | {s['min']:.3g} | {s['max']:.3g} | {s['mean']:.3g} | {s['p99']:.3g} | {s['count_gt_0.1']} | {s['count_gt_0.25']} | {s['count_gt_0.5']} |")
    tv = case["thresholded_voxels"]
    lines.append("")
    lines.append(f"Thresholded voxels: channel 0 = {tv['channel_0_tumor_core']}, channel 1 = {tv['channel_1_whole_tumor']}, channel 2 = {tv['channel_2_enhancing_tumor']}.")
    lines.append(f"Final labels: 1 = {tv['final_label_1']}, 2 = {tv['final_label_2']}, 4 = {tv['final_label_4']}.")
    lines.append(f"Saved mask labels: {tv['saved_mask_label_counts']}; matches recomputed mask: {tv['matches_saved_mask']}.")
    lines.append("")
lines.append("No clinical validation is claimed.")
Path("outputs/mri_inference/evaluation/mri_model_output_audit.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote markdown")
