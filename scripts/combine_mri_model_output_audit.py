import json
from pathlib import Path

OUT = Path("outputs/mri_inference/evaluation")
cases = []
for p in sorted(OUT.glob("mri_model_output_audit_BraTS*.json")):
    cases.append(json.loads(p.read_text()))

payload = {
    "audit_type": "model_output_channel_and_postprocessing_audit",
    "official_postprocessing": {
        "source": "MONAI brats_mri_segmentation bundle inference.json",
        "activations": "sigmoid",
        "threshold": 0.5,
        "label_mapping": "torch.where(channel_2 > 0, 4, torch.where(channel_0 > 0, 1, torch.where(channel_1 > 0, 2, 0)))",
        "channel_0": "tumor core",
        "channel_1": "whole tumor",
        "channel_2": "enhancing tumor",
    },
    "channel_trace": {
        "channel_0": "sigmoid probability >= 0.5 -> tumor_core",
        "channel_1": "sigmoid probability >= 0.5 -> whole_tumor",
        "channel_2": "sigmoid probability >= 0.5 -> enhancing_tumor",
        "postprocessing_order": ["labels[whole_tumor] = 2", "labels[tumor_core] = 1", "labels[enhancing_tumor] = 4"],
        "final_label_1": "tumor_core voxels not assigned to enhancing_tumor",
        "final_label_2": "whole_tumor voxels not assigned to tumor_core or enhancing_tumor",
        "final_label_4": "enhancing_tumor voxels",
    },
    "root_cause": "Whole-tumor Dice is high while enhancing-tumor Dice is low because the saved predicted masks contain very few label-4 voxels. The model's channel 2 rarely exceeds 0.5; this is a model/output limitation, not an evaluation error.",
    "cases": cases,
}
(OUT / "mri_model_output_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("wrote", OUT / "mri_model_output_audit.json")
