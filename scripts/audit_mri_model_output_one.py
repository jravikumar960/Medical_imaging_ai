import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import mri_segmentation as ms

ROOT = Path("outputs/mri_inference")
OUT = ROOT / "evaluation"
OUT.mkdir(parents=True, exist_ok=True)

case = sys.argv[1]
cdir = ROOT / case
paths = {
    "t1c": cdir / f"{case}-t1c.nii.gz",
    "t1": cdir / f"{case}-t1n.nii.gz",
    "t2": cdir / f"{case}-t2w.nii.gz",
    "flair": cdir / f"{case}-t2f.nii.gz",
}

flair_image, tensor = ms._validated_inputs(paths)
roi = ms._roi_for_shape(tuple(int(v) for v in tensor.shape[2:]))
inferer = ms._modules()[1](roi_size=roi, sw_batch_size=1, overlap=0.5, mode="constant")
model = ms.load_model()

with torch.inference_mode():
    output = inferer(tensor, model)

probs = torch.sigmoid(output[0]).cpu().numpy()
tc, wt, et = probs
tc_mask = tc >= 0.5
wt_mask = wt >= 0.5
et_mask = et >= 0.5
labels = np.zeros(tc.shape, dtype=np.uint8)
labels[wt_mask] = 2
labels[tc_mask] = 1
labels[et_mask] = 4

import nibabel as nib

saved_mask = np.asarray(nib.load(str(cdir / f"{case}_mask.nii.gz")).get_fdata())

def stats(p):
    return {
        "min": float(p.min()),
        "max": float(p.max()),
        "mean": float(p.mean()),
        "p50": float(np.percentile(p, 50)),
        "p90": float(np.percentile(p, 90)),
        "p99": float(np.percentile(p, 99)),
        "p999": float(np.percentile(p, 99.9)),
        "count_gt_0.1": int((p > 0.1).sum()),
        "count_gt_0.25": int((p > 0.25).sum()),
        "count_gt_0.5": int((p > 0.5).sum()),
        "fraction_gt_0.5": float((p > 0.5).mean()),
    }

result = {
    "case": case,
    "input_shapes": {k: list(nib.load(str(v)).shape) for k, v in paths.items()},
    "roi_size": list(roi),
    "channel_stats": {
        "channel_0_tumor_core": stats(tc),
        "channel_1_whole_tumor": stats(wt),
        "channel_2_enhancing_tumor": stats(et),
    },
    "thresholded_voxels": {
        "channel_0_tumor_core": int(tc_mask.sum()),
        "channel_1_whole_tumor": int(wt_mask.sum()),
        "channel_2_enhancing_tumor": int(et_mask.sum()),
        "final_label_1": int((labels == 1).sum()),
        "final_label_2": int((labels == 2).sum()),
        "final_label_4": int((labels == 4).sum()),
        "saved_mask_label_counts": {str(int(x)): int((saved_mask == x).sum()) for x in sorted(set(int(v) for v in np.unique(saved_mask)))},
        "matches_saved_mask": bool(np.array_equal(labels, saved_mask)),
    },
}
(OUT / f"mri_model_output_audit_{case}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print("wrote", case)
