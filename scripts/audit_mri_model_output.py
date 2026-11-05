import json
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import math
from pathlib import Path

import numpy as np
import torch

import mri_segmentation as ms

ROOT = Path("outputs/mri_inference")
OUT = ROOT / "evaluation"
OUT.mkdir(parents=True, exist_ok=True)

cases = ["BraTS-GLI-00005-100", "BraTS-GLI-00005-101", "BraTS-GLI-00006-100", "BraTS-GLI-00008-100", "BraTS-GLI-00009-100"]
model = ms.load_model()

def channel_stats(prob):
    flat = prob.reshape(-1)
    return {
        "min": float(prob.min()),
        "max": float(prob.max()),
        "mean": float(prob.mean()),
        "p50": float(np.percentile(prob, 50)),
        "p90": float(np.percentile(prob, 90)),
        "p99": float(np.percentile(prob, 99)),
        "p999": float(np.percentile(prob, 99.9)),
        "count_gt_0.1": int((prob > 0.1).sum()),
        "count_gt_0.25": int((prob > 0.25).sum()),
        "count_gt_0.5": int((prob > 0.5).sum()),
        "fraction_gt_0.5": float((prob > 0.5).mean()),
    }

results = []
for case in cases:
    cdir = ROOT / case
    paths = {
        "t1c": cdir / f"{case}-t1c.nii.gz",
        "t1": cdir / f"{case}-t1n.nii.gz",
        "t2": cdir / f"{case}-t2w.nii.gz",
        "flair": cdir / f"{case}-t2f.nii.gz",
    }
    # Some directories may use t1 not t1n? Check.
    for ch in ms.MRI_CHANNELS:
        if not paths[ch].is_file():
            # fallback names
            pass
    flair_image, tensor = ms._validated_inputs(paths)
    inferer = ms._modules()[1](roi_size=ms._roi_for_shape(tuple(int(v) for v in tensor.shape[2:])), sw_batch_size=1, overlap=0.5, mode="constant")
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
    results.append({
        "case": case,
        "input_shapes": {k: list(nib.load(str(v)).shape) for k, v in paths.items()},
        "roi_size": list(ms._roi_for_shape(tuple(int(v) for v in tensor.shape[2:]))),
        "channel_stats": {
            "channel_0_tumor_core": channel_stats(tc),
            "channel_1_whole_tumor": channel_stats(wt),
            "channel_2_enhancing_tumor": channel_stats(et),
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
    })

(OUT / "mri_model_output_audit.json").write_text(json.dumps({"official_postprocessing": "sigmoid then threshold 0.5 then labels: ET=4, TC=1, WT remainder=2", "cases": results}, indent=2), encoding="utf-8")
print("wrote", OUT / "mri_model_output_audit.json")
PY
