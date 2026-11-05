import nibabel as nib
import numpy as np

p = "outputs/mri_inference/BraTS-GLI-00005-100/BraTS-GLI-00005-100_mask.nii.gz"
r = "outputs/mri_inference/BraTS-GLI-00005-100/BraTS-GLI-00005-100-seg.nii.gz"


def dice(a, b):
    inter = np.logical_and(a > 0, b > 0).sum()
    denom = (a > 0).sum() + (b > 0).sum()
    return 2 * inter / denom if denom else 1.0


pm = nib.load(p).get_fdata()
rm = nib.load(r).get_fdata()
print("direct WT", dice(pm, rm), "pred", (pm > 0).sum(), "ref", (rm > 0).sum(), "inter", np.logical_and(pm > 0, rm > 0).sum())

pc = nib.as_closest_canonical(nib.load(p)).get_fdata()
rc = nib.as_closest_canonical(nib.load(r)).get_fdata()
print("canonical WT", dice(pc, rc), "pred", (pc > 0).sum(), "ref", (rc > 0).sum(), "inter", np.logical_and(pc > 0, rc > 0).sum())
