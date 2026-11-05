import nibabel as nib
import numpy as np
from pathlib import Path

base = Path("outputs/ct_inference/kits_0108")
ct = nib.load(str(base / "kits_0108.nii.gz"))
mask = nib.load(str(base / "kits_0108_mask.nii.gz"))
print("ct shape", ct.shape, "dtype", ct.get_data_dtype())
print("mask shape", mask.shape, "dtype", mask.get_data_dtype())
print("affine equal", np.allclose(ct.affine, mask.affine))
print("foreground voxels", int(np.count_nonzero(mask.get_fdata() > 0.5)))
print("mask unique", np.unique(mask.get_fdata())[:10])
print("ct finite", bool(np.isfinite(ct.get_fdata()).all()))
