# Local MONAI spleen checkpoint

The local `models/model.pt` file is an unmodified copy of the official MONAI
`spleen_ct_segmentation` v0.6.1 checkpoint from
<https://huggingface.co/MONAI/spleen_ct_segmentation>.

- File size: 19,297,197 bytes
- SHA-256: `502c312893994ec071a85c4f2e7d83a43f7789969faa73dae62f1291177f50fe`
- Official metadata: MONAI 1.4.0, PyTorch 2.4.0, NumPy 1.24.4, nibabel 5.2.1
- State dict: 148 tensors; strictly matched to the MONAI 3D U-Net architecture
- Software license: Apache-2.0; see the official bundle `LICENSE`
- Training data: MSD Task09 Spleen; see official bundle `docs/data_license.txt`

This project uses PyTorch 2.4.1+cpu in `.venv-seg` due to a Windows DLL import
failure with the official 2.4.0 wheel. The copied checkpoint was loaded with
`torch.load(weights_only=True)` and `strict=True`; its SHA-256 and size match
the official Hugging Face file metadata. The separately provided `model.ts`
was not downloaded or executed. The full MONAI bundle configuration remains
maintained upstream; this directory contains the verified checkpoint only.

No authorized CT volume is included. Geometry attestation is still required,
and no segmentation or clinical validation is claimed.
