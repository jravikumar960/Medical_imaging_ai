# MONAI BraTS MRI segmentation checkpoint

- Bundle: `brats_mri_segmentation` version 0.5.4
- Model: MONAI `SegResNet`, trained for 3D primary-glioma subregion segmentation.
- Training source: BraTS 2018 Challenge; its data usage agreement requires citation of the BraTS benchmark manuscripts.
- Input: four co-registered, skull-stripped 1 mm isotropic NIfTI volumes, ordered T1c, T1, T2, FLAIR.
- Output: thresholded channels for tumor core, whole tumor, and enhancing tumor, written as BraTS labels 1, 2, and 4.
- Model artifact: public `MONAI/brats_mri_segmentation` Hugging Face repository, `models/model.pt`.
- Artifact SHA-256: `860ccb3f1c21c99d0410ad8a1ac4ef6b8fab60cec0a503b0ba42675741a750ae`
- Artifact size: 18,840,620 bytes.
- Bundle software license: Apache-2.0; see [LICENSE](./LICENSE).
- MONAI bundle metadata reports validation mean Dice 0.8518 (TC 0.8559, WT 0.9026, ET 0.7905) on its own BraTS 2018 validation partition. These source-reported values are not results from this application and do not establish independent performance or clinical validity.

The bundle documentation describes research use and explicitly says it is not for diagnosis. This application only supports the documented BraTS-style task; it does not validate acquisition sequences, co-registration, skull stripping, or diagnosis. It must not be used for patient-care decisions. No MRI dataset or ground-truth volume is included in this project.

Sources:

- [MONAI bundle documentation](https://github.com/Project-MONAI/model-zoo/tree/dev/models/brats_mri_segmentation)
- [Published checkpoint metadata](https://huggingface.co/api/models/MONAI/brats_mri_segmentation/tree/main/models?expand=true)
- [BraTS 2018 data terms and citation requirements](https://github.com/Project-MONAI/model-zoo/blob/dev/models/brats_mri_segmentation/docs/data_license.txt)
- [MONAI model card](https://huggingface.co/MONAI/brats_mri_segmentation)
