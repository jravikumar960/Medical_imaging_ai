# Reproducible CXR Evaluation Protocol

Status: **blocked — do not compute or publish model-performance metrics yet.**

This protocol applies only to the 2D chest X-ray classifier. It does not apply to the separate MONAI spleen CT segmentation workflow.

## Why evaluation is pending

There is no authorized independent labeled CXR cohort in the repository. The configured model is TorchXRayVision DenseNet121 `densenet121-res224-all`. Installed TorchXRayVision 1.5.7 metadata names its training sources as `nih-pc-chex-mimic_ch-google-openi-rsna`. No record-level training manifest or overlap report is available, so source labels do not establish whether particular cases overlap.

In particular:

- NIH, RSNA, and other derivatives of these sources are not an independent test cohort without an auditable case-level exclusion review. The installed TorchXRayVision RSNA adapter describes RSNA as a subset of NIH ChestX-ray14.
- README.md discusses `densenet121-res224-chex` as a possible alternative. It has only 11 non-empty output heads in the installed package metadata, and the current evaluator hardcodes `densenet121-res224-all` and rejects empty output-label metadata. Its checkpoint release rights and case-level overlap are not established here. RSNA Target's copied values for Pneumonia and Lung Opacity are not equivalent by default to CheXpert report labels. Do not switch checkpoints or map labels without a separate review and compatible evaluator changes.
- The existing RSNA preparation script is for schema preparation only. It maps the same RSNA Target to Pneumonia and Lung Opacity, leaves the other 16 outputs unknown, and does not make RSNA independent from checkpoint training.

Do not run evaluation for a performance claim until all prerequisites below are documented. Synthetic test fixtures verify evaluator behavior only and are not model results.

## Prerequisites

1. **Authorized images and labels.** Record dataset source, version, access/usage terms, permitted local use, attribution obligations, de-identification status, annotation source, labeling protocol, and image-to-label linkage. Keep restricted data local and outside source control.
2. **Checkpoint identity and independence.** Record the selected checkpoint key and package version. Establish that the evaluation cases are excluded from training, including through parent datasets and derived datasets. Prefer a checkpoint with an auditable provenance that excludes the entire evaluation cohort. Source names alone are insufficient; if independence remains uncertain, report evaluation as pending or exploratory and do not call it independent.
3. **Compatible label definitions.** For each output, cite the dataset's exact annotation definition and decide whether it supports that output. Use 0 only for an explicit negative under that definition, 1 for an explicit positive, and blank for unknown or incompatible. Do not convert absent annotations or uncertain labels to negatives. The evaluator checks label names, not semantic equivalence.
4. **Patient-level split.** Create train, validation, and test partitions before examining model scores. Use true patient identifiers or an authorized crosswalk, and keep every patient in one split. Record the split algorithm, seed (if randomized), crosswalk source, and split-map hash. If patient identifiers are unavailable, describe the alternative independence evidence and state that the evaluator cannot check patient leakage.
5. **Frozen test set.** Keep test data untouched while selecting thresholds and making implementation decisions. The evaluator cannot prevent repeated test-set inspection; reserve one final test analysis and record any deviations.
6. **Image preparation.** Supply authorized readable PNG/JPEG files with explicit manifest paths. `evaluate.py` uses the shared grayscale, center-crop, resize, and TorchXRayVision normalization path. It does not determine whether a raster depicts a chest X-ray; provenance must establish that.

## Manifest contract

Create one CSV row per image. Required columns, in this order for readability, are:

    image_path,split,patient_id,Atelectasis,Consolidation,Infiltration,Pneumothorax,Edema,Emphysema,Fibrosis,Effusion,Pneumonia,Pleural_Thickening,Cardiomegaly,Nodule,Mass,Hernia,Lung Lesion,Fracture,Lung Opacity,Enlarged Cardiomediastinum

The exact label order is taken from the configured model's `pathologies` metadata. `patient_id` is optional to the evaluator but required for a claim of programmatically checked patient-level separation. If present, it must be non-empty in every row and must not cross splits. Every label cell must be 0, 1, or blank. Splits must be `train`, `validation`, or `test`; both validation and test are required.

Before evaluation, independently confirm:

- every path refers to an authorized, de-identified CXR and the intended image;
- validation/test paths exist under the specified image root;
- no image path is reused across splits and no patient appears in more than one split;
- all labels follow the documented protocol and both classes are available where AUROC/thresholds are expected;
- any source-dataset overlap with model training has been reviewed.

The optional `prepare_rsna_manifest.py` is not an independence solution. It requires an externally prepared split-map CSV with `patientId`, true `patient_id`, and `split`; its `patient_id` must not simply copy the RSNA exam identifier. It accepts authorized local files only and does not download data. Its current two-label mapping is documented above and must not be generalized to other datasets.

## Execution procedure

From PowerShell in the repository root, after all prerequisites are met:

1. Record the exact Git revision if available; this workspace may not contain Git metadata.
2. Record the interpreter and installed packages without modifying the environment:

       .\.venv\Scripts\python.exe --version
       .\.venv\Scripts\python.exe -m pip freeze > .\evaluation_results\environment.txt
       Get-FileHash .\authorized_cxr\manifest.csv -Algorithm SHA256
       Get-FileHash .\authorized_cxr\split-map.csv -Algorithm SHA256

   Create the output directory before redirecting the package list. Keep hashes and protocol records free of patient identifiers.
3. Run the evaluator once on the frozen manifest:

       .\.venv\Scripts\python.exe evaluate.py .\authorized_cxr\manifest.csv --image-root .\authorized_cxr --output-prefix .\evaluation_results\densenet121_res224_all

4. Preserve the JSON and CSV outputs with the report template, environment record, label protocol, split map/provenance, dataset access evidence, and checkpoint-overlap evidence. Do not retune thresholds from test outcomes.

The evaluator excludes train rows from inference, selects a per-label threshold by maximum F1 on validation only (ties select the higher threshold), then calculates held-out test metrics. It reports raw model scores, not calibrated probabilities. Per-finding outputs include labeled/missing/positive/negative counts, AUROC, sensitivity, specificity, precision, recall, F1, confusion counts, and undefined-metric status. AUROC requires both test classes; thresholded measures require an available validation threshold. Null metrics must remain explicitly undefined, not be replaced with zero.

## Required decision before reporting

If checkpoint overlap, patient grouping, dataset authorization, or label compatibility is unresolved, the report must say **evaluation pending** and contain no model-performance values. Even a completed held-out evaluation would be a research evaluation, not clinical validation.

## CT spleen-segmentation evaluation

`evaluate_segmentation.py` compares saved binary prediction and reference NIfTI masks; it does not run inference. Its CSV manifest must include `case_id`, `patient_id`, `split`, `prediction_path`, `reference_path`, `dataset`, `reference_source`, and `label_definition`, with distinct `train`, `validation`, `calibration`, and `test` groups. Patient identifiers must be disjoint across splits. Only `test` masks are scored. The evaluator rejects non-binary masks, mismatched shapes, and affine disagreement above `atol=rtol=1e-5`.

It reports per-volume Dice, intersection over union, sensitivity, precision, voxel-level false positives/negatives, and case-level bootstrap percentile intervals for mean metrics. These are research metrics only. Before interpreting results, independently establish reference-label provenance, exact-volume orientation/provenance requirements, dataset authorization, patient independence, and checkpoint overlap. The current local Task09 Spleen NIfTI lacks exact-volume source-DICOM conversion evidence; it must not be treated as an eligible evaluation case based on its label overlay or matching headers.

No authorized independent CXR evaluation cohort, provenance-qualified CT evaluation cohort, or independent labeled MRI evaluation cohort is currently recorded. A task-specific pretrained BraTS MRI workflow has been added, but no local real MRI inference or performance evaluation has been completed.

## Brain MRI tumor-segmentation workflow

The application uses the MONAI `brats_mri_segmentation` v0.5.4 bundle and its public Apache-2.0 checkpoint (`bundles/brats_mri_segmentation/models/model.pt`). The specific task is 3D primary-glioma subregion segmentation (tumor core, whole-tumor remainder, enhancing tumor) on four aligned, skull-stripped, 1 mm isotropic NIfTI sequences in T1c/T1/T2/FLAIR order. The model is not for diagnosis. Metadata validation cannot verify that a volume is genuinely a required sequence, skull-stripped, correctly registered, or from the intended tumor population.

MONAI bundle documentation reports a BraTS 2018 validation mean Dice of 0.8518 (TC 0.8559, WT 0.9026, ET 0.7905). Those are upstream model-card metrics, not an independent evaluation of this application and must not be reported as local results. This project has no suitable real MRI volume/reference-label cohort: local MRI inference sample count is 0, independently scored test sample count is 0, and local Dice/IoU/sensitivity metrics are unavailable. The model/source license and BraTS citation/use notes are documented in `bundles/brats_mri_segmentation/README.md`.

To conduct a project-level evaluation, obtain a lawfully accessible, de-identified BraTS-compatible cohort with reference subregion labels and patient IDs; document its license, citations, preprocessing, split independence, and checkpoint overlap; reserve patient-disjoint held-out cases; save model output and reference masks in identical physical geometry; then compare TC, WT, and ET separately using Dice, IoU, sensitivity, precision, case counts, and uncertainty intervals. Do not infer on the available Task09 CT dataset as an MRI substitute.
