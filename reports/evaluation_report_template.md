# CXR Evaluation Report Template

**Status:** PENDING — no evaluation performed until every prerequisite is supported by evidence.

Replace bracketed fields only with documented facts or actual evaluator output. Do not fill metric cells with estimates, synthetic test values, or values from training/validation data.

## Study identity

- Evaluation date: [pending]
- Project revision / source archive hash: [pending]
- Evaluator command: [pending]
- Python version: [pending]
- Dependency record path and SHA-256: [pending]
- Output prefix: [pending]

## Dataset, access, and labels

- Dataset name/version/source: [pending]
- Access authorization and permitted use: [pending]
- License/terms, attribution, redistribution limits: [pending]
- Local data location (do not include sensitive paths in a public report): [pending]
- De-identification and image-to-label linkage evidence: [pending]
- Labeling source/protocol/reviewer process: [pending]
- Unknown/uncertain label handling: [pending]
- Image preprocessing/export details: [pending]

| Model output label | Dataset label/annotation | Definition and source citation | Mapping decision and rationale |
|---|---|---|---|
| Atelectasis | [pending] | [pending] | [pending] |
| Consolidation | [pending] | [pending] | [pending] |
| Infiltration | [pending] | [pending] | [pending] |
| Pneumothorax | [pending] | [pending] | [pending] |
| Edema | [pending] | [pending] | [pending] |
| Emphysema | [pending] | [pending] | [pending] |
| Fibrosis | [pending] | [pending] | [pending] |
| Effusion | [pending] | [pending] | [pending] |
| Pneumonia | [pending] | [pending] | [pending] |
| Pleural_Thickening | [pending] | [pending] | [pending] |
| Cardiomegaly | [pending] | [pending] | [pending] |
| Nodule | [pending] | [pending] | [pending] |
| Mass | [pending] | [pending] | [pending] |
| Hernia | [pending] | [pending] | [pending] |
| Lung Lesion | [pending] | [pending] | [pending] |
| Fracture | [pending] | [pending] | [pending] |
| Lung Opacity | [pending] | [pending] | [pending] |
| Enlarged Cardiomediastinum | [pending] | [pending] | [pending] |

## Checkpoint provenance and independence

- Checkpoint key/package version: [pending; configured default is densenet121-res224-all]
- Weight artifact hash/source/license evidence: [pending]
- Documented training datasets: [pending]
- Record-level overlap review method and evidence: [pending]
- Potential parent/derived-dataset overlap: [pending]
- Independence conclusion: [pending / unresolved]

Do not state that the cohort is independent from a checkpoint unless record-level or otherwise authoritative evidence supports that claim. For the current all checkpoint, the installed TorchXRayVision metadata lists NIH, PC, CheX, MIMIC-CH, Google, OpenI, and RSNA source tags, while a record-level training manifest is unavailable in this repository.

## Split protocol and validation checks

- Split source/algorithm/version: [pending]
- Random seed, if any: [pending]
- Patient identifier/crosswalk source: [pending]
- Train / validation / test image counts: [pending]
- Unique patient counts per split: [pending]
- Patient overlap result: [pending]
- Duplicate path/image review: [pending]
- Manifest SHA-256: [pending]
- Train rows inferred: [must be 0]
- Validation-only threshold selection confirmed: [pending]
- Test set remained untouched until final run: [pending]

## Results

Paste actual per-finding values from the evaluator CSV/JSON. Preserve `null` and the associated undefined reason. Do not convert undefined to 0.

| Finding | Test n | Missing | Positive | Negative | Validation threshold (raw score) | AUROC | Sensitivity | Specificity | Precision | Recall | F1 | TN | FP | FN | TP | Undefined reason |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| [copy one row per actual output] | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] |

- Scores are raw model outputs, not calibrated probabilities: [confirm]
- Thresholds were selected on validation data only: [confirm]
- JSON result path/hash: [pending]
- CSV result path/hash: [pending]

## Interpretation and limitations

- Performance claim status: [pending]
- Checkpoint/data overlap limitation: [pending]
- Label compatibility limitation: [pending]
- Missing classes or labels and undefined metrics: [pending]
- Patient-level separation limitation: [pending]
- This is not clinical validation and must not guide patient care.

If prerequisites remain unresolved, replace this section with: **Evaluation pending; no performance metrics were computed.**
