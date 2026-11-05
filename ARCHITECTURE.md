# Architecture

```text
Authorized raster/DICOM input
      |
local modality verifier accepts the image
      |
TorchXRayVision normalization to [-1024, 1024]
      |
center crop + bilinear resize to 224 x 224
      |
      +------------------------+
      |                        |
      v                        v
TorchXRayVision DenseNet121   Grad-CAM on final dense-block conv
      |                        |
ranked raw finding scores    selected finding overlay
      +------------+-----------+
                   v
             Streamlit UI
                   |
                   v
Local review CSV (opaque session UUID, status, decision, optional notes)
```

The separate React/FastAPI `/infer` route rejects every supported image type with HTTP 422 before reading the body or loading the finding model.

Separate, opt-in CT workflow (not connected to the CXR classifier):

```text
attested 3D NIfTI CT + local checkpoint
      -> nibabel geometry validation (affine/qform/sform)
      -> MONAI channel/orientation/spacing/intensity transforms
      -> 3D UNet + 96³ sliding-window inference
      -> inverse transforms + mask validation
      -> uint8 NIfTI with the source affine/header
```

## Image and model flow

`imaging.py` contains pixel preprocessing: raster decode, grayscale conversion, EXIF orientation, center crop, 224 by 224 resize, and TorchXRayVision normalization `(pixel / 255 - 0.5) * 2048`. The local Streamlit path runs the finding model only after the local modality verifier accepts an image. One authorized local image produced 18 finite scores and Grad-CAM visualizations for three findings as a technical smoke; this is not a medical-model evaluation. File format and DICOM metadata alone do not prove pixel content.

`app.py` loads the pretrained `densenet121-res224-all` model through Streamlit's resource cache. The output labels come from the model's `pathologies` field and are checked against the output vector. The configured TorchXRayVision model rescales outputs around its operating-point thresholds; the API identifies these as operating-point-normalized scores, not calibrated probabilities.

`inference_api.py` exposes the X-ray endpoint through local FastAPI, but returns HTTP 422 for every supported image type before reading request bodies, loading the finding model, or generating outputs. This fail-closed behavior applies to direct callers as well as React. The React helper rejects locally and displays a clear explanation. The separate Streamlit app uses the modality verifier before loading the finding model. The API model loader remains lazy and is not reached by `/infer`. CORS allows only localhost Vite origins. The API has no authentication and is for loopback research use only; do not expose it to a network.

`tests/local_web_api_smoke.py` starts a temporary API on an ephemeral loopback port and invokes the same client helper used by React. It checks local frontend rejection and direct API HTTP 422 responses for a generated car illustration and a generated metadata-qualified DICOM; it verifies that neither response contains predictions/Grad-CAM and the finding model remains unloaded. The smoke never loads weights or runs inference. Historical records below describe earlier behavior and are not the current contract.

`dicom_io.py` supports a limited single-frame, single-channel MONOCHROME DICOM subset. DICOM decoding utilities still validate supported encodings and apply available transforms, but the current inference endpoint blocks DICOM before decoding because its metadata cannot establish pixel modality. Metadata stays in-process and is not retained in the image or audit log. DICOM tags can be inaccurate; unknown burned-in annotation is not evidence of de-identification. Pixel display conversion is not a validated diagnostic DICOM rendering pipeline.

`orthanc_client.py` uses Basic-authenticated Orthanc REST calls for server version, opaque study/instance IDs, and selected instance bytes. It validates origin-only URLs, rejects embedded credentials, redirects and non-loopback plain HTTP, validates Orthanc's five-group SHA-1 identifier format, applies a finite positive timeout, and bounds response reads to 25 MiB. It does not request patient tags. Retrieved DICOM bytes remain in Streamlit session memory until explicit clear or session end and are not written to disk. The DICOM decoder remains a limited 2D image-preview path, not a CT-series loader. Mocked HTTP tests cover request/error/response behavior; Orthanc is not bundled and no live service verification has occurred.

Grad-CAM targets `features.denseblock4.denselayer16.conv2` in DenseNet-121 and hooks are removed even if hook registration or visualization fails. Matplotlib uses the headless Agg backend for Windows environments without Tcl/Tk. A failure is shown in the UI while the model output and review controls remain available. The overlay is an explanatory visualization, not proof of causal reasoning.

Numeric affine validity and qform/sform agreement are not proof of anatomical orientation. Pretrained inference now requires the existing signed review and, in the CLI/library path, actual source-metadata and conversion-record JSON files checked by `ct_provenance.py`. The verifier checks record hashes, series/frame and slice inventory, slice spacing/order, a declared voxel-index mapping, and the DICOM LPS to NIfTI RAS affine relationship. It does not prove that source metadata was truthfully extracted or authenticate the reviewer. Task09_Spleen provides no source DICOM series or per-case conversion record; the mismatch reported for the MSD images leaves `spleen_10` unattested.

`segmentation.py` provides the optional spleen CT workflow. Its MONAI and nibabel imports are lazy, and it is not imported by the Streamlit CXR app or FastAPI classification API. It accepts only `.nii`/`.nii.gz`, requires explicit geometry attestation, validates numeric affine and qform/sform consistency, applies the bundle-shaped preprocessing and sliding-window contract, inverts predictions, and saves labels in the original dimensions and affine. The hash-verified MONAI checkpoint is stored locally at `bundles/spleen_ct_segmentation/models/model.pt` and ignored by Git. The loader uses `torch.load(weights_only=True)` and strict state-dict matching. Random `build_unet()` weights remain explicitly untrained.

`segmentation_api.py` is a separate optional FastAPI service. The default launch port and frontend fallback are both loopback port 8002. `/ct/inspect` returns an exact-file/geometry token without loading the model and sets an HttpOnly, SameSite browser-session cookie. `/ct/evidence` accepts bounded base64 JSON source records and a signed-review header, validates their content and geometry binding, then issues a one-use token bound to that browser session and unique inspection. `/ct/segment` requires that token and the same volume/review, and the library rechecks the source records against the staged NIfTI before checkpoint loading. The React CT client uploads both records, shows the server validation status, and proceeds only with the returned token. HMAC checks integrity but not reviewer identity or source authenticity; the cookie scopes the local workflow but is not user authentication. No service was restarted for these source changes. Keep the service loopback-only. Mocked tests do not establish real CT inference or anatomical correctness.

## Task-specific brain MRI flow

```text
four bounded multipart NIfTI uploads (T1c, T1, T2, FLAIR)
      -> nibabel decode and canonical RAS orientation
      -> identical shape/affine + 1 mm isotropic geometry checks
      -> channel-wise nonzero intensity normalization
      -> MONAI SegResNet sliding-window inference (sigmoid, 0.5 threshold)
      -> TC/WT-remainder/ET labels 1/2/4, FLAIR preview, and NIfTI mask
      -> React native-slice viewer and per-study Doctor Review
```

`mri_segmentation.py` checks the local `MONAI/brats_mri_segmentation` checkpoint against its expected size and SHA-256, then loads it with `weights_only=True` and a strict SegResNet state-dict match. Input validation occurs before checkpoint loading. `mri_api.py` is a separate loopback-only service on port 8003. It requires a bounded request length, limits each upload to 128 MiB and the full request to 512 MiB, and serializes inference to avoid concurrent CPU memory pressure. The React client has explicit sequence-labeled file selectors, displays the returned model metadata, geometry and voxel counts, provides an original/overlay slice viewer, and offers the predicted NIfTI mask for download.

This workflow supports only the MONAI BraTS 2018-style primary-glioma subregion task. Inputs must already be co-registered, skull-stripped and 1 mm isotropic; metadata checks cannot verify acquisition sequence identity, skull stripping, registration quality, or tumor type. One complete real BraTS case completed technical inference, but no held-out Dice/IoU or clinical performance evaluation was performed. It is not general MRI interpretation and is not for diagnosis. The source bundle reports validation Dice values, which are not results of this integration; license and attribution notes are in `bundles/brats_mri_segmentation/README.md`.

## Evaluation flow

`evaluate.py` checks that required label columns match the runtime model output label set, rejects empty/duplicate model labels, and processes labels in model output order. This verifies column compatibility only; label meaning still needs dataset-specific review. If `patient_id` is supplied, every row must have one and each value must be limited to one split. It resolves manifest image identities across all splits and rejects duplicates or conflicting duplicate labels, including train/test reuse, without opening or inferring on training files. Relative paths must stay under `--image-root`. Blank labels are excluded and counted for validation and test. Per-finding reports include sample/positive/negative counts, AUROC, sensitivity, specificity, precision, recall, F1, and confusion counts where defined. Thresholds use validation scores only; raw scores are not calibrated probabilities. Outputs explicitly state that performance claims and checkpoint/data independence are not established by the evaluator. The configured `densenet121-res224-all` checkpoint is documented as trained on a mixture including NIH and RSNA; its record-level training manifest is unavailable, so independence cannot be established for those sources. No suitable independent labeled evaluation dataset is bundled; performance evaluation remains pending.

## Audit flow and trust boundaries

Review events are persisted in `data/audit_log.csv` with UTC time, model identifier, opaque per-session prediction UUID, finding, raw model score, status (`Pending Review`, `Reviewed`, or `Needs Further Review`), decision (`Agree`, `Disagree`, or `Inconclusive`), and optional notes. Events append status changes; allowed transitions are validated. The UUID is generated for one loaded input in one Streamlit session; its source-byte digest is kept only in session state, not written to the CSV. The event links the reviewer decision to that prediction session and selected finding without a patient-ID or filename field. Notes are free text saved locally as entered; the UI warns against identifiers, but does not detect or guarantee their absence. Review feedback does not alter model output. The CSV is local and editable, and is not a compliant medical record or tamper-proof audit system. The app has no clinician authentication, role controls, signatures, encryption, or multi-user integrity controls. Legacy review rows are retained and normalized when the CSV is read/upgraded.

Images are processed locally after pretrained weights are fetched. The first model download requires network access; uploaded images are not sent to a remote inference service. Raster uploads are rejected before model loading. DICOM candidate metadata is transient and the pixel decoder returns an image without attached metadata. React clears the file reference after each request; Streamlit upload bytes stay in session memory until cleared with the UI control or the session ends. Use only authorized, de-identified images.

The React CXR path sends image bytes to local FastAPI `/infer`, but the API rejects requests before loading the finding model; no predictions or Grad-CAM are returned through this route. The separate Streamlit path verifies modality before local inference. One authorized real X-ray was used for a technical check, not a held-out evaluation. Generated non-medical fixtures in API tests establish rejection behavior only, not medical-model performance. No uploaded filename is sent as request metadata.

## Packaging and artifact tracking

`Dockerfile` packages only the Streamlit CXR classifier (`app.py`), not React/FastAPI or the MONAI segmentation workflow. It uses pinned `requirements-docker.txt` runtime dependencies, runs as UID 10001, and has a Streamlit liveness health check; that check does not load or validate the model. `/app/data` and `$HOME/.torchxrayvision` are writable, volume-ready paths; the model is not copied into the image and the first inference requires network access unless the cache volume is already populated. `.dockerignore` excludes patient image/volume formats, datasets and tabular manifests, weight artifacts, common secrets, temporary files, DVC internals and pointer files, and local environments/caches while leaving application source visible. The documented host mapping binds to `127.0.0.1`. Docker, Podman, nerdctl, and containerd are absent in this workspace, so package installation, image build, runtime, volume permissions, and health-check execution are unverified. Static tests only check configuration text and ignore-pattern intent; they do not prove Docker's actual build-context behavior or that the image builds.

`.dvc/config` contains only disabled analytics; `.dvcignore` excludes Python environments, Node dependencies/build output, caches, and the local review CSV while allowing deliberate dataset/checkpoint paths to be selected for DVC. `.gitignore` excludes local settings/secrets, datasets, medical image files, model weights, and generated outputs; `.dockerignore` excludes `.dvc`, datasets/manifests, medical image formats, weights, and environments from the Docker context. The local `.dvc` directory has no stages, lock file, cache, remote, or pointer files. Git metadata exists, but the current `master` branch has no commit and the repository has no remotes or indexed files; agent checkpoint refs contain prior commits. DVC CLI/remote operations have not been verified. The static config checks passed (5 tests in `tests/test_dvc_config.py`); they do not substitute for DVC CLI verification. `prepare_rsna_manifest.py` consumes an externally created train/validation/test split map and a patient crosswalk; it does not generate splits or set a random seed. `evaluate.py` accepts the 18 model output label columns, with values `0`, `1`, or blank, excludes train rows from inference, and selects thresholds from validation only. RSNA preparation fills only `Pneumonia` and `Lung Opacity` from the same `Target`; label semantics and checkpoint overlap still require external review. No authorized per-finding evaluation dataset or metric report is present. See README's DVC audit section for the reproducible command and remaining setup requirements.

## Assignment technology checklist (2026-10-23)

| Technology | Status | Verification |
|---|---|---|
| PyTorch | Implemented | Local Streamlit performed technical inference on one authorized X-ray and returned 18 finite labeled outputs. This is not a held-out evaluation. |
| Grad-CAM | Implemented in the CXR path | Unit tests verify hook behavior with a tiny model; one authorized local X-ray produced three Grad-CAM images. This does not establish explanation validity or clinical performance. |
| Docker | Partial | Dockerfile and static packaging tests exist; local image build/smoke test blocked because no container engine is installed. |
| DICOM | Partial | pydicom decoder is used by Streamlit and API; generated single-frame tests pass. Unsupported formats and diagnostic rendering remain out of scope. |
| Orthanc | Partial | Authenticated REST client and mocked request tests pass; no Orthanc server is installed for live integration verification. |
| DVC | Partial | Seed config and safety tests exist; DVC initialization, pointer generation, and push were not tested. |
| MONAI 3D U-Net | Partial | Isolated Python 3.10.11 environment has PyTorch 2.4.1+cpu, MONAI 1.4.0, nibabel 5.2.1, and NumPy 1.24.4. The official CT checkpoint is hash-verified and loads safely with strict state-dict matching. Tests exercise actual MONAI transforms, synthetic untrained forward inference, checkpoint integrity, and CT API/provenance safeguards. The extracted Task09_Spleen `spleen_10` remains unattested because no source DICOM or authoritative exact-volume conversion evidence was found. No pretrained CT inference was performed. |
| MONAI BraTS MRI SegResNet | Implemented, evaluation pending | Public 18,840,620-byte checkpoint passes size/SHA-256 verification and strict safe loading. Synthetic software tests cover canonicalization, preprocessing, geometry rejection, output labels, bounded multipart API behavior, and the React upload/viewer contract. One complete real BraTS case completed inference and output-geometry checks. No held-out Dice/IoU or clinical performance metrics are available; the task is limited to the documented BraTS-style primary-glioma input. |
| React | Partial | The unified X-ray/CT/MRI UI has unit coverage for X-ray API handling, CT upload/inspection/attestation, MRI request/response parsing, and synthetic slice composition. The current frontend suite passes 26 tests and the production build succeeds. This is not browser-rendered or medical-model evaluation evidence. |

The official MONAI `spleen_ct_segmentation` bundle matches the implemented 3D U-Net contract: 1-channel 96-cube CT patches, 2 outputs (background/spleen), channels 16/32/64/128/256, strides 2/2/2/2, and two residual units. The bundle metadata pins MONAI 1.4.0/PyTorch 2.4.0; the isolated Windows environment uses verified PyTorch 2.4.1+cpu because the 2.4.0 wheel failed loading `fbgemm.dll`. The file located in Downloads matched official `models/model.pt` size (19,297,197 bytes) and SHA-256 (`502c3128…17f50fe`), then was copied to the project bundle directory. A safe `weights_only=True` load and strict state-dict load succeeded for all 148 tensors (4,808,917 parameters). Hugging Face metadata marks `model.pt` safe and `model.ts` suspicious; `model.ts` was not copied or executed. The bundle is trained on MSD Task09 Spleen, whose geometry caveat is documented in issue #766. The extracted Task09_Spleen dataset exists locally and its matched training pair was inspected. The user reports a label overlay in 3D Slicer 5.12.4, but no source-DICOM or known-good orientation comparison was provided; inference was rejected before model loading. No network weight or patient data download occurred.

On 2026-10-23, the isolated segmentation suite (`.venv-seg\Scripts\python.exe -m pytest ... tests/test_segmentation.py tests/test_segmentation_integration.py tests/test_segmentation_checkpoint.py -q`, with one OpenMP thread) passed: 17 tests. The integration test used generated NIfTI data and random explicitly untrained weights; it exercised actual MONAI transforms, a real MONAI 3D U-Net forward pass, inverse geometry transforms, and mask export. The checkpoint test verified the local file size/hash and safe `weights_only=True`, strict state-dict load. `pip check` reported no broken requirements. Non-escalated `.venv` execution is blocked by the workspace sandbox boundary on its external Python 3.13 base interpreter; read-only ACL inspection showed the logged-in Windows account has FullControl. The full-suite command succeeded through approved read-only escalation: 66 passed, 2 skipped. No `.venv` files/configuration were changed. `npm.cmd test` passed all 5 tests and `npm.cmd run build` succeeded with Vite 8.3.5. The Task09_Spleen training pair described above was not inferred because semantic geometry was not independently attested; the safeguard returned GeometryError before inference and no predicted mask or Dice was produced.

## Limits

The model's training datasets have differing populations, acquisition systems, and labels. Dataset bias and domain shift can affect performance. There is no independent clinical validation, prospective evaluation, or regulatory review. DICOM support is limited and unvalidated; Orthanc was verified only with mocked HTTP; CT pretrained inference remains blocked for the available volume by its evidence requirements. MRI technical inference was verified on one real BraTS case, but independent performance evaluation remains unverified. This is an educational/research prototype only and must not guide patient care.

## Final submission run commands

Submission documents are maintained in `reports/final_project_report.md` and `reports/submission_demo_script.md`. The report links the evaluation protocol, audit, and blank results template. A project/rubric search found no local grading rubric or submission-instructions file; no unstated rubric is assumed.

The Streamlit CXR app uses the configured TorchXRayVision checkpoint only after the modality verifier accepts an image. The React/FastAPI `/infer` path remains fail-closed and does not load the model. The API is not authenticated and must not be exposed publicly. If the checkpoint is not in the TorchXRayVision cache, eligible Streamlit inference may trigger a weight download. Generated non-medical fixtures are software smoke inputs only and must not be treated as clinical findings.

From the repository root, run FastAPI:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
.\.venv\Scripts\python.exe -m uvicorn inference_api:api --host 127.0.0.1 --port 8000
```

In another PowerShell terminal:

```powershell
Set-Location .\web
$env:VITE_INFERENCE_API_URL = 'http://127.0.0.1:8000'
npm.cmd run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

For the optional CT workflow, install `requirements-segmentation.txt` in `.venv-seg` when dependencies are not already present, then start the separate service from the repository root with `.\.venv-seg\Scripts\python.exe -m uvicorn segmentation_api:app --host 127.0.0.1 --port 8002`. The React checkout points `VITE_SEGMENTATION_API_URL` to `http://127.0.0.1:8002` in `web/.env.local`; never place the CT review HMAC key in Vite configuration. Before launch, provide `CT_ORIENTATION_REVIEW_HMAC_KEY` to the CT process through the approved out-of-band secret mechanism. The CT service is optional to the CXR workflow and remains loopback-only research software.

For MRI, start the separate API with `.\.venv-seg\Scripts\python.exe -m uvicorn mri_api:app --host 127.0.0.1 --port 8003`. `web/.env.local` sets the non-secret loopback URL `VITE_MRI_API_URL`; the frontend receives no CT signing key.

For tests, the final-submission commands are the Python suite in `.venv`, the three segmentation test modules in `.venv-seg` with one OpenMP/MKL thread and a workspace-local pytest base directory, and `npm.cmd test` / `npm.cmd run build` from `web/`. Exact outputs are recorded in the final project report. Segmentation uses a separate Python 3.10 environment and the locally verified MONAI checkpoint; it requires explicit geometry attestation and has no API/UI connection to the classifier. Optional Orthanc configuration uses `ORTHANC_URL`, `ORTHANC_USERNAME`, and `ORTHANC_PASSWORD` or Streamlit secrets. The frontend accepts optional `VITE_INFERENCE_API_URL`, defaulting to `http://localhost:8000`.

Current evidence remains limited: the modality verifier gates local Streamlit inference, and one authorized X-ray has been used for technical verification only. The FastAPI route remains fail-closed. No independent labeled cohort or per-finding clinical performance metrics are available. CT inference remains blocked for the available volume pending exact-volume source evidence, a fresh signed orientation review, and service-side HMAC configuration. One complete real BraTS case completed MRI inference, but no held-out Dice/IoU or clinical performance evaluation was performed. Docker build/runtime, DVC CLI/remote operations, and live Orthanc remain unverified.

Latest software regression results are recorded in README.md and the current verification update in `reports/final_project_report.md`. They are software checks, not medical-model evaluation. No CT inference was performed, no services were stopped or restarted, and browser-level live API integration remains unverified.
