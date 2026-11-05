# Medical Imaging AI Research Demo

**Research and educational use only. Not for diagnosis, treatment, or patient care.**

This repository contains a React + Vite frontend, a local Streamlit X-ray application, and three local FastAPI services:

- **X-ray / doctor-review API** (`inference_api.py`, port `8000`): `/infer` verifies decoded PNG/JPEG and qualifying DICOM pixels with the local modality classifier before loading the finding model. The API also hosts local doctor-review persistence endpoints.
- **Local X-ray Streamlit app** (`app.py`, port `8501`): uses the same fail-closed modality verifier before running the existing TorchXRayVision DenseNet121. The verifier threshold is `0.99`; its score is not a calibrated probability.
- **CT segmentation API** (`segmentation_api.py`, port `8002`): an isolated MONAI 3D spleen CT segmentation workflow with NIfTI inspection, source provenance/evidence checks, reviewer-signed orientation review, and segmentation.
- **MRI segmentation API** (`mri_api.py`, port `8003`): an isolated MONAI BraTS brain-tumor segmentation workflow for four aligned skull-stripped 1 mm NIfTI volumes. The latest fresh API inference attempt on a real BraTS case timed out after 900 seconds, so fresh end-to-end MRI inference is not currently verified. Existing mask artifacts were checked for finite, nonempty values, allowed labels, and alignment with the canonical T1c geometry; this does not establish segmentation quality. No MRI Dice/IoU or clinical performance evaluation was performed.

No clinical validation, clinical accuracy, or diagnostic utility is claimed. The modality verifier was evaluated on the labeled, source-specific research collection described below; this does not establish real-world or clinical performance. Synthetic/generated volumes were used for software verification. A fresh MRI API inference attempt timed out, so fresh end-to-end MRI inference is not currently verified. No patient imaging data is included in the repository.

## Features

- React 19 + Vite frontend with modality selection for X-ray, CT, and MRI.
- X-ray analysis is available in React/FastAPI and Streamlit only after modality-verifier acceptance. The API returns all 18 DenseNet121 outputs and a Grad-CAM overlay for the selected finding; inference status and persistence status are shown separately.
- Modality-verifier decisions use the checkpoint's `0.99` score threshold. Unsupported, invalid, low-confidence, and metadata-insufficient inputs remain rejected; verifier and finding-model scores are not calibrated probabilities.
- Study persistence uses uploaded-byte SHA-256 identifiers. Duplicate submissions by an authorized user return the existing immutable record without overwriting its stored results; distinct images remain separate entries.
- CT workflow: inspect a 3D NIfTI volume, upload bounded source-metadata and conversion-record JSON, validate a reviewer-signed orientation review, then run the fixed local MONAI spleen checkpoint.
- MRI workflow: upload four aligned NIfTI volumes (`t1c`, `t1`, `t2`, `flair`) and run the fixed local BraTS checkpoint.
- NIfTI slice viewers for CT and MRI results, including mask overlay and mask download.
- Local SQLite doctor-review persistence keyed by a 64-character lowercase SHA-256 study ID, with a shared bearer token.
- Frontend tests and a production build are included.

## Architecture

```text
React/Vite frontend (:5173)
  |
  +-- X-ray/review FastAPI (:8000)  -> inference_api.py + doctor_reviews.py
  |      X-ray inference: fail-closed pixel verifier -> DenseNet121 -> Grad-CAM
  |      Doctor reviews: /reviews/{study_id} GET/PUT -> SQLite
  |
  +-- CT FastAPI (:8002)            -> segmentation_api.py
  |      /ct/inspect -> /ct/evidence -> /ct/segment
  |      MONAI CPU checkpoint, provenance and signed orientation review
  |
  +-- MRI FastAPI (:8003)           -> mri_api.py
         /mri/segment
         MONAI CPU checkpoint, four aligned NIfTI inputs
```

The CT and MRI services are separate from the X-ray workflow. They do not use the TorchXRayVision chest X-ray model. The frontend communicates with loopback services only.

## Prerequisites

- **X-ray / review environment:** Python 3.13 (`.venv`).
- **CT / MRI environment:** Python 3.10 (`.venv-seg`).
- Node.js and npm for the React frontend. Verified with Node `v24.18.0` and npm `11.16.0`; a current Node LTS release is expected to work.
- Windows PowerShell examples are shown. Adjust virtual-environment paths for other shells.
- The CT and MRI checkpoints must be present locally:
  - `bundles/spleen_ct_segmentation/models/model.pt` (19,297,197 bytes)
  - `bundles/brats_mri_segmentation/models/model.pt` (18,840,620 bytes)
- CT and MRI model weights, datasets, patient images, and generated inference outputs are intentionally excluded from source publication; provide compatible, appropriately licensed weights locally before using those workflows.
- The TorchXRayVision checkpoint (`densenet121-res224-all`) is loaded by the Streamlit app only after modality verification; FastAPI `/infer` remains blocked. The first Streamlit use may download the checkpoint if it is not cached.

## Installation

From the repository root:

### 1. X-ray / review Python environment

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. CT / MRI Python environment

```powershell
py -3.10 -m venv .venv-seg
.\.venv-seg\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-segmentation.txt
```

### 3. Frontend

```powershell
Set-Location .\web
npm.cmd install
```

## Configuration

Environment variables:

| Variable | Service | Required | Purpose |
|---|---|---:|---|
| `DOCTOR_REVIEW_API_TOKEN` | X-ray/review API | Yes for review persistence | Shared bearer token, at least 32 bytes. |
| `MEDICAL_REVIEW_DB_PATH` | X-ray/review API | No | Override SQLite path. Default is `%LOCALAPPDATA%\medical-imaging-ai\doctor-reviews.sqlite3` (or `~/.local/share/...` if `LOCALAPPDATA` is unset). |
| `CT_ORIENTATION_REVIEW_HMAC_KEY` | CT API | Yes for CT evidence/segmentation | Shared HMAC key for reviewer-signed orientation reviews, at least 32 bytes. |
| `REVIEW_SESSION_SECRET` | Legacy `auth.py` / `reviews_api.py` component and related tests | Required when those modules are used/imported | Strong random session-signing value; not the bearer token for the current review API. Keep it server-side. |
| `VITE_INFERENCE_API_URL` | Frontend | No | X-ray/review API URL. Default `http://localhost:8000`. |
| `VITE_SEGMENTATION_API_URL` | Frontend | No | CT API URL. Default `http://127.0.0.1:8002`. |
| `VITE_MRI_API_URL` | Frontend | No | MRI API URL. Default `http://127.0.0.1:8003`. |
| `VITE_XRAY_API_URL` | Frontend | No | Review API URL used by `reviewApi.js`. Default `http://127.0.0.1:8000`. |
| `ORTHANC_URL`, `ORTHANC_USERNAME`, `ORTHANC_PASSWORD` | Optional integration | No | Orthanc connector settings. Live Orthanc integration is not verified. |

`web/.env.local` already contains the non-secret CT and MRI loopback URLs.

`DOCTOR_REVIEW_API_TOKEN` and `CT_ORIENTATION_REVIEW_HMAC_KEY` are server-side secrets. Supply them to their respective processes through the local shell or another secret manager; never put them in Vite variables, frontend files, or source control. The Streamlit X-ray application does not use either secret.

## Startup

Start each service in a separate terminal. Use loopback only.

### X-ray / doctor-review API (port 8000)

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:DOCTOR_REVIEW_API_TOKEN = '<at-least-32-character-secret>'
.\.venv\Scripts\python.exe -m uvicorn inference_api:api --host 127.0.0.1 --port 8000
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected: `status: ok`, `model_status: not_loaded`, `model_loaded: false`.

### CT API (port 8002)

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:CT_ORIENTATION_REVIEW_HMAC_KEY = '<at-least-32-byte-secret>'
.\.venv-seg\Scripts\python.exe -m uvicorn segmentation_api:app --host 127.0.0.1 --port 8002
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8002/health
```

Expected: `checkpoint_present: true` and `trusted_orientation_review_configured: true`.

### MRI API (port 8003)

```powershell
$env:PYTHONIOENCODING = 'utf-8'
.\.venv-seg\Scripts\python.exe -m uvicorn mri_api:app --host 127.0.0.1 --port 8003
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8003/health
```

Expected: `checkpoint_present: true`.

### Local X-ray Streamlit app (port 8501)

The local Streamlit app and FastAPI `POST /infer` both provide X-ray finding inference. Each requires modality-verifier acceptance before DenseNet121 inference. This technical capability is not clinical validation.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501`.

### Frontend (port 5173)

```powershell
Set-Location .\web
$env:VITE_INFERENCE_API_URL = 'http://127.0.0.1:8000'
$env:VITE_SEGMENTATION_API_URL = 'http://127.0.0.1:8002'
$env:VITE_MRI_API_URL = 'http://127.0.0.1:8003'
npm.cmd run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Open `http://127.0.0.1:5173`.

## API Endpoint Overview

### X-ray / review API (`http://127.0.0.1:8000`)

| Method | Path | Input | Notes |
|---|---|---|---|
| `GET` | `/health` | None | Returns `status`, `model_loaded`, `model_status`, `model_error`. |
| `POST` | `/infer` | `Content-Type: image/png`, `image/jpeg`, or `application/dicom`; body up to 25 MiB. | Checks decoded image suitability first; rejected inputs return a clear HTTP 422 reason. Accepted images return the model outputs and Grad-CAM. |
| `POST` | `/doctor-reviews/studies` | Authenticated study hash and AI results. | Creates a study or returns the authorized existing record for the same hash without overwriting it. |
| `GET` | `/reviews/{study_id}` | `Authorization: Bearer <DOCTOR_REVIEW_API_TOKEN>`; `study_id` is a lowercase 64-character SHA-256 hex string. | Returns 404 if no review exists. |
| `PUT` | `/reviews/{study_id}` | Same bearer authorization; JSON body. | Creates or updates a review. |

Review JSON fields:

| Field | Type / constraints | Notes |
|---|---|---|
| `status` | `"Pending"`, `"In Review"`, `"Reviewed"`, or `"Requires Further Investigation"` | Required. |
| `ai_summary` | string, 1 to 8000 characters | Required. Immutable after the first save for the same study ID. |
| `doctor_comments` | string, up to 10000 characters | Optional. |
| `agreement` | `"undetermined"`, `"agree"`, `"partially_agree"`, or `"disagree"` | Optional, default `"undetermined"`. |
| `corrected_findings` | string, up to 5000 characters | Optional. |
| `follow_up_recommendations` | string, up to 5000 characters | Optional. |
| `reviewer_id` | string, 1 to 128 characters | Required by API schema; the UI collects it but does not independently authenticate it. |

### CT API (`http://127.0.0.1:8002`)

| Method | Path | Input | Notes |
|---|---|---|---|
| `GET` | `/health` | None | Returns workflow, checkpoint presence, and review-key configuration. |
| `POST` | `/ct/inspect` | Raw NIfTI bytes; `Content-Type: application/octet-stream`; `X-Volume-Filename: *.nii` or `*.nii.gz`. Request body up to 256 MiB. | Returns file SHA-256, byte size, geometry, geometry SHA-256, attestation token, and inspection metadata. No inference. |
| `POST` | `/ct/evidence` | `Content-Type: application/json` with exactly `source_metadata_base64` and `conversion_record_base64`; headers `X-Geometry-Token`, `X-Orientation-Review`. Body up to 3 MiB. | Validates bounded source provenance records and returns a provenance token. |
| `POST` | `/ct/nifti-origin/record` | `Content-Type: application/json` with `dataset_json_base64` and `image_path`; header `X-Geometry-Token`. | For the selected Task09 candidate, validates `dataset.json` membership and returns a server-generated NIfTI-origin record bound to inspected volume/geometry. No approval or signature is generated. |
| `POST` | `/ct/nifti-origin/evidence` | Same dataset JSON body plus `X-Geometry-Token` and `X-Orientation-Review`. | Verifies the separate NIfTI-origin orientation review and returns a one-use, short-lived token bound to the exact record, file, geometry, review and browser session. |
| `POST` | `/ct/segment` | Raw NIfTI bytes; headers `X-Volume-Filename`, `X-Geometry-Token`, `X-Orientation-Review`, `X-Provenance-Token`. | Returns compressed volume preview and mask, mask SHA-256, labels, geometry, display window, and warnings. |

CT review/evidence requirements:

- Inspection token must match the uploaded file bytes and geometry.
- Orientation review must be reviewer-signed JSON with schema `medical-imaging-ct-orientation-review/v1`, matching file and geometry hashes, unexpired, and containing both reviewer checks.
- Source metadata and conversion records must be bounded JSON files.
- `CT_ORIENTATION_REVIEW_HMAC_KEY` must be configured with at least 32 bytes.
- The DICOM-origin and NIfTI-origin evidence schemas are distinct; a Task09 distribution is never represented as DICOM or as a known DICOM conversion.
- The NIfTI-origin route accepts only `spleen_1.nii.gz` with the Task09 test-list path `./imagesTs/spleen_1.nii.gz` and checks the supplied manifest's Task09 Spleen name, release, CT modality and test membership. The generated record includes the `dataset.json` byte hash, inspected volume SHA-256 and geometry fingerprint, and explicitly states that DICOM metadata was not supplied, conversion history is unavailable, the test label is absent, and uploaded image-byte authenticity is not independently established by manifest membership.
- NIfTI-origin review schema is `medical-imaging-nifti-origin-orientation-review/v1`; it must bind the generated origin-record SHA-256, volume hash and geometry hash, assert an independent reviewer and explicit laterality/physical-slice-direction checks, cite the orientation reference/method, and expire within seven days. The service does not sign it or treat RAS header codes as anatomical verification. HMAC integrity does not authenticate reviewer identity.

### MRI API (`http://127.0.0.1:8003`)

| Method | Path | Input | Notes |
|---|---|---|---|
| `GET` | `/health` | None | Returns workflow, checkpoint presence, required sequences, and intended use. |
| `POST` | `/mri/segment` | `multipart/form-data` with exactly four file fields: `t1c`, `t1`, `t2`, `flair`. Each file must be non-empty `.nii` or `.nii.gz`, no larger than 128 MiB; combined upload no larger than 512 MiB. | Returns compressed mask/preview arrays, geometry, spacing, study ID, output classes, label counts, and a downloadable NIfTI mask. |

## Test Commands and Results

Frontend tests and build:

```powershell
Set-Location .\web
npm.cmd test
npm.cmd run build
```

Latest recorded result: `npm.cmd test` passed **26 tests**, 0 failed. `npm.cmd run build` succeeded.

Classifier environment tests (`.venv`, Python 3.13). The segmentation and MRI modules require `.venv-seg` and should be excluded here:

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  --ignore=tests\test_segmentation.py `
  --ignore=tests\test_segmentation_api.py `
  --ignore=tests\test_segmentation_checkpoint.py `
  --ignore=tests\test_segmentation_evaluation.py `
  --ignore=tests\test_spleen_evaluation.py `
  --ignore=tests\test_segmentation_integration.py `
  --ignore=tests\test_mri_api.py `
  --ignore=tests\test_mri_segmentation.py
```

Reported result: **175 passed**, 1 Starlette deprecation warning.

Focused X-ray safety tests (`.venv`, Python 3.13). Set a temporary test-only value because review-auth imports require the variable; do not reuse a production secret:

```powershell
$secretBytes = New-Object byte[] 32
$rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
$rng.GetBytes($secretBytes)
$env:REVIEW_SESSION_SECRET = [Convert]::ToBase64String($secretBytes)
.\.venv\Scripts\python.exe -m pytest -q tests\test_inference_api.py tests\test_modality_training.py
Remove-Item Env:\REVIEW_SESSION_SECRET
$rng.Dispose()
```

Latest focused X-ray API, modality-verifier, preprocessing, Streamlit gate, and study-persistence run: **104 passed**, 0 failed, 1 Starlette deprecation warning. The React suite passed **49 tests**; `npm.cmd run build` succeeded.

### Modality-verifier evaluation

The active checkpoint's threshold was recalibrated from `0.9998095631599426` to `0.99`; model weights were not changed. The previous threshold was the minimum positive calibration score, and the rejection message rounded it to `1.000`, obscuring why a score such as `0.992` was rejected. DICOM decoding also used min/max scaling while the verifier's training preprocessing used the 0.5th/99.5th percentiles. On generated DICOM representations, that mismatch caused 11/150 calibration CXRs and 4/150 held-out CXRs to be rejected at the old threshold. DICOM display scaling now uses the matching percentile range.

The labeled collection contains 8,800 images. Each calibration and held-out split has 3,000 non-CXRs and 150 CXRs. At `0.99`, the calibration raster and percentile-normalized DICOM variants each had 0/3,000 false accepts and 0/150 false rejects. On held-out data, raster results were also 0/3,000 false accepts and 0/150 false rejects; the generated DICOM variant had 1/3,000 false accepts (0.0333%) and 0/150 false rejects. A lower `0.505` cutoff increased held-out DICOM false accepts to 12/3,000 (0.4%), so it was not selected. These are source-specific research results; generated DICOM variants are not a real-world DICOM cohort and none of these numbers are clinical claims.

Segmentation environment tests (`.venv-seg`, Python 3.10). Use one OpenMP/MKL thread and a workspace-local pytest base directory:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
New-Item -ItemType Directory -Force .tmp-seg-test\pytest | Out-Null
.\.venv-seg\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp .tmp-seg-test\pytest `
  tests\test_segmentation.py tests\test_segmentation_integration.py tests\test_segmentation_checkpoint.py tests\test_spleen_evaluation.py -q
```

Reported result: **24 passed**, 1 TorchScript deprecation warning.

CT API tests (`.venv-seg`, Python 3.10):

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
New-Item -ItemType Directory -Force .tmp-ctapi-test\pytest | Out-Null
.\.venv-seg\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp .tmp-ctapi-test\pytest `
  tests\test_segmentation_api.py -q
```

Reported result: **45 passed**, 2 warnings.

MRI tests (`.venv-seg`, Python 3.10):

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
New-Item -ItemType Directory -Force .tmp-mri-test\pytest | Out-Null
.\.venv-seg\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp .tmp-mri-test\pytest `
  tests\test_mri_api.py tests\test_mri_segmentation.py -q
```

Reported result: **18 passed**, 2 warnings.

Visual-review renderer tests (`.venv`, Python 3.13):

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_visual_review_render.py -q
```

Reported result: **12 passed**, 0 warnings.

Notes:

- Do not run the full unfiltered `.venv` pytest suite as a release gate: segmentation and MRI modules are installed only in `.venv-seg` and will fail in `.venv`.
- CT/MRI tests use generated NIfTI fixtures and software behavior checks. They are not clinical or anatomical validation.
- Browser-level E2E verification is not automated in this repository. Phase 1 checks verified frontend HTTP reachability and API health only; DOM-level browser interaction was not performed.

## Offline Spleen Evaluation (training set only)

Separate pipeline for evaluating the fixed MONAI spleen U-Net on Task09_Spleen NIfTI **training** cases. This is training-set evaluation only: it is not held-out performance, not clinical validation, and does not touch production inference safety gates (orientation review, provenance, geometry).

`powershell
 = '1'
 = '1'
.\.venv-seg\Scripts\python.exe scripts\evaluate_spleen_segmentation.py --help
.\.venv-seg\Scripts\python.exe scripts\evaluate_spleen_segmentation.py --dataset-root <Task09_Spleen> --output-dir outputs/spleen_evaluation --limit 1
`

Full 41-case training-set run (2026-10-26): 41/41 successful, 0 failed; mean Dice 0.9751 (range 0.9361-0.9854), precision 0.9835, recall 0.9668; pooled Dice 0.9773; mean runtime 48.4 s/case (range 12.6-127.9 s), total 1986 s; peak RSS ~1.9 GB (per-case smoke test). 2 extra labels: SpleenLabelmap_1.nii.gz, spleen_10_1.nii.gz. Checkpoint SHA-256 verified. Training-set evaluation only, not held-out performance.
Phase 5 viewer: `research_evaluation_viewer.html` now supports all axial slices (prev/next + slider), three overlay modes (prediction only, ground truth only, comparison with TP green/FP red/FN blue), legend, case ID, slice index, and voxel counts. Reference labels are used only for evaluation/visualization, never as model input. CT windowing is fixed at [-57, 164] HU in all modes.

Visual review (Phase 4): three representative training cases (lowest/median/highest Dice: spleen_56, spleen_10, spleen_19) were re-run with `--save-predictions` into `outputs/spleen_evaluation/visual_review`. Predictions were verified (shape, affine, valid labels 0/1, voxel counts, Dice) — see `verification_manifest.json`. A standalone, clearly labelled research-only viewer is `research_evaluation_viewer.html` (open directly in a browser; no production safety gates are bypassed). Production CT safety controls are unchanged.

Access the existing research-only visual review by opening `outputs/spleen_evaluation/visual_review/research_evaluation_viewer.html` directly in a browser. It is a research visualization only and does not bypass production CT safety gates. Standalone worst-slice PNGs are in `outputs/spleen_slice_research/`.


## Model Descriptions

### X-ray model

- Model: TorchXRayVision DenseNet121, weights `densenet121-res224-all`.
- Runtime output: 18 ordered finding labels from the model's `pathologies` field.
- Score type: operating-point-normalized scores when `op_threshs` is present; these are not calibrated probabilities.
- Explainability: Grad-CAM targets the final DenseNet dense-block convolution.
- Current status: the active modality-verifier checkpoint uses a measured `0.99` threshold and both Streamlit and FastAPI require its approval before finding inference. The verifier score is not a calibrated probability. Its source-specific labeled evaluation is described above; DenseNet121 finding outputs have not been independently evaluated for clinical performance.

### CT model

- Model: MONAI `spleen_ct_segmentation` 3D U-Net.
- Checkpoint: `bundles/spleen_ct_segmentation/models/model.pt`.
- Checkpoint size: 19,297,197 bytes.
- Checkpoint SHA-256: `502c312893994ec071a85c4f2e7d83a43f7789969faa73dae62f1291177f50fe`.
- Runtime: CPU.
- Input: one 3D NIfTI CT volume.
- Output: binary mask with labels 0 (background) and 1 (spleen).
- Requirements: exact-volume provenance, source metadata, conversion record, and a reviewer-signed orientation review.

### MRI model

- Model: MONAI `brats_mri_segmentation` (BraTS MRI segmentation v0.5.4).
- Checkpoint: `bundles/brats_mri_segmentation/models/model.pt`.
- Checkpoint size: 18,840,620 bytes.
- Checkpoint SHA-256: `860ccb3f1c21c99d0410ad8a1ac4ef6b8fab60cec0a503b0ba42675741a750ae`.
- Runtime: CPU.
- Input: four aligned, skull-stripped 1 mm volumes in order `t1c`, `t1`, `t2`, `flair`.
- Output labels: 0 background, 1 tumor core, 2 whole-tumor remainder, 4 enhancing tumor.
- Threshold: sigmoid probability >= 0.5. This is a segmentation cutoff, not a confidence estimate.

## Dataset Requirements

- **X-ray modality verifier:** A labeled, source-specific collection has calibration and held-out splits and was used for the threshold evaluation summarized above. Its generated DICOM variants are not a real-world DICOM cohort.
- **X-ray finding model:** The 18 DenseNet121 outputs and Grad-CAM are technical model outputs only. No independent per-finding clinical-performance metrics are claimed.
- **CT:** Only synthetic/generated NIfTI fixtures have been used for software verification. Real CT use requires exact source metadata, conversion records, and independent orientation review. No real-volume validation has been performed.
- **MRI:** A fresh API inference attempt on a real BraTS case timed out after 900 seconds, so fresh end-to-end MRI inference is not currently verified. Existing mask artifacts were checked for finite, nonempty values, allowed labels, and alignment with canonical T1c geometry only; this does not establish segmentation quality. No Dice/IoU or clinical performance metrics have been measured. The model expects BraTS 2018-style, skull-stripped, co-registered 1 mm T1c/T1/T2/FLAIR primary-glioma inputs. File metadata cannot verify protocol, registration, skull stripping, or tumor type.
- Do not use patient-identifying data. No patient data is included in this repository.

## Known Limitations

- No clinical validation and no clinical performance metrics. This project is a research prototype, not a clinically validated diagnostic system.
- The X-ray modality verifier is an input-suitability gate, not evidence that the finding model is clinically accurate. Inputs below its evaluated threshold are rejected; accepted DenseNet121 scores are not calibrated probabilities or diagnoses.
- CT inference was not run: no current independent, signed orientation review was available for service-side validation. A valid review bound to the exact volume and geometry, plus trusted `CT_ORIENTATION_REVIEW_HMAC_KEY` configuration, is required before segmentation.
- Fresh MRI API inference timed out after 900 seconds; existing mask checks do not verify fresh inference or segmentation quality. Dice/IoU and clinical performance remain unevaluated.
- CT service cannot establish source authenticity or independently prove anatomical orientation.
- MRI service cannot verify sequence identity, skull stripping, registration, or tumor type from file metadata.
- Doctor review uses a shared bearer token and a self-entered reviewer ID; it does not provide independent reviewer authentication.
- Doctor-review storage is local SQLite without schema migration/versioning.
- The review API has no authentication beyond the shared token and must remain loopback-only.
- Notes are free text and are not automatically screened for identifiers.
- No automated browser E2E suite is currently in the repository. Browser checks in this handoff were limited to frontend HTTP reachability; DOM-level interaction was not automated.
- Deprecation warnings remain for Starlette/httpx and TorchScript.
- Docker build/runtime, live Orthanc, and DVC CLI/remote operations remain unverified.
- If a container is run for local experimentation, bind it to loopback only: `docker run --publish 127.0.0.1:8501:8501 ...`. The Streamlit CXR classifier may use the TorchXRayVision cache under `/home/appuser/.torchxrayvision`; keep that path mounted or cached as needed.

## Safety and Privacy

- **Educational/research use only. Never use these outputs for diagnosis, treatment, or patient-care decisions.**
- TorchXRayVision scores are operating-point-normalized model scores, not calibrated probabilities. Dataset bias, labeling differences, acquisition differences, and domain shift may affect outputs.
- Grad-CAM is a model sensitivity visualization. It does not establish why a prediction occurred.
- Processing runs locally after model weights are downloaded. Do not use patient-identifying information.
- The local review database stores UTC time, model, finding, raw score, status, Agree/Disagree/Inconclusive decision, optional notes, and an opaque per-session prediction UUID. It does not store the image, filename, or patient-ID field.
- Notes are free text and are saved as entered. The UI warns reviewers not to include names, IDs, dates of birth, or other identifiers, but it cannot automatically guarantee that notes contain no identifying details.
- Review feedback is separate from and does not change model scores.
- The editable local SQLite database is not authenticated, tamper-evident, or suitable as a compliant medical record.
- The APIs are intended for loopback research use only. Do not expose them to a network.
- Keep `DOCTOR_REVIEW_API_TOKEN` and `CT_ORIENTATION_REVIEW_HMAC_KEY` out of frontend configuration and source control.

## Demo Checklist

1. Start the X-ray/review API, CT API, MRI API, local Streamlit app, and frontend using the Startup commands above (`.venv` for ports 8000/8501, `.venv-seg` for 8002/8003, npm for 5173). Confirm the API `/health` endpoints report the expected status.
2. **X-ray:** use the React/FastAPI or local Streamlit workflow with supported images. Inference runs only after the modality verifier accepts the input; the FastAPI endpoint returns model outputs and Grad-CAM after successful inference.
3. **CT:** inspect a `.nii` or `.nii.gz` volume and provide the required provenance records. Run segmentation only after the backend validates independent signed orientation evidence for that exact volume; otherwise keep the workflow inspection-only.
4. **MRI:** select four NIfTI volumes, run segmentation, and confirm the slice viewer, label counts, and downloaded mask.
5. **Doctor review:** enter a token of at least 32 characters and a reviewer ID, save a review, refresh the page, retrieve the saved review, and confirm the saved comment content.
6. Before X-ray inference, the FastAPI health response should report `model_status: not_loaded`; after a successful FastAPI inference, the model may be loaded.
7. Stop all services with Ctrl+C. Remove generated test volumes and temporary files. Do not describe synthetic fixtures as clinical or model-performance evidence.
