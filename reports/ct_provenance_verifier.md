# CT source-provenance verifier

## Purpose and boundary

`ct_provenance.py` adds a non-inference verifier for a source-DICOM metadata JSON file and a NIfTI conversion-record JSON file. `segmentation.segment_nifti` invokes it for pretrained mode after the existing signed review, exact-file, and geometry checks, and before loading the checkpoint. The CLI accepts `--source-metadata` and `--conversion-record` paths.

The verifier checks the contents and consistency of the supplied records. It does not read DICOM pixel data, prove that metadata was truthfully extracted from source DICOM, authenticate the human reviewer, or make synthetic fixtures into anatomical evidence. Its tests use generated metadata and bytes only.

## Input record contract

The source metadata JSON uses schema `medical-imaging-source-dicom-series/v1` and contains:

- `patient_coordinate_system: "LPS"`, `series_instance_uid`, and `frame_of_reference_uid`.
- `slices`: one object for each source instance, containing `sop_instance_uid`, matching series and frame UIDs, `modality: "CT"`, `patient_position`, `image_orientation_patient` (six values), `image_position_patient` (three values), `rows`, `columns`, `pixel_spacing_mm` (`[row, column]`), and `spacing_between_slices_mm`.

The conversion JSON uses schema `medical-imaging-nifti-conversion/v1` and contains:

- `nifti_affine_coordinate_system: "RAS"`, `nifti_sha256`, `source_metadata_sha256`, `nifti_shape`, and `source_sop_instance_uids_by_slice_index` in ascending physical slice projection order.
- `nifti_voxel_to_source_indices`: an integer 4×4 signed-axis-permutation matrix mapping NIfTI voxel `[i,j,k,1]` indices to DICOM `[column,row,slice,1]` indices. Resampling and cropping are not represented or accepted; mapped voxel extents must equal all source-grid extents, and flip translations must be exactly `source_extent - 1`.
- `conversion_software` and `conversion_version`, which must match the signed review.

The existing signed review's source and conversion digest fields must equal the SHA-256 of the exact JSON file bytes. The conversion record must bind to the exact NIfTI byte hash and metadata-file hash. These format identifiers and field names are this project's verifier contract. `ct_dicom_conversion.py` now provides a supported extractor for a genuine single-series CT DICOM directory: it validates the series, writes a NIfTI volume in ascending physical slice order, and emits the exact source metadata and conversion records above. It does not attest anatomy; reviewer approval remains separate.

## Checks performed

The verifier rejects missing/unreadable records, unsupported schemas and coordinate conventions, hash mismatches, wrong NIfTI hash/shape, unrelated Series/Frame UIDs, duplicate SOP Instance UIDs, inconsistent modality, patient position, dimensions, pixel spacing or direction cosines, missing nominal slice spacing, duplicate/non-monotonic positions, slice gaps/irregular stacks, and a source-slice inventory/order that differs from the conversion record. It validates the conversion index map as a full-grid bijection: a signed permutation must map each NIfTI axis to one source axis with equal extent, and translations must be zero for positive axes or `source_extent - 1` for flipped axes. Thus valid permutations and flips are supported, but crops, shifts, partial coverage, and out-of-bounds mappings are rejected. It then builds the source DICOM LPS voxel-to-patient affine from ImageOrientationPatient, ImagePositionPatient, PixelSpacing, and slice steps, converts LPS to RAS, and compares the result to the inspected NIfTI affine.

These checks establish mathematical consistency among submitted records. A person must still establish the records' provenance and anatomical meaning from the original source series and a documented conversion. In particular, a valid SHA-256 only proves byte identity after the record is supplied; it does not prove that the record is authentic.

## Current API upload contract

1. `POST /ct/inspect` receives raw NIfTI bytes and returns a short-lived geometry token. The server computes the volume and geometry hashes, creates a unique inspection identifier, and sets an HttpOnly, SameSite=Strict browser-session cookie. The token is bound to the hash of that cookie; inspection tokens are not interchangeable even for repeated uploads of identical volume bytes.
2. `POST /ct/evidence` receives `Content-Type: application/json`, the same browser-session cookie, `X-Geometry-Token`, and the existing `X-Orientation-Review` header. Its JSON body must contain exactly `source_metadata_base64` and `conversion_record_base64`. Each decoded UTF-8 JSON file is limited to 1 MiB; the complete request body is limited to 3 MiB. The server decodes and hashes those bytes, validates the signed review, and runs this verifier against geometry and volume hash embedded in the server-signed inspection token. It returns an opaque, one-use `provenance_token`, valid no longer than the inspection token (15 minutes) and bound to that exact inspection, browser session, volume, geometry, and review.
3. `POST /ct/segment` receives the exact original NIfTI bytes plus the same browser-session cookie, `X-Geometry-Token`, `X-Orientation-Review`, and `X-Provenance-Token`. The server rehashes and reinspects the NIfTI, requires it to match both tokens and review, consumes the one-use evidence token, and passes the cached source-record bytes to the library. The library rechecks their hashes and geometry against the staged NIfTI before checkpoint loading.

At most eight pending record pairs are retained in process memory (at most 16 MiB total); each pair is at most 2 MiB and expires with the inspection token. Entries are keyed by random one-use tokens, never overwritten by volume identity, and lazily purged when expired. Temporary verification files are deleted after each request. A process restart invalidates pending tokens. Provenance records are not persisted. The React CT client now uploads both JSON records with credentials, displays the API validation status, and sends the returned token only with the matching inspection/review. Browser cookie isolation is a local workflow boundary, not user authentication: clients that deliberately share cookies and bearer tokens are not independently authenticated.

Do not use an API request, self-signed test fixture, or synthetic record as proof of orientation. The API checks integrity and consistency, not source authenticity or reviewer independence.

## NIfTI-origin Task09 research path

The DICOM verifier and `/ct/evidence` contract above remain unchanged. A distinct Task09 NIfTI-origin path exists for the distributed `imagesTs/spleen_1.nii.gz` candidate:

1. `/ct/nifti-origin/record` requires the existing inspection token/session and a bounded base64 `dataset.json` with `image_path: "./imagesTs/spleen_1.nii.gz"`. It checks the Task09 Spleen name/release, CT modality, and that exact path in the manifest's `test` list. The server creates `medical-imaging-ct-nifti-origin/v1` from those actual manifest bytes and the server-inspected file/geometry identity. The record states `split: "test"`, no distributed label, `volume_source_authenticity: "not_independently_authenticated"`, `original_dicom_metadata: "not_supplied_in_the_dataset_distribution"`, and `conversion_history: "unavailable"`. It does not assert that a DICOM source never existed or that the image bytes are authenticated by the manifest.
2. A human reviewer must independently inspect orientation/laterality and physical slice direction against a suitable authoritative reference. The reviewer prepares and signs schema `medical-imaging-nifti-origin-orientation-review/v1` outside this service. It binds the generated record SHA-256, exact NIfTI SHA-256 and geometry SHA-256, and records the independent-reviewer assertion, reference/method, explicit checks, review time and expiry. The server generates neither approval nor signature. NIfTI RAS axis codes are only header values, not orientation evidence.
3. `/ct/nifti-origin/evidence` rechecks the supplied Task09 dataset manifest against the inspection, validates the HMAC review and expiry, verifies the record/file/geometry binding, and issues a one-use token lasting no longer than the 15-minute inspection. `/ct/segment` rehashes/reinspects the exact volume, binds the same review and session, consumes the token, and the library revalidates the record before loading pretrained weights. DICOM-origin evidence tokens cannot authorize the NIfTI-origin path or vice versa.

The HTTP wrapper uses a shared HMAC key. A valid HMAC only verifies integrity relative to that secret; it cannot establish reviewer identity or independence. Signing-key custody and review approval must remain controlled outside the UI. No claim of anatomical orientation is made until a real independent reviewer has supplied acceptable evidence. The NIfTI record alone is not inference authorization.

## Read-only verification of the downloaded Task09 files

The downloaded files at `C:\Users\jravi\Downloads\Task09_Spleen\Task09_Spleen` were checked without modifying the dataset:

- `imagesTs/spleen_1.nii.gz` SHA-256: `a502f7e81d6b3742f5100e501b4e80180264d00e9aa04dfee43d821b1c6b07e7` (matches the origin record).
- `dataset.json` SHA-256: `919b07212b67023580b759f54a354ee93e01f26b09112ca4f3aceeed6eaa5b3f` (matches the origin record); the manifest lists `./imagesTs/spleen_1.nii.gz` among its 20 test cases.
- Inspected geometry exactly matches the record: shape `[512, 512, 34]`, spacing `[0.6425780057907104, 0.6425780057907104, 5.0]` mm, axis codes `R/A/S`, qform/sform codes `1/1`, datatype `float32`, and 8,912,896 voxels. Geometry SHA-256: `5d3b380cd436250436a3f9cdd1a12856bafd25d08400ebc38b7ce1fe9f65c2d9`.
- The live CT API returned `inspection_only` for these bytes and generated an origin record whose canonical SHA-256, `b17f4d956bc5372648dcb9c0efb22145d10248bd34cac976b07c8e6750466144`, matches the checked-in origin record.
- The CT health endpoint reports the trusted-review-key configuration flag as true and the checkpoint as present. This does not verify how the shared secret is protected or authenticate any reviewer.

These are identity and metadata checks, not orientation approval. The NIfTI RAS header codes do not establish laterality or physical slice direction. The distributed test image has no corresponding label, so it supports technical inference only, not Dice evaluation.

## Remaining authorization for Task09 `spleen_1`

The generated NIfTI-origin record is metadata and exact-volume identity evidence only. Before inference, an independent human reviewer must use an appropriate authoritative reference to verify laterality and physical slice direction for this exact volume; NIfTI RAS header codes alone are insufficient. The reviewer must prepare the NIfTI-origin review manifest with the record, file, and geometry hashes, review method/reference, explicit checks, reviewer identity/independence assertion, and an expiry no more than seven days after review. A trusted operator must sign that manifest out of band with the configured `CT_ORIENTATION_REVIEW_HMAC_KEY`; the application does not create reviewer approval or signatures. This shared-key HMAC provides integrity, not reviewer identity, so signing-key custody must be managed separately.

To authorize one inference, use the downloaded files whose hashes are listed above (or repeat the checks if either file changed), load them through the NIfTI-origin UI, and generate the server-side origin record from the distributed `dataset.json`. Independently inspect the exact volume against an authoritative orientation reference and obtain the completed human review/signature described in [ct_nifti_orientation_review_process.md](./ct_nifti_orientation_review_process.md). Upload that review before expiry. The browser must retain the same inspection session and exact file; the evidence token lasts no longer than the 15-minute inspection and is one-use. Any changed file, geometry, origin record, review, session, or expired token requires a fresh inspection and review. The Task09 test split supplies no label for this volume, so it supports technical inference only, not Dice evaluation.

No human review or signature has been generated, no real CT inference or predicted mask was run, and anatomical orientation remains unverified.
