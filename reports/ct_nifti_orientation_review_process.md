# Task09 Spleen NIfTI orientation review

The CT workflow must remain inspection-only until a qualified, independent human reviewer has verified orientation and the CT service has accepted the signed evidence. The review template is [ct_nifti_orientation_review_template.json](./ct_nifti_orientation_review_template.json). Its false/empty fields and empty signature are intentional; it is not valid authorization.

The Task09-specific reviewer package, including a fresh live-API origin record, exact source copies, unsigned bound template, and Windows signing/submission instructions, is in [task09_spleen_1_reviewer_package](./task09_spleen_1_reviewer_package/README.md).

## What is currently known

The provenance record at [task09_spleen_1_nifti_origin.json](./task09_spleen_1_nifti_origin.json) describes the Medical Segmentation Decathlon Task09_Spleen, release `1.0 06/08/2018`, test image `./imagesTs/spleen_1.nii.gz`. It records shape `512 × 512 × 34`, spacing `0.642578 × 0.642578 × 5.0 mm`, and NIfTI axis codes `R/A/S`. These header values are not proof of anatomical laterality or physical slice direction.

That record states that original DICOM metadata and conversion history were not supplied and image-byte authenticity was not independently authenticated. The Task09 NIfTI must therefore use the NIfTI-origin evidence route; do not invent DICOM UIDs or conversion metadata.

The downloaded files were found at `C:\Users\jravi\Downloads\Task09_Spleen\Task09_Spleen\imagesTs\spleen_1.nii.gz` and `C:\Users\jravi\Downloads\Task09_Spleen\Task09_Spleen\dataset.json`. Both files were read-only inspected; the original dataset was not modified. Their measured hashes match the origin record:

| Item | SHA-256 | Result |
|---|---|---|
| `imagesTs/spleen_1.nii.gz` | `a502f7e81d6b3742f5100e501b4e80180264d00e9aa04dfee43d821b1c6b07e7` | Matches `volume.sha256` |
| `dataset.json` | `919b07212b67023580b759f54a354ee93e01f26b09112ca4f3aceeed6eaa5b3f` | Matches `dataset.dataset_json_sha256` |
| Canonical checked-in origin record | `b17f4d956bc5372648dcb9c0efb22145d10248bd34cac976b07c8e6750466144` | Matches the live API-generated origin record |

The manifest identifies Task09_Spleen release `1.0 06/08/2018`, CT modality, 41 training cases and 20 test cases, and explicitly lists `./imagesTs/spleen_1.nii.gz` in the test split. Its SHA-256, test membership, and all inspected geometry fields were independently checked against the record. The measured geometry is shape `512 × 512 × 34`, spacing `0.6425780057907104 × 0.6425780057907104 × 5.0 mm`, axis codes `R/A/S`, qform/sform codes `1/1`, datatype `float32`, and 8,912,896 voxels. Its geometry SHA-256 is `5d3b380cd436250436a3f9cdd1a12856bafd25d08400ebc38b7ce1fe9f65c2d9`.

The live CT API also inspected these exact bytes and returned `inspection_only`; it generated a NIfTI-origin record with the same canonical SHA-256 above. This proves matching file identity, manifest declaration, and header geometry, not anatomical orientation.

The affine's header-derived direction is increasing voxel `i` toward R by about `0.642578 mm`, increasing `j` toward A by about `0.642578 mm`, and increasing `k` toward S by `5 mm`. These are claims encoded in the NIfTI header for the reviewer to compare with the authoritative reference; they are not independent proof that the volume was correctly oriented.

No signed reviewer-orientation manifest was supplied or found in the dataset directory or reviewer-package files. Therefore this workspace still cannot perform reviewer-authorized segmentation.

The current CT service health response reports the review-key setting as configured and the segmentation checkpoint as present. The health endpoint does not expose or attest secure key custody; the service owner must verify its secure storage, access controls, and trusted signing process without exposing the key.

## Required reviewer evidence

The reviewer must supply a completed and signed JSON manifest with every field from the template:

| Field | Required reviewer evidence |
|---|---|
| `schema` | Keep `medical-imaging-nifti-origin-orientation-review/v1`. |
| `origin_record_sha256` | SHA-256 of the exact server-generated NIfTI-origin record for this inspection. The template value is the canonical digest of the checked-in origin record; replace it if the service-generated digest differs. |
| `volume_sha256` | SHA-256 of the exact `spleen_1.nii.gz` bytes inspected by the service. |
| `geometry_sha256` | SHA-256 of the exact server-inspected geometry object, including affine, shape, spacing, axis codes, and qform/sform fields. |
| `nifti_axis_codes` | Exact axis-code array returned by that inspection; the checked-in record says `["R", "A", "S"]`. |
| `reviewer_id` | Accountable reviewer identity, not a generic role or anonymous label. The service validates presence/format but does not authenticate this identity. |
| `reviewer_independent` | `true` only when the reviewer is independent of the person preparing/submitting the volume. |
| `laterality_verified` | `true` only after independently establishing left/right anatomy from an authoritative reference tied to this exact volume. |
| `slice_direction_verified` | `true` only after independently establishing physical slice direction from that reference. |
| `physical_slice_direction` | `inferior_to_superior` or `superior_to_inferior`, as actually verified. |
| `orientation_reference` | Specific authoritative source identifier/location and enough detail to retrieve the source used. |
| `orientation_method` | How the reviewer matched the reference to this exact volume and checked laterality and physical direction. |
| `reviewed_at`, `expires_at` | Timezone-qualified ISO-8601 timestamps. The review must be current and expire no more than seven days after review. |
| `review_notes` | Reviewer rationale and any limitations. |
| `signature_hmac_sha256` | HMAC-SHA256 signature over the canonical JSON of all other fields (keys sorted, compact separators, ASCII encoding). This must be generated by an authorized signing process using the service's trusted shared key; never guess or fabricate it. |

The server validates the schema, HMAC, reviewer-independence attestation, explicit laterality/direction checks, physical direction, timestamps, axis codes, and exact file/geometry/origin-record hashes. It does not authenticate reviewer credentials or determine whether the reviewer's claimed approval is legitimate. HMAC proves integrity against the configured shared key; it does not independently authenticate the named reviewer or prove that their anatomical claims are true. Those remain human and operational responsibilities.

The current NIfTI-origin schema has no separate `approved` field. Approval is represented only by the accountable reviewer identity, the explicit independence and orientation attestations, the reviewer rationale, and the valid signature. Do not treat a generated origin record or a valid signature alone as approval.

The CT screen reports five workflow states: **Awaiting independent review**, **Review submitted**, **Authorization validated**, **Segmentation available**, and **Review rejected or expired**. Loading a local JSON file advances only to review-submitted; only the CT backend response advances authorization. A completed state requires the actual pretrained inference response and a nonempty binary mask with matching geometry. Inspection/evidence tokens are short-lived and evidence is one-use, so re-inspect and revalidate before retrying after expiry or an inference failure. A still-current signed review may be reused only when its file, geometry, origin-record hashes, and expiry still match.

## Reviewer and operator procedure

1. **Use the verified source files.** The downloaded volume and manifest currently match the recorded hashes above, and the manifest lists this image in its test split. Preserve the originals. If either file changes, repeat the verification and use a newly generated record; do not rely on the checked-in hashes for changed bytes.
2. **Create a fresh inspection in the application.** Select the NIfTI-origin Task09 path and inspect this exact `spleen_1.nii.gz`. Confirm the returned file hash and geometry match the values above. Upload this `dataset.json` and generate the server-side NIfTI-origin record. The current live API generated the matching canonical record hash above, but each authorization attempt still needs a fresh same-session inspection. Keep the application open in that browser session; its inspection and evidence tokens expire.
   The CT screen shows the exact file and geometry hashes and the complete inspected geometry. Use **Download unsigned review template bound to these hashes** only after the record is generated. The downloaded template contains no approval or signature; it is an input for the independent reviewer, not authorization.
3. **Establish orientation independently.** A qualified reviewer who did not prepare the submission must inspect the exact volume against an authoritative reference that establishes left/right anatomy and physical slice direction. The distributed Task09 record does not include original DICOM metadata or conversion history, and RAS header codes alone are insufficient. If no authoritative, volume-matched reference is available, stop: the reviewer must not approve or sign the evidence.
4. **Complete the template truthfully.** Keep the NIfTI-origin schema and exact hashes/axis codes from the live inspection and generated origin record. Fill in the reviewer identity, independence attestation, verified side/direction, reference, method, UTC review/expiry timestamps (within seven days), and notes. Set a verification boolean to `true` only after the corresponding check was completed. Do not add DICOM-origin fields.
5. **Sign in the trusted reviewer environment.** The service owner must provision `CT_ORIENTATION_REVIEW_HMAC_KEY` out of band in secure service configuration (at least 32 UTF-8 bytes). An authorized signing environment must use the same controlled key through `orientation_review.sign_orientation_review`; do not put the key in the browser, source, template, command history, reports, or chat. The helper rejects explicit keys shorter than 32 bytes. Save the resulting manifest as a JSON file. If the key cannot be made available under approved controls, do not sign.
6. **Submit and validate evidence.** Upload the signed manifest in the same browser session and select **Validate NIfTI-origin evidence**. Proceed only when the API returns `provenance_checked` for this inspection. A mismatch, expired review, missing key, or any rejection keeps segmentation blocked.
7. **Run and verify inference.** Select **Run pretrained spleen segmentation** on the same inspected file before the one-use evidence token expires. Confirm the API reports `completed` and `model_status: pretrained`; the returned mask must be finite, nonempty, use only labels `0` and `1`, and have the same shape and affine geometry as the input. The UI decodes and displays that returned mask and volume, not a placeholder. Record the HTTP response and these checks as the verification evidence.

Until steps 1–6 succeed with genuine reviewer evidence, do not run or claim authorized CT segmentation. No reviewer identity, anatomical approval, signature, or mask has been supplied or generated for this workspace.
