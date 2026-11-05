import { assessCtMaskCompatibility } from "./ctGeometry.js";

const DEFAULT_CT_API_URL = import.meta.env?.VITE_SEGMENTATION_API_URL || "http://127.0.0.1:8002";
const ORIENTATION_REVIEW_SCHEMA = "medical-imaging-ct-orientation-review/v1";
const NIFTI_ORIGIN_REVIEW_SCHEMA = "medical-imaging-nifti-origin-orientation-review/v1";
const TASK09_IMAGE_PATH = "./imagesTs/spleen_1.nii.gz";
const MAX_REVIEW_JSON_BYTES = 12_000;
const MAX_PROVENANCE_RECORD_BYTES = 1024 * 1024;
const MAX_PROVENANCE_BODY_BYTES = 3 * 1024 * 1024;

function niftiName(file) {
  const name = String(file?.name || "");
  if (!name.toLowerCase().endsWith(".nii") && !name.toLowerCase().endsWith(".nii.gz")) {
    throw new Error("Choose a 3D NIfTI volume with a .nii or .nii.gz filename.");
  }
  return name;
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

async function parseResponse(response) {
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error("The CT service returned an unreadable response.");
  }
  if (!response.ok) {
    throw new Error(typeof payload.detail === "string" ? payload.detail : "CT processing failed.");
  }
  return payload;
}

async function requestFailed(fetchImpl, url, options, fallback) {
  try {
    return await fetchImpl(url, options);
  } catch {
    throw new Error(fallback);
  }
}

export async function inspectCtVolume(file, fetchImpl = fetch, apiUrl = DEFAULT_CT_API_URL) {
  const filename = niftiName(file);
  const response = await requestFailed(fetchImpl, `${apiUrl}/ct/inspect`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/octet-stream", "X-Volume-Filename": filename },
    body: file,
  }, "Could not reach the local CT service. Start the isolated segmentation API and try again.");
  const payload = await parseResponse(response);
  if (payload.status !== "inspection_only" || payload.model_loaded !== false || payload.orientation_verified !== false ||
      !payload.attestation_required || typeof payload.attestation_token !== "string" ||
      !/^[0-9a-f]{64}$/.test(payload.file_sha256) ||
      !/^[0-9a-f]{64}$/.test(payload.geometry_sha256) ||
      !Array.isArray(payload.geometry?.shape) || payload.geometry.shape.length !== 3) {
    throw new Error("The CT service returned invalid inspection metadata.");
  }
  return payload;
}

export function orientationReviewIssues(review, inspection, originRecordSha256 = null) {
  const issues = [];
  if (!review || typeof review !== "object" || Array.isArray(review)) {
    return ["The review JSON must contain a top-level object."];
  }
  const niftiOrigin = review.schema === NIFTI_ORIGIN_REVIEW_SCHEMA;
  const expectedSchema = niftiOrigin ? NIFTI_ORIGIN_REVIEW_SCHEMA : ORIENTATION_REVIEW_SCHEMA;
  if (review.schema !== expectedSchema) issues.push(`schema must be "${expectedSchema}".`);
  const requiredFields = niftiOrigin
    ? [
      "origin_record_sha256", "volume_sha256", "geometry_sha256", "nifti_axis_codes",
      "reviewer_id", "reviewer_independent", "laterality_verified", "slice_direction_verified",
      "physical_slice_direction", "orientation_reference", "orientation_method",
      "reviewed_at", "expires_at", "review_notes", "signature_hmac_sha256",
    ]
    : [
      "volume_sha256", "geometry_sha256", "source_metadata_sha256", "conversion_record_sha256",
      "series_instance_uid", "frame_of_reference_uid", "image_orientation_patient",
      "first_image_position_patient", "last_image_position_patient", "slice_order",
      "conversion_software", "conversion_version", "reviewer_id", "reviewed_at",
      "expires_at", "laterality_verified", "slice_direction_verified", "nifti_axis_codes",
      "review_notes", "signature_hmac_sha256",
    ];
  for (const field of requiredFields) {
    if (review[field] === undefined || review[field] === null || review[field] === "") {
      issues.push(`${field} is required and is currently missing or blank.`);
    }
  }
  if (niftiOrigin) {
    const allowedFields = new Set(["schema", ...requiredFields]);
    const extraFields = Object.keys(review).filter((field) => !allowedFields.has(field));
    if (extraFields.length) issues.push(`Unsupported NIfTI-origin field(s): ${extraFields.join(", ")}.`);
  }
  if (!inspection) {
    issues.push("Inspect the exact NIfTI volume before uploading reviewer evidence.");
    return issues;
  }
  if (review.volume_sha256 !== inspection.file_sha256) {
    issues.push(`volume_sha256 mismatch: review has ${review.volume_sha256 || "(blank)"}, inspection has ${inspection.file_sha256}.`);
  }
  if (review.geometry_sha256 !== inspection.geometry_sha256) {
    issues.push(`geometry_sha256 mismatch: review has ${review.geometry_sha256 || "(blank)"}, inspection has ${inspection.geometry_sha256}.`);
  }
  if (JSON.stringify(review.nifti_axis_codes) !== JSON.stringify(inspection.geometry?.axis_codes)) {
    issues.push(`nifti_axis_codes must exactly match inspection axis codes ${JSON.stringify(inspection.geometry?.axis_codes)}.`);
  }
  if (niftiOrigin) {
    if (!originRecordSha256) {
      issues.push("Generate the NIfTI-origin record before loading or validating its review.");
    } else if (review.origin_record_sha256 !== originRecordSha256) {
      issues.push(`origin_record_sha256 mismatch: review has ${review.origin_record_sha256 || "(blank)"}, generated record has ${originRecordSha256}.`);
    }
    if (review.reviewer_independent !== true) issues.push("reviewer_independent must be true after the reviewer confirms independence.");
    if (review.physical_slice_direction !== "inferior_to_superior" &&
        review.physical_slice_direction !== "superior_to_inferior") {
      issues.push("physical_slice_direction must be the independently verified inferior_to_superior or superior_to_inferior value.");
    }
    for (const field of ["reviewer_id", "orientation_reference", "orientation_method", "review_notes"]) {
      if (typeof review[field] === "string" && review[field].trim() === "") {
        issues.push(`${field} must contain meaningful reviewer-supplied information.`);
      }
    }
  }
  if (review.laterality_verified !== true) issues.push("laterality_verified must be true only after independent verification against an authoritative reference.");
  if (review.slice_direction_verified !== true) issues.push("slice_direction_verified must be true only after independent verification against an authoritative reference.");
  if (!/^[0-9a-f]{64}$/.test(review.signature_hmac_sha256 || "")) {
    issues.push("signature_hmac_sha256 must be a 64-character lowercase hexadecimal HMAC generated by the authorized signing process.");
  }
  const reviewedAt = Date.parse(review.reviewed_at || "");
  const expiresAt = Date.parse(review.expires_at || "");
  const timestampHasZone = (value) => typeof value === "string" &&
    /(?:Z|[+-]\d{2}:\d{2})$/i.test(value);
  if (!Number.isFinite(reviewedAt) || !timestampHasZone(review.reviewed_at)) {
    issues.push("reviewed_at must be a timezone-qualified ISO-8601 timestamp.");
  }
  if (!Number.isFinite(expiresAt) || !timestampHasZone(review.expires_at)) {
    issues.push("expires_at must be a timezone-qualified ISO-8601 timestamp.");
  } else if (expiresAt <= Date.now()) {
    issues.push("The review has expired; obtain current reviewer evidence.");
  }
  if (Number.isFinite(reviewedAt) && Number.isFinite(expiresAt)) {
    if (reviewedAt > Date.now() + 5 * 60 * 1000) issues.push("reviewed_at is more than five minutes in the future.");
    if (expiresAt <= reviewedAt) issues.push("expires_at must be later than reviewed_at.");
    if (expiresAt - reviewedAt > 7 * 24 * 60 * 60 * 1000) issues.push("Review validity cannot exceed seven days.");
  }
  return [...new Set(issues)];
}

export async function readOrientationReview(file, inspection, originRecordSha256 = null) {
  if (!file || typeof file.text !== "function" || file.size > MAX_REVIEW_JSON_BYTES) {
    throw new Error("Select a reviewer-signed orientation review JSON file smaller than 12 KB.");
  }
  let review;
  try {
    review = JSON.parse(await file.text());
  } catch {
    throw new Error("The orientation review file is not valid JSON.");
  }
  const issues = orientationReviewIssues(review, inspection, originRecordSha256);
  if (issues.length) throw new Error(`Orientation review rejected: ${issues.join(" ")}`);
  return review;
}

function encodeReviewHeader(review) {
  const bytes = new TextEncoder().encode(JSON.stringify(review));
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  const encoded = btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
  if (encoded.length > 16_384) throw new Error("The orientation review manifest exceeds the request header limit.");
  return encoded;
}

async function encodeJsonRecord(file, label) {
  if (!file || typeof file.arrayBuffer !== "function" || !String(file.name || "").toLowerCase().endsWith(".json")) {
    throw new Error(`Select the ${label} JSON file.`);
  }
  if (file.size <= 0 || file.size > MAX_PROVENANCE_RECORD_BYTES) {
    throw new Error(`${label} must be a non-empty JSON file no larger than 1 MiB.`);
  }
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  for (let index = 0; index < bytes.length; index += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(index, Math.min(index + 0x8000, bytes.length)));
  }
  return btoa(binary);
}

export async function createNiftiOriginRecord(
  inspection,
  datasetJsonFile,
  fetchImpl = fetch,
  apiUrl = DEFAULT_CT_API_URL,
) {
  if (!inspection?.attestation_token) throw new Error("Inspect this exact NIfTI volume before creating its origin record.");
  const response = await requestFailed(fetchImpl, `${apiUrl}/ct/nifti-origin/record`, {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      "X-Geometry-Token": inspection.attestation_token,
    },
    body: JSON.stringify({
      dataset_json_base64: await encodeJsonRecord(datasetJsonFile, "Task09 dataset.json"),
      image_path: TASK09_IMAGE_PATH,
    }),
  }, "Could not reach the local CT service to create the NIfTI-origin record.");
  const result = await parseResponse(response);
  const record = result.origin_record;
  if (result.status !== "nifti_origin_record_created" || result.provenance_type !== "nifti_origin" ||
      !/^[0-9a-f]{64}$/.test(result.origin_record_sha256 || "") ||
      record?.schema !== "medical-imaging-ct-nifti-origin/v1" ||
      record?.dataset?.name !== "Medical Segmentation Decathlon" ||
      record?.dataset?.task !== "Task09_Spleen" || record?.dataset?.release !== "1.0 06/08/2018" ||
      record?.dataset?.split !== "test" || record?.dataset?.image_path !== TASK09_IMAGE_PATH ||
      record?.volume?.sha256 !== inspection.file_sha256 ||
      record?.volume?.geometry_sha256 !== inspection.geometry_sha256 ||
      canonicalJson(record?.volume?.geometry) !== canonicalJson(inspection.geometry) ||
      record?.volume_source_authenticity !== "not_independently_authenticated" ||
      record?.original_dicom_metadata !== "not_supplied_in_the_dataset_distribution" ||
      record?.conversion_history !== "unavailable") {
    throw new Error("The CT service returned an invalid or mismatched NIfTI-origin record.");
  }
  return result;
}

export async function uploadNiftiOriginEvidence(
  inspection,
  orientationReview,
  datasetJsonFile,
  fetchImpl = fetch,
  apiUrl = DEFAULT_CT_API_URL,
) {
  if (!inspection?.attestation_token || orientationReview?.schema !== NIFTI_ORIGIN_REVIEW_SCHEMA) {
    throw new Error("Inspect this NIfTI volume and select its independent NIfTI-origin orientation review.");
  }
  if (orientationReview.volume_sha256 !== inspection.file_sha256 ||
      orientationReview.geometry_sha256 !== inspection.geometry_sha256 ||
      !/^[0-9a-f]{64}$/.test(orientationReview.origin_record_sha256 || "") ||
      !Number.isFinite(Date.parse(orientationReview.expires_at)) ||
      Date.parse(orientationReview.expires_at) <= Date.now()) {
    throw new Error("The NIfTI-origin review is stale or does not match this exact CT inspection and generated record.");
  }
  const response = await requestFailed(fetchImpl, `${apiUrl}/ct/nifti-origin/evidence`, {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      "X-Geometry-Token": inspection.attestation_token,
      "X-Orientation-Review": encodeReviewHeader(orientationReview),
    },
    body: JSON.stringify({
      dataset_json_base64: await encodeJsonRecord(datasetJsonFile, "Task09 dataset.json"),
      image_path: TASK09_IMAGE_PATH,
    }),
  }, "Could not reach the local CT service to validate NIfTI-origin evidence.");
  const result = await parseResponse(response);
  if (result.status !== "provenance_checked" || result.provenance_type !== "nifti_origin" ||
      result.origin_record_sha256 !== orientationReview.origin_record_sha256 ||
      typeof result.provenance_token !== "string" || !result.provenance_token ||
      !Number.isFinite(result.expires_at_unix) || result.expires_at_unix * 1000 <= Date.now() ||
      typeof result.warning !== "string") {
    throw new Error("The CT service did not explicitly confirm NIfTI-origin provenance checks.");
  }
  return result;
}

export async function uploadCtEvidence(
  inspection,
  orientationReview,
  sourceMetadataFile,
  conversionRecordFile,
  fetchImpl = fetch,
  apiUrl = DEFAULT_CT_API_URL,
) {
  if (!inspection?.attestation_token || !orientationReview) {
    throw new Error("Inspect the CT and load the independent reviewer-signed orientation review before uploading provenance.");
  }
  if (orientationReview.volume_sha256 !== inspection.file_sha256 ||
      orientationReview.geometry_sha256 !== inspection.geometry_sha256 ||
      Date.parse(orientationReview.expires_at) <= Date.now()) {
    throw new Error("The orientation review is stale or does not match this exact CT inspection.");
  }
  const payload = {
    source_metadata_base64: await encodeJsonRecord(sourceMetadataFile, "source DICOM metadata"),
    conversion_record_base64: await encodeJsonRecord(conversionRecordFile, "NIfTI conversion record"),
  };
  const body = JSON.stringify(payload);
  if (new TextEncoder().encode(body).byteLength > MAX_PROVENANCE_BODY_BYTES) {
    throw new Error("Combined provenance files exceed the 3 MiB request limit.");
  }
  const response = await requestFailed(fetchImpl, `${apiUrl}/ct/evidence`, {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      "X-Geometry-Token": inspection.attestation_token,
      "X-Orientation-Review": encodeReviewHeader(orientationReview),
    },
    body,
  }, "Could not reach the local CT service to validate source provenance.");
  const result = await parseResponse(response);
  if (result.status !== "provenance_checked" || typeof result.provenance_token !== "string" ||
      !result.provenance_token || !Number.isFinite(result.expires_at_unix) ||
      result.expires_at_unix * 1000 <= Date.now() || typeof result.warning !== "string") {
    throw new Error("The CT service did not explicitly confirm the provenance checks.");
  }
  return result;
}

async function gunzipBase64(value) {
  if (typeof value !== "string" || !/^[A-Za-z0-9+/]+={0,2}$/.test(value)) {
    throw new Error("The CT service returned malformed compressed slice data.");
  }
  if (typeof DecompressionStream !== "function") {
    throw new Error("This browser cannot decompress the local CT viewer response. Use a current Chrome or Edge version.");
  }
  let bytes;
  try {
    const binary = atob(value);
    bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  } catch {
    throw new Error("The CT service returned invalid compressed slice data.");
  }
}

export async function segmentCtVolume(
  file,
  inspection,
  orientationReview,
  provenanceToken,
  fetchImpl = fetch,
  apiUrl = DEFAULT_CT_API_URL,
) {
  const filename = niftiName(file);
  if (!inspection?.attestation_token || !orientationReview || !provenanceToken) {
    throw new Error("Inspect the CT, validate source provenance, and select a reviewer-signed orientation review before segmentation.");
  }
  if (orientationReview.volume_sha256 !== inspection.file_sha256 ||
      orientationReview.geometry_sha256 !== inspection.geometry_sha256 ||
      Date.parse(orientationReview.expires_at) <= Date.now()) {
    throw new Error("The orientation review is stale or does not match this exact CT inspection.");
  }
  const response = await requestFailed(fetchImpl, `${apiUrl}/ct/segment`, {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "application/octet-stream",
      "X-Volume-Filename": filename,
      "X-Geometry-Token": inspection.attestation_token,
      "X-Orientation-Review": encodeReviewHeader(orientationReview),
      "X-Provenance-Token": provenanceToken,
    },
    body: file,
  }, "Could not reach the local CT service. Check that the isolated segmentation API is running.");
  const payload = await parseResponse(response);
  if (payload.status !== "completed" || payload.model_status !== "pretrained" ||
      payload.input_sha256 !== inspection.file_sha256 ||
      (orientationReview.schema === NIFTI_ORIGIN_REVIEW_SCHEMA && payload.provenance_type !== "nifti_origin") ||
      (payload.provenance_type && orientationReview.schema !== NIFTI_ORIGIN_REVIEW_SCHEMA &&
       payload.provenance_type !== "dicom") ||
      !Array.isArray(payload.labels) || !payload.labels.includes(0) || !payload.labels.includes(1) ||
      payload.labels.some((label) => label !== 0 && label !== 1)) {
    throw new Error("The CT service response failed the pretrained-model or exact-file checks.");
  }
  const geometry = payload.geometry;
  const compatibility = assessCtMaskCompatibility(inspection.geometry, geometry, { reviewEvidenceAccepted: true });
  if (!compatibility.compatible) throw new Error(`CT mask overlay blocked: ${compatibility.reason}`);
  const expectedVoxels = geometry.shape.reduce((product, dimension) => product * dimension, 1);
  const [volumePreview, mask] = await Promise.all([
    gunzipBase64(payload.volume_preview_gzip_base64),
    gunzipBase64(payload.mask_gzip_base64),
  ]);
  if (volumePreview.length !== expectedVoxels || mask.length !== expectedVoxels ||
      mask.some((value) => value !== 0 && value !== 1)) {
    throw new Error("The CT service returned slice data inconsistent with the inspected volume dimensions.");
  }
  if (!mask.some((value) => value === 1)) {
    throw new Error("The CT service returned an empty spleen-label mask; no valid nonempty segmentation is available.");
  }
  return {
    geometry,
    volumePreview,
    mask,
    labels: [...new Set(mask)].sort(),
    httpStatus: Number.isInteger(response.status) ? response.status : null,
    status: payload.status,
    warning: payload.warning,
    model: payload.model,
  };
}
