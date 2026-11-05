const REVIEW_SCHEMA = "medical-imaging-nifti-origin-orientation-review/v1";
const SHA256_PATTERN = /^[0-9a-f]{64}$/;

const AUTHORIZATION_PRESENTATIONS = {
  awaiting_review: {
    label: "Awaiting independent review",
    detail: "Inspect the volume and generate its NIfTI-origin record before requesting reviewer evidence.",
    badge: "neutral",
  },
  review_submitted: {
    label: "Review submitted",
    detail: "The reviewer file is loaded; the CT backend must validate its signature and exact-file bindings.",
    badge: "neutral",
  },
  authorization_validated: {
    label: "Authorization validated",
    detail: "The CT backend accepted the signed review for this inspection. Segmentation is enabled until the one-use token expires.",
    badge: "success",
  },
  segmentation_available: {
    label: "Segmentation available",
    detail: "Authorized pretrained inference completed and the returned mask passed output validation.",
    badge: "success",
  },
  review_rejected: {
    label: "Review rejected or expired",
    detail: "The CT backend did not accept current reviewer evidence. Segmentation is blocked.",
    badge: "danger",
  },
};

export function getCtAuthorizationPresentation(state) {
  const presentation = AUTHORIZATION_PRESENTATIONS[state];
  if (!presentation) throw new RangeError(`Unknown CT authorization state: ${state}`);
  return presentation;
}

export function createNiftiOriginReviewTemplate(inspection, originRecordSha256) {
  const volumeSha256 = inspection?.file_sha256;
  const geometrySha256 = inspection?.geometry_sha256;
  const axisCodes = inspection?.geometry?.axis_codes;
  if (!SHA256_PATTERN.test(volumeSha256 || "") ||
      !SHA256_PATTERN.test(geometrySha256 || "") ||
      !SHA256_PATTERN.test(originRecordSha256 || "") ||
      !Array.isArray(axisCodes) || axisCodes.length !== 3 ||
      axisCodes.some((code) => typeof code !== "string" || !code)) {
    throw new Error("A valid inspection, geometry, and generated origin-record hash are required to prepare the reviewer template.");
  }
  return {
    schema: REVIEW_SCHEMA,
    origin_record_sha256: originRecordSha256,
    volume_sha256: volumeSha256,
    geometry_sha256: geometrySha256,
    nifti_axis_codes: axisCodes,
    reviewer_id: "",
    reviewer_independent: false,
    laterality_verified: false,
    slice_direction_verified: false,
    physical_slice_direction: "",
    orientation_reference: "",
    orientation_method: "",
    reviewed_at: "",
    expires_at: "",
    review_notes: "",
    signature_hmac_sha256: "",
  };
}
