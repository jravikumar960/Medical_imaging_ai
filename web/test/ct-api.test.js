import test from "node:test";
import assert from "node:assert/strict";
import { gzipSync } from "node:zlib";
import {
  createNiftiOriginRecord,
  inspectCtVolume,
  orientationReviewIssues,
  readOrientationReview,
  segmentCtVolume,
  uploadCtEvidence,
  uploadNiftiOriginEvidence,
} from "../src/ctApi.js";

const geometry = {
  shape: [2, 3, 4],
  affine: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]],
  spacing_mm: [1, 1, 2], axis_codes: ["R", "A", "S"],
  qform: null, sform: null, qform_code: 0, sform_code: 0,
};
const reviewedAt = new Date(Date.now() - 60_000).toISOString();
const expiresAt = new Date(Date.now() + 6 * 24 * 60 * 60 * 1000).toISOString();
const file = new Blob(["generated synthetic NIfTI bytes"], { type: "application/octet-stream" });
file.name = "synthetic_ct.nii.gz";
const inspection = {
  status: "inspection_only", file_sha256: "a".repeat(64), geometry_sha256: "f".repeat(64), file_bytes: 32,
  attestation_token: "signed-test-token", attestation_required: true, orientation_verified: false, model_loaded: false, geometry,
};
const review = {
  schema: "medical-imaging-ct-orientation-review/v1",
  volume_sha256: inspection.file_sha256,
  geometry_sha256: inspection.geometry_sha256,
  series_instance_uid: "1.2.840.10008.1.2.3",
  frame_of_reference_uid: "1.2.840.10008.1.2.4",
  source_metadata_sha256: "b".repeat(64), conversion_record_sha256: "c".repeat(64),
  image_orientation_patient: [1, 0, 0, 0, 1, 0],
  first_image_position_patient: [0, 0, 0], last_image_position_patient: [0, 0, 10],
  slice_order: "ascending", nifti_axis_codes: geometry.axis_codes,
  conversion_software: "ExampleConverter", conversion_version: "1.0",
  reviewer_id: "reviewer-demo", laterality_verified: true, slice_direction_verified: true,
  reviewed_at: reviewedAt, expires_at: expiresAt,
  review_notes: "Synthetic frontend contract data only; not anatomical evidence.",
  signature_hmac_sha256: "d".repeat(64),
};

test("CT inspection sends the NIfTI bytes to the separate endpoint and validates its preflight response", async () => {
  let request;
  const result = await inspectCtVolume(file, async (url, options) => {
    request = { url, options };
    return { ok: true, json: async () => inspection };
  });
  assert.equal(request.url, "http://127.0.0.1:8002/ct/inspect");
  assert.equal(request.options.method, "POST");
  assert.equal(request.options.credentials, "include");
  assert.equal(request.options.headers["X-Volume-Filename"], file.name);
  assert.equal(request.options.body, file);
  assert.equal(result.attestation_token, "signed-test-token");
  assert.equal(result.orientation_verified, false);
});

test("CT evidence upload sends both bounded JSON records and requires API validation", async () => {
  const sourceMetadata = new Blob(['{"schema":"synthetic-source"}'], { type: "application/json" });
  sourceMetadata.name = "source-metadata.json";
  const conversionRecord = new Blob(['{"schema":"synthetic-conversion"}'], { type: "application/json" });
  conversionRecord.name = "conversion-record.json";
  let request;
  const result = await uploadCtEvidence(
    inspection,
    review,
    sourceMetadata,
    conversionRecord,
    async (url, options) => {
      request = { url, options };
      return {
        ok: true,
        json: async () => ({
          status: "provenance_checked",
          provenance_token: "synthetic-one-use-token",
          expires_at_unix: Math.floor(Date.now() / 1000) + 60,
          warning: "Synthetic test only.",
        }),
      };
    },
    "http://127.0.0.1:8002",
  );
  assert.equal(request.url, "http://127.0.0.1:8002/ct/evidence");
  assert.equal(request.options.credentials, "include");
  assert.equal(request.options.headers["X-Geometry-Token"], inspection.attestation_token);
  assert.ok(request.options.headers["X-Orientation-Review"]);
  const payload = JSON.parse(request.options.body);
  assert.equal(Buffer.from(payload.source_metadata_base64, "base64").toString(), '{"schema":"synthetic-source"}');
  assert.equal(Buffer.from(payload.conversion_record_base64, "base64").toString(), '{"schema":"synthetic-conversion"}');
  assert.equal(result.provenance_token, "synthetic-one-use-token");

  await assert.rejects(uploadCtEvidence(
    inspection, review, sourceMetadata, conversionRecord,
    async () => ({ ok: true, json: async () => ({ status: "inspection_only" }) }),
  ), /did not explicitly confirm/);
});

test("CT evidence upload refuses missing or oversized files before making a request", async () => {
  let calls = 0;
  const fetchMock = async () => {
    calls += 1;
    return { ok: true, json: async () => ({}) };
  };
  const sourceMetadata = new Blob(["{}"], { type: "application/json" });
  sourceMetadata.name = "metadata.json";
  const oversized = new Blob([new Uint8Array(1024 * 1024 + 1)], { type: "application/json" });
  oversized.name = "conversion.json";
  await assert.rejects(uploadCtEvidence(inspection, review, null, oversized, fetchMock), /Select the source DICOM metadata/);
  await assert.rejects(uploadCtEvidence(inspection, review, sourceMetadata, oversized, fetchMock), /no larger than 1 MiB/);
  assert.equal(calls, 0);
});

const niftiOriginRecord = {
  schema: "medical-imaging-ct-nifti-origin/v1",
  origin_type: "nifti_dataset_distribution",
  dataset: {
    name: "Medical Segmentation Decathlon", task: "Task09_Spleen",
    release: "1.0 06/08/2018", split: "test",
    image_path: "./imagesTs/spleen_1.nii.gz",
  },
  distributed_representation: "NIfTI",
  volume_source_authenticity: "not_independently_authenticated",
  original_dicom_metadata: "not_supplied_in_the_dataset_distribution",
  conversion_history: "unavailable",
  volume: {
    sha256: inspection.file_sha256,
    geometry_sha256: inspection.geometry_sha256,
    geometry,
  },
};
const niftiOriginReview = {
  schema: "medical-imaging-nifti-origin-orientation-review/v1",
  origin_record_sha256: "e".repeat(64),
  volume_sha256: inspection.file_sha256,
  geometry_sha256: inspection.geometry_sha256,
  nifti_axis_codes: geometry.axis_codes,
  reviewer_id: "independent-reviewer",
  reviewer_independent: true,
  laterality_verified: true,
  slice_direction_verified: true,
  physical_slice_direction: "inferior_to_superior",
  orientation_reference: "Human-reviewed authoritative reference",
  orientation_method: "Independent visual/anatomical comparison",
  reviewed_at: reviewedAt,
  expires_at: expiresAt,
  review_notes: "Human review required; synthetic test manifest only.",
  signature_hmac_sha256: "d".repeat(64),
};

test("Task09 NIfTI-origin record is generated by the API and checked against the exact inspection", async () => {
  const dataset = new Blob(['{"name":"Spleen"}'], { type: "application/json" });
  dataset.name = "dataset.json";
  let request;
  const responseRecord = {
    status: "nifti_origin_record_created",
    provenance_type: "nifti_origin",
    origin_record: niftiOriginRecord,
    origin_record_sha256: "e".repeat(64),
  };
  const result = await createNiftiOriginRecord(inspection, dataset, async (url, options) => {
    request = { url, options };
    return { ok: true, json: async () => responseRecord };
  }, "http://127.0.0.1:8002");
  assert.equal(request.url, "http://127.0.0.1:8002/ct/nifti-origin/record");
  assert.equal(request.options.credentials, "include");
  assert.equal(request.options.headers["X-Geometry-Token"], inspection.attestation_token);
  const payload = JSON.parse(request.options.body);
  assert.equal(Buffer.from(payload.dataset_json_base64, "base64").toString(), '{"name":"Spleen"}');
  assert.equal(payload.image_path, "./imagesTs/spleen_1.nii.gz");
  assert.equal(result.origin_record_sha256, "e".repeat(64));
  await assert.rejects(createNiftiOriginRecord(inspection, dataset, async () => ({
    ok: true,
    json: async () => ({ ...responseRecord, origin_record: { ...niftiOriginRecord, volume: { ...niftiOriginRecord.volume, sha256: "0".repeat(64) } } }),
  })), /invalid or mismatched/);
});

test("NIfTI-origin review must bind the generated record, file and geometry", async () => {
  const manifest = new Blob([JSON.stringify(niftiOriginReview)]);
  manifest.name = "nifti-origin-review.json";
  assert.equal((await readOrientationReview(manifest, inspection, niftiOriginReview.origin_record_sha256)).reviewer_id,
    "independent-reviewer");
  await assert.rejects(readOrientationReview(manifest, inspection, "0".repeat(64)), /origin_record_sha256 mismatch/);
  await assert.rejects(readOrientationReview(new Blob([JSON.stringify({
    ...niftiOriginReview, reviewer_independent: false,
  })]), inspection, niftiOriginReview.origin_record_sha256), /independent/);
});

test("review diagnostics identify incomplete and mismatched NIfTI-origin fields", () => {
  const incomplete = {
    ...niftiOriginReview,
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
  const issues = orientationReviewIssues(incomplete, inspection, niftiOriginReview.origin_record_sha256);
  for (const field of [
    "reviewer_id", "reviewer_independent", "laterality_verified", "slice_direction_verified",
    "physical_slice_direction", "orientation_reference", "orientation_method",
    "reviewed_at", "expires_at", "review_notes", "signature_hmac_sha256",
  ]) {
    assert.ok(issues.some((issue) => issue.includes(field)), `Expected diagnostic for ${field}`);
  }
  const mismatched = orientationReviewIssues({
    ...niftiOriginReview,
    volume_sha256: "0".repeat(64),
    geometry_sha256: "1".repeat(64),
    origin_record_sha256: "2".repeat(64),
    nifti_axis_codes: ["L", "P", "I"],
  }, inspection, niftiOriginReview.origin_record_sha256);
  for (const field of ["volume_sha256", "geometry_sha256", "origin_record_sha256", "nifti_axis_codes"]) {
    assert.ok(mismatched.some((issue) => issue.includes(field)), `Expected mismatch diagnostic for ${field}`);
  }
});

test("NIfTI-origin evidence uses its distinct endpoint and requires server validation", async () => {
  const dataset = new Blob(["{}"], { type: "application/json" });
  dataset.name = "dataset.json";
  let request;
  const result = await uploadNiftiOriginEvidence(
    inspection,
    niftiOriginReview,
    dataset,
    async (url, options) => {
      request = { url, options };
      return {
        ok: true,
        json: async () => ({
          status: "provenance_checked",
          provenance_type: "nifti_origin",
          provenance_token: "nifti-origin-one-use-token",
          origin_record_sha256: niftiOriginReview.origin_record_sha256,
          expires_at_unix: Math.floor(Date.now() / 1000) + 60,
          warning: "Synthetic only.",
        }),
      };
    },
    "http://127.0.0.1:8002",
  );
  assert.equal(request.url, "http://127.0.0.1:8002/ct/nifti-origin/evidence");
  assert.equal(request.options.credentials, "include");
  assert.ok(request.options.headers["X-Orientation-Review"]);
  assert.equal(result.provenance_token, "nifti-origin-one-use-token");
  await assert.rejects(uploadNiftiOriginEvidence(
    inspection, niftiOriginReview, dataset,
    async () => ({ ok: true, json: async () => ({ status: "inspection_only" }) }),
  ), /did not explicitly confirm NIfTI-origin/);
  await assert.rejects(uploadNiftiOriginEvidence(
    inspection, niftiOriginReview, dataset,
    async () => ({
      ok: true,
      json: async () => ({
        status: "provenance_checked",
        provenance_type: "nifti_origin",
        provenance_token: "expired-token",
        origin_record_sha256: niftiOriginReview.origin_record_sha256,
        expires_at_unix: Math.floor(Date.now() / 1000) - 1,
        warning: "Synthetic only.",
      }),
    }),
  ), /did not explicitly confirm NIfTI-origin/);
});

test("review manifest must be signed, unexpired, and bound to the inspected file and geometry", async () => {
  const manifestFile = new Blob([JSON.stringify(review)], { type: "application/json" });
  manifestFile.name = "orientation-review.json";
  const result = await readOrientationReview(manifestFile, inspection);
  assert.equal(result.reviewer_id, "reviewer-demo");
  await assert.rejects(readOrientationReview(manifestFile, { ...inspection, file_sha256: "0".repeat(64) }), /volume_sha256 mismatch/);
  await assert.rejects(readOrientationReview(manifestFile, { ...inspection, geometry_sha256: "0".repeat(64) }), /geometry_sha256 mismatch/);
  await assert.rejects(readOrientationReview(new Blob(["{}"]), inspection), /schema/);
  const expired = { ...review, expires_at: "2000-01-01T00:00:00Z" };
  const expiredFile = new Blob([JSON.stringify(expired)]);
  await assert.rejects(readOrientationReview(expiredFile, inspection), /expired/);
});

test("CT result is parsed only with exact-file, matching geometry, and a signed review artifact", async () => {
  // Mocked response contract only: these generated bytes/mask do not represent inference.
  let request;
  const preview = Uint8Array.from({ length: 24 }, (_, index) => index);
  const mask = Uint8Array.from({ length: 24 }, (_, index) => index % 2);
  const payload = {
    status: "completed", model: "MONAI spleen_ct_segmentation", model_status: "pretrained",
    input_sha256: inspection.file_sha256, labels: [0, 1], geometry,
    volume_preview_gzip_base64: gzipSync(preview).toString("base64"),
    mask_gzip_base64: gzipSync(mask).toString("base64"), warning: "Research demo",
  };
  const result = await segmentCtVolume(file, inspection, review, "synthetic-provenance-token", async (url, options) => {
    request = { url, options };
    return { ok: true, status: 200, json: async () => payload };
  }, "http://127.0.0.1:8002");
  assert.equal(request.url, "http://127.0.0.1:8002/ct/segment");
  assert.equal(request.options.credentials, "include");
  assert.equal(request.options.headers["X-Geometry-Token"], inspection.attestation_token);
  assert.ok(request.options.headers["X-Orientation-Review"]);
  assert.equal(request.options.headers["X-Provenance-Token"], "synthetic-provenance-token");
  assert.deepEqual(result.geometry.shape, inspection.geometry.shape);
  assert.equal(result.volumePreview.length, inspection.geometry.shape.reduce((a, b) => a * b, 1));
  assert.equal(result.mask.length, inspection.geometry.shape.reduce((a, b) => a * b, 1));
  assert.deepEqual(result.volumePreview, preview);
  assert.deepEqual(result.mask, mask);
  assert.deepEqual(result.labels, [0, 1]);
  assert.equal(result.httpStatus, 200);
  assert.equal(result.status, "completed");
});

test("CT result rejects empty masks and invalid output labels", async () => {
  const payload = {
    status: "completed", model: "MONAI spleen_ct_segmentation", model_status: "pretrained",
    input_sha256: inspection.file_sha256, labels: [0, 1], geometry,
    volume_preview_gzip_base64: gzipSync(new Uint8Array(24)).toString("base64"),
    mask_gzip_base64: gzipSync(new Uint8Array(24)).toString("base64"),
  };
  await assert.rejects(segmentCtVolume(file, inspection, review, "synthetic-token",
    async () => ({ ok: true, json: async () => payload })), /empty spleen-label mask/);

  await assert.rejects(segmentCtVolume(file, inspection, review, "synthetic-token",
    async () => ({ ok: true, json: async () => ({
      ...payload,
      labels: [0, 2],
      mask_gzip_base64: gzipSync(Uint8Array.from({ length: 24 }, (_, index) => index % 2)).toString("base64"),
    }) })), /pretrained-model or exact-file checks/);
});

test("synthetic or untrained segmentation response is never returned as a pretrained result", async () => {
  const payload = {
    status: "completed", model: "synthetic fixture", model_status: "synthetic",
    input_sha256: inspection.file_sha256, labels: [0, 1], geometry,
    volume_preview_gzip_base64: "unused", mask_gzip_base64: "unused",
  };
  await assert.rejects(segmentCtVolume(file, inspection, review, "synthetic-provenance-token",
    async () => ({ ok: true, json: async () => payload })), /pretrained-model or exact-file checks/);
});

test("CT request blocks missing, stale, and mismatched review evidence", async () => {
  let calls = 0;
  const fetchMock = async () => {
    calls += 1;
    return { ok: true, json: async () => ({}) };
  };
  await assert.rejects(segmentCtVolume(file, inspection, null, null, fetchMock), /orientation review/);
  await assert.rejects(segmentCtVolume(file, inspection, { ...review, expires_at: "2000-01-01T00:00:00Z" }, "synthetic-token", fetchMock), /stale or does not match/);
  assert.equal(calls, 0);

  const wrongHash = { status: "completed", model_status: "pretrained", input_sha256: "b".repeat(64), labels: [0, 1], geometry };
  await assert.rejects(segmentCtVolume(file, inspection, review, "synthetic-provenance-token",
    async () => ({ ok: true, json: async () => wrongHash })), /exact-file checks/);

  const wrongGeometry = { ...wrongHash, input_sha256: inspection.file_sha256, geometry: { ...geometry, shape: [2, 3, 5] } };
  await assert.rejects(segmentCtVolume(file, inspection, review, "synthetic-provenance-token",
    async () => ({ ok: true, json: async () => wrongGeometry })), /dimensions do not match/);
});

test("CT helper surfaces malformed input, server rejection, and service-unavailable errors", async () => {
  const unsupported = new Blob(["not nifti"]);
  unsupported.name = "scan.dcm";
  await assert.rejects(inspectCtVolume(unsupported, fetch), /3D NIfTI volume/);
  await assert.rejects(inspectCtVolume(file, async () => ({
    ok: false, json: async () => ({ detail: "CT qform and sform disagree." }),
  })), /qform and sform disagree/);
  await assert.rejects(inspectCtVolume(file, async () => { throw new Error("offline"); }), /segmentation API/);
});
