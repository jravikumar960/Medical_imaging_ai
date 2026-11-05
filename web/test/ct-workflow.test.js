import test from "node:test";
import assert from "node:assert/strict";
import {
  createNiftiOriginReviewTemplate,
  getCtAuthorizationPresentation,
} from "../src/ctWorkflow.js";

test("CT authorization workflow exposes all required states", () => {
  assert.equal(getCtAuthorizationPresentation("awaiting_review").label, "Awaiting independent review");
  assert.equal(getCtAuthorizationPresentation("review_submitted").label, "Review submitted");
  assert.equal(getCtAuthorizationPresentation("authorization_validated").label, "Authorization validated");
  assert.equal(getCtAuthorizationPresentation("segmentation_available").label, "Segmentation available");
  assert.equal(getCtAuthorizationPresentation("review_rejected").label, "Review rejected or expired");
  assert.throws(() => getCtAuthorizationPresentation("approved_locally"), /Unknown CT authorization state/);
});

test("review template binds to supplied live hashes and remains unsigned", () => {
  const inspection = {
    file_sha256: "a".repeat(64),
    geometry_sha256: "b".repeat(64),
    geometry: { axis_codes: ["R", "A", "S"] },
  };
  const template = createNiftiOriginReviewTemplate(inspection, "c".repeat(64));
  assert.equal(template.schema, "medical-imaging-nifti-origin-orientation-review/v1");
  assert.equal(template.volume_sha256, inspection.file_sha256);
  assert.equal(template.geometry_sha256, inspection.geometry_sha256);
  assert.equal(template.origin_record_sha256, "c".repeat(64));
  assert.deepEqual(template.nifti_axis_codes, ["R", "A", "S"]);
  assert.equal(template.reviewer_independent, false);
  assert.equal(template.laterality_verified, false);
  assert.equal(template.slice_direction_verified, false);
  assert.equal(template.signature_hmac_sha256, "");
  assert.throws(() => createNiftiOriginReviewTemplate(inspection, ""), /valid inspection/);
});
