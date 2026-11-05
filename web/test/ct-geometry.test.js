import test from "node:test";
import assert from "node:assert/strict";
import { assessCtMaskCompatibility } from "../src/ctGeometry.js";

function syntheticGeometry(shape = [4, 5, 6], affine = [
  [1, 0, 0, -2],
  [0, 1, 0, -2],
  [0, 0, 2, -5],
  [0, 0, 0, 1],
]) {
  return { shape, affine };
}

test("synthetic CT and matching mask geometry permit overlay only after accepted signed review", () => {
  const geometry = syntheticGeometry();
  assert.equal(assessCtMaskCompatibility(geometry, geometry).compatible, false);
  const result = assessCtMaskCompatibility(geometry, geometry, { reviewEvidenceAccepted: true });
  assert.match(result.reason, /does not independently prove anatomical orientation/);
  assert.equal(result.compatible, true);
  assert.match(result.reason, /signature acceptance does not independently prove/);
});

test("synthetic dimension or affine mismatches reject mask overlay", () => {
  const volume = syntheticGeometry();
  assert.match(assessCtMaskCompatibility(volume, syntheticGeometry([4, 5, 7]), {
    reviewEvidenceAccepted: true,
  }).reason, /dimensions do not match/);
  const shifted = syntheticGeometry([4, 5, 6], [
    [1, 0, 0, -1], [0, 1, 0, -2], [0, 0, 2, -5], [0, 0, 0, 1],
  ]);
  assert.match(assessCtMaskCompatibility(volume, shifted, {
    reviewEvidenceAccepted: true,
  }).reason, /affines do not match/);
});

test("malformed and singular synthetic geometry is rejected", () => {
  const volume = syntheticGeometry();
  assert.equal(assessCtMaskCompatibility(volume, { shape: [4, 5], affine: [] }, {
    reviewEvidenceAccepted: true,
  }).compatible, false);
  const singular = syntheticGeometry([4, 5, 6], [
    [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1],
  ]);
  assert.equal(assessCtMaskCompatibility(singular, singular, {
    reviewEvidenceAccepted: true,
  }).compatible, false);
});
