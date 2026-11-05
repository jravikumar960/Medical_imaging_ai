import test from "node:test";
import assert from "node:assert/strict";
import { composeCtSlice } from "../src/ctSlice.js";

test("synthetic CT navigation composes source-grid slices and overlays only mask label 1", () => {
  const volumePreview = Uint8Array.from([10, 20, 30, 40, 50, 60, 70, 80]);
  const mask = Uint8Array.from([0, 1, 0, 0, 0, 0, 1, 0]);
  const firstSlice = composeCtSlice({
    shape: [2, 2, 2], volumePreview, mask, sliceIndex: 0,
  });
  assert.equal(firstSlice.width, 2);
  assert.equal(firstSlice.height, 2);
  assert.deepEqual([...firstSlice.data], [
    10, 10, 10, 255,
    255, 56, 48, 255,
    30, 30, 30, 255,
    40, 40, 40, 255,
  ]);

  const result = composeCtSlice({
    shape: [2, 2, 2],
    volumePreview,
    mask,
    sliceIndex: 1,
  });
  assert.equal(result.width, 2);
  assert.equal(result.height, 2);
  assert.deepEqual([...result.data], [
    50, 50, 50, 255,
    60, 60, 60, 255,
    255, 56, 48, 255,
    80, 80, 80, 255,
  ]);
});

test("viewer rejects inconsistent dimensions, slice indices, and malformed shapes", () => {
  const base = { shape: [2, 2, 2], volumePreview: new Uint8Array(8), mask: new Uint8Array(8), sliceIndex: 0 };
  assert.throws(() => composeCtSlice({ ...base, mask: new Uint8Array(7) }), /does not match/);
  assert.throws(() => composeCtSlice({ ...base, sliceIndex: 2 }), /outside the volume/);
  assert.throws(() => composeCtSlice({ ...base, shape: [2, 0, 2] }), /positive integers/);
});

test("CT source-only rendering preserves grayscale when no predicted mask is supplied", () => {
  const result = composeCtSlice({
    shape: [2, 1, 1],
    volumePreview: Uint8Array.from([18, 230]),
    mask: null,
    sliceIndex: 0,
  });
  assert.deepEqual([...result.data], [
    18, 18, 18, 255,
    230, 230, 230, 255,
  ]);
});
