import test from "node:test";
import assert from "node:assert/strict";
import { composeMriSlice } from "../src/mriSlice.js";

test("MRI slice compositor renders grayscale and distinct predicted label colors", () => {
  const slice = composeMriSlice({
    shape: [2, 2, 2],
    volumePreview: new Uint8Array([30, 40, 50, 60, 70, 80, 90, 100]),
    mask: new Uint8Array([0, 1, 2, 4, 0, 0, 0, 0]),
    sliceIndex: 0,
  });

  assert.equal(slice.width, 2);
  assert.equal(slice.height, 2);
  assert.deepEqual([...slice.data.slice(0, 4)], [30, 30, 30, 255]);
  assert.deepEqual([...slice.data.slice(4, 8)], [255, 65, 54, 255]);
  assert.deepEqual([...slice.data.slice(8, 12)], [255, 190, 35, 255]);
  assert.deepEqual([...slice.data.slice(12, 16)], [235, 60, 211, 255]);
});

test("MRI slice compositor validates dimensions and slice bounds", () => {
  assert.throws(() => composeMriSlice({
    shape: [2, 2, 2],
    volumePreview: new Uint8Array(4),
    mask: new Uint8Array(8),
    sliceIndex: 0,
  }), /does not match/);
  assert.throws(() => composeMriSlice({
    shape: [2, 2, 2],
    volumePreview: new Uint8Array(8),
    mask: new Uint8Array(8),
    sliceIndex: 2,
  }), /outside/);
});
test("MRI slice compositor renders the second native slice with correct voxel order", () => {
  const slice = composeMriSlice({
    shape: [2, 2, 2],
    volumePreview: new Uint8Array([10, 20, 30, 40, 50, 60, 70, 80]),
    mask: new Uint8Array([0, 0, 0, 0, 0, 4, 1, 2]),
    sliceIndex: 1,
  });

  assert.equal(slice.width, 2);
  assert.equal(slice.height, 2);
  assert.deepEqual([...slice.data.slice(0, 4)], [50, 50, 50, 255]);
  assert.deepEqual([...slice.data.slice(4, 8)], [235, 60, 211, 255]);
  assert.deepEqual([...slice.data.slice(8, 12)], [255, 65, 54, 255]);
  assert.deepEqual([...slice.data.slice(12, 16)], [255, 190, 35, 255]);
});
