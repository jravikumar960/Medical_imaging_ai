/** Compose one native voxel-axis slice. This function does not reorient anatomy. */
export function composeCtSlice({ shape, volumePreview, mask, sliceIndex }) {
  if (!Array.isArray(shape) || shape.length !== 3 || shape.some((value) => !Number.isInteger(value) || value <= 0)) {
    throw new Error("CT slice dimensions must be three positive integers.");
  }
  const [width, height, depth] = shape;
  const voxelsPerSlice = width * height;
  const expected = voxelsPerSlice * depth;
  if (volumePreview?.length !== expected || (mask !== null && mask?.length !== expected)) {
    throw new Error("Slice data does not match the CT dimensions.");
  }
  if (!Number.isInteger(sliceIndex) || sliceIndex < 0 || sliceIndex >= depth) {
    throw new Error("CT slice index is outside the volume.");
  }
  const data = new Uint8ClampedArray(expected / depth * 4);
  const offset = sliceIndex * voxelsPerSlice;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const sourceIndex = offset + y * width + x;
      const targetIndex = (y * width + x) * 4;
      const gray = volumePreview[sourceIndex];
      if (mask?.[sourceIndex] === 1) {
        data[targetIndex] = 255;
        data[targetIndex + 1] = 56;
        data[targetIndex + 2] = 48;
      } else {
        data[targetIndex] = gray;
        data[targetIndex + 1] = gray;
        data[targetIndex + 2] = gray;
      }
      data[targetIndex + 3] = 255;
    }
  }
  return { width, height, data };
}
/** Compose a research slice with prediction-vs-GT TP/FP/FN colour coding. */
export function composeCtResearchSlice({ shape, volumePreview, predictionMask, referenceMask, sliceIndex }) {
  if (!Array.isArray(shape) || shape.length !== 3 || shape.some((value) => !Number.isInteger(value) || value <= 0)) {
    throw new Error("CT slice dimensions must be three positive integers.");
  }
  const [width, height, depth] = shape;
  const voxelsPerSlice = width * height;
  const expected = voxelsPerSlice * depth;
  if (volumePreview?.length !== expected) {
    throw new Error("Slice data does not match the CT dimensions.");
  }
  if (predictionMask !== null && predictionMask?.length !== expected) {
    throw new Error("Prediction mask does not match the CT dimensions.");
  }
  if (referenceMask !== null && referenceMask?.length !== expected) {
    throw new Error("Reference mask does not match the CT dimensions.");
  }
  if (!Number.isInteger(sliceIndex) || sliceIndex < 0 || sliceIndex >= depth) {
    throw new Error("CT slice index is outside the volume.");
  }
  const data = new Uint8ClampedArray(voxelsPerSlice * 4);
  const offset = sliceIndex * voxelsPerSlice;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const sourceIndex = offset + y * width + x;
      const targetIndex = (y * width + x) * 4;
      const gray = volumePreview[sourceIndex];
      const pred = predictionMask?.[sourceIndex] === 1;
      const ref = referenceMask?.[sourceIndex] === 1;
      if (pred && ref) {
        data[targetIndex] = 34;
        data[targetIndex + 1] = 197;
        data[targetIndex + 2] = 94;
      } else if (pred && !ref) {
        data[targetIndex] = 239;
        data[targetIndex + 1] = 68;
        data[targetIndex + 2] = 68;
      } else if (!pred && ref) {
        data[targetIndex] = 59;
        data[targetIndex + 1] = 130;
        data[targetIndex + 2] = 246;
      } else {
        data[targetIndex] = gray;
        data[targetIndex + 1] = gray;
        data[targetIndex + 2] = gray;
      }
      data[targetIndex + 3] = 255;
    }
  }
  return { width, height, data };
}
