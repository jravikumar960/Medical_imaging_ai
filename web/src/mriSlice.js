/** Compose a native voxel-axis FLAIR slice with the predicted BraTS labels. */
export function composeMriSlice({ shape, volumePreview, mask, sliceIndex }) {
  if (!Array.isArray(shape) || shape.length !== 3 || shape.some((value) => !Number.isInteger(value) || value <= 0)) {
    throw new Error("MRI slice dimensions must be three positive integers.");
  }
  const [width, height, depth] = shape;
  const voxels = width * height * depth;
  if (volumePreview?.length !== voxels || (mask !== null && mask?.length !== voxels)) {
    throw new Error("MRI slice data does not match the reported dimensions.");
  }
  if (!Number.isInteger(sliceIndex) || sliceIndex < 0 || sliceIndex >= depth) {
    throw new Error("MRI slice index is outside the volume.");
  }

  const data = new Uint8ClampedArray(width * height * 4);
  const offset = sliceIndex * width * height;
  for (let index = 0; index < width * height; index += 1) {
    const sourceIndex = offset + index;
    const targetIndex = index * 4;
    const gray = volumePreview[sourceIndex];
    const label = mask?.[sourceIndex];
    if (label === 4) {
      data[targetIndex] = 235;
      data[targetIndex + 1] = 60;
      data[targetIndex + 2] = 211;
    } else if (label === 1) {
      data[targetIndex] = 255;
      data[targetIndex + 1] = 65;
      data[targetIndex + 2] = 54;
    } else if (label === 2) {
      data[targetIndex] = 255;
      data[targetIndex + 1] = 190;
      data[targetIndex + 2] = 35;
    } else {
      data[targetIndex] = gray;
      data[targetIndex + 1] = gray;
      data[targetIndex + 2] = gray;
    }
    data[targetIndex + 3] = 255;
  }
  return { width, height, data };
}
