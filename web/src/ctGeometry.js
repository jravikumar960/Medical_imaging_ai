const IDENTITY_BOTTOM_ROW = [0, 0, 0, 1];

function validGeometry(metadata) {
  const shape = metadata?.shape;
  const affine = metadata?.affine;
  if (!Array.isArray(shape) || shape.length !== 3 ||
      shape.some((dimension) => !Number.isInteger(dimension) || dimension <= 0)) return false;
  if (!Array.isArray(affine) || affine.length !== 4 ||
      affine.some((row) => !Array.isArray(row) || row.length !== 4 ||
        row.some((value) => typeof value !== "number" || !Number.isFinite(value)))) return false;
  if (!affine[3].every((value, index) => Math.abs(value - IDENTITY_BOTTOM_ROW[index]) <= 1e-6)) return false;
  const [a, b, c] = affine;
  const determinant = a[0] * (b[1] * c[2] - b[2] * c[1]) -
    a[1] * (b[0] * c[2] - b[2] * c[0]) +
    a[2] * (b[0] * c[1] - b[1] * c[0]);
  return Number.isFinite(determinant) && Math.abs(determinant) > 1e-8;
}

/**
 * Check whether a CT volume and mask may share a voxel overlay after the
 * service accepted a signed review artifact. This compares numeric geometry
 * only; it does not independently verify anatomical orientation.
 */
export function assessCtMaskCompatibility(volume, mask, { reviewEvidenceAccepted = false, tolerance = 1e-5 } = {}) {
  if (!reviewEvidenceAccepted) {
    return { compatible: false, reason: "A signed volume-specific orientation review must be accepted first." };
  }
  if (!validGeometry(volume) || !validGeometry(mask)) {
    return { compatible: false, reason: "Both files need valid 3D dimensions and finite, non-singular 4×4 affines." };
  }
  if (volume.shape.some((dimension, index) => dimension !== mask.shape[index])) {
    return { compatible: false, reason: "CT and mask dimensions do not match." };
  }
  const affineMatches = volume.affine.every((row, rowIndex) =>
    row.every((value, columnIndex) => Math.abs(value - mask.affine[rowIndex][columnIndex]) <= tolerance));
  if (!affineMatches) {
    return { compatible: false, reason: "CT and mask affines do not match within tolerance." };
  }
  return { compatible: true, reason: "Numeric geometry matches; signature acceptance does not independently prove anatomical orientation." };
}
