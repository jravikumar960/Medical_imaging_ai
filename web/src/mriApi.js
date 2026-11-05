const API_BASE = (import.meta.env.VITE_MRI_API_URL || "http://127.0.0.1:8003").replace(/\/+$/, "");
const MAX_FILE_BYTES = 128 * 1024 * 1024;
const MAX_TOTAL_BYTES = 512 * 1024 * 1024;
const SEQUENCES = ["t1c", "t1", "t2", "flair"];
const MASK_LABELS = [0, 1, 2, 4];
const OUTPUT_CLASS_KEYS = ["1", "2", "4"];

function isNifti(file) {
  const name = file.name.toLowerCase();
  return name.endsWith(".nii") || name.endsWith(".nii.gz");
}

function decodeBase64(value) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

async function gunzipBase64(value) {
  if (typeof DecompressionStream !== "function") {
    throw new Error("This browser does not support decompressing MRI previews. Use a current browser.");
  }
  const compressed = decodeBase64(value);
  const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export async function segmentMri(files) {
  if (!files || SEQUENCES.some((sequence) => !(files[sequence] instanceof File))) {
    throw new Error("Select one NIfTI volume for each required sequence: T1c, T1, T2, and FLAIR.");
  }
  const selected = SEQUENCES.map((sequence) => [sequence, files[sequence]]);
  const totalBytes = selected.reduce((sum, [, file]) => sum + file.size, 0);
  for (const [sequence, file] of selected) {
    if (!isNifti(file)) {
      throw new Error(`${sequence.toUpperCase()} must be a .nii or .nii.gz file.`);
    }
    if (file.size === 0 || file.size > MAX_FILE_BYTES) {
      throw new Error(`${sequence.toUpperCase()} must be non-empty and no larger than 128 MiB.`);
    }
  }
  if (totalBytes > MAX_TOTAL_BYTES) {
    throw new Error("The four MRI volumes exceed the 512 MiB combined upload limit.");
  }

  const body = new FormData();
  for (const [sequence, file] of selected) body.append(sequence, file, file.name);

  let response;
  try {
    response = await fetch(`${API_BASE}/mri/segment`, { method: "POST", body });
  } catch {
    throw new Error("Could not reach the local MRI service at 127.0.0.1:8003.");
  }
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(payload?.detail || `MRI segmentation failed (HTTP ${response.status}).`);
  }
  if (payload?.status !== "completed" || !Array.isArray(payload.geometry?.shape)
      || !Array.isArray(payload.geometry?.spacing_mm) || typeof payload.study_id !== "string"
      || !/^[0-9a-f]{64}$/.test(payload.study_id)
      || !Array.isArray(payload.input_sequence_order)
      || payload.input_sequence_order.join(",") !== SEQUENCES.join(",")
      || !Array.isArray(payload.geometry.affine)
      || payload.geometry.affine.length !== 4
      || payload.geometry.affine.some((row) => !Array.isArray(row) || row.length !== 4
        || row.some((value) => !Number.isFinite(value)))
      || payload.geometry.spacing_mm.length !== 3
      || payload.geometry.spacing_mm.some((value) => !Number.isFinite(value) || value <= 0)) {
    throw new Error("The MRI service returned an incomplete segmentation response.");
  }
  const { shape } = payload.geometry;
  if (shape.length !== 3 || shape.some((dimension) => !Number.isInteger(dimension) || dimension <= 0)) {
    throw new Error("The MRI service returned invalid output dimensions.");
  }
  const outputClassKeys = Object.keys(payload.output_classes || {}).sort();
  if (outputClassKeys.join(",") !== OUTPUT_CLASS_KEYS.join(",")
      || OUTPUT_CLASS_KEYS.some((key) => typeof payload.output_classes[key] !== "string")) {
    throw new Error("The MRI service returned an incomplete output-label map.");
  }
  const [preview, mask] = await Promise.all([
    gunzipBase64(payload.flair_preview_gzip_base64),
    gunzipBase64(payload.mask_array_gzip_base64),
  ]);
  const expectedVoxels = shape.reduce((product, dimension) => product * dimension, 1);
  if (preview.length !== expectedVoxels || mask.length !== expectedVoxels) {
    throw new Error("The MRI service returned arrays that do not match its reported volume geometry.");
  }
  const actualCounts = Object.fromEntries(MASK_LABELS.map((label) => [label, 0]));
  for (const label of mask) {
    if (!MASK_LABELS.includes(label)) {
      throw new Error(`The MRI service returned an unsupported segmentation label: ${label}.`);
    }
    actualCounts[label] += 1;
  }
  const reportedCounts = payload.label_counts;
  if (!reportedCounts || MASK_LABELS.some((label) => (
    !Number.isSafeInteger(reportedCounts[String(label)])
    || reportedCounts[String(label)] !== actualCounts[label]
  ))) {
    throw new Error("The MRI service returned label counts that do not match its segmentation mask.");
  }
  const maskNifti = decodeBase64(payload.mask_nifti_gzip_base64);
  if (maskNifti.length === 0) {
    throw new Error("The MRI service returned an empty downloadable NIfTI mask.");
  }
  return {
    ...payload,
    volumePreview: preview,
    mask,
    maskNifti,
  };
}

export const MRI_SEQUENCES = SEQUENCES;
