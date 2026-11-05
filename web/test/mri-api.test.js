import test from "node:test";
import assert from "node:assert/strict";
import { File as NodeFile } from "node:buffer";
import { gzipSync } from "node:zlib";
import { createServer } from "vite";
import { fileURLToPath } from "node:url";

const webRoot = fileURLToPath(new URL("..", import.meta.url));

test("MRI client validates all four sequence files before network access", async () => {
  const vite = await createServer({
    root: webRoot,
    appType: "custom",
    logLevel: "silent",
    server: { middlewareMode: true },
  });
  const originalFetch = globalThis.fetch;
  globalThis.File ??= NodeFile;
  let requests = 0;
  globalThis.fetch = async () => {
    requests += 1;
    throw new Error("unexpected fetch");
  };
  try {
    const { segmentMri } = await vite.ssrLoadModule("/src/mriApi.js");
    await assert.rejects(segmentMri({}), /Select one NIfTI volume/);
    const invalidFiles = {
      t1c: new File(["x"], "t1c.png"),
      t1: new File(["x"], "t1.nii"),
      t2: new File(["x"], "t2.nii"),
      flair: new File(["x"], "flair.nii"),
    };
    await assert.rejects(segmentMri(invalidFiles), /T1C must be a .nii/);
    assert.equal(requests, 0);
  } finally {
    globalThis.fetch = originalFetch;
    await vite.close();
  }
});

test("MRI client posts four sequences and parses actual service arrays", async () => {
  const vite = await createServer({
    root: webRoot,
    appType: "custom",
    logLevel: "silent",
    server: { middlewareMode: true },
  });
  const originalFetch = globalThis.fetch;
  globalThis.File ??= NodeFile;
  let sentFields = [];
  const bytes = Buffer.from([0, 1, 2, 4]);
  const preview = gzipSync(Buffer.from([10, 20, 30, 40])).toString("base64");
  const mask = gzipSync(bytes).toString("base64");
  const nifti = Buffer.from([31, 139, 8]).toString("base64");
  const payload = {
    status: "completed",
    study_id: "a".repeat(64),
    model: "MONAI/brats_mri_segmentation",
    task: "synthetic API response",
    threshold: 0.5,
    input_sequence_order: ["t1c", "t1", "t2", "flair"],
    geometry: {
      shape: [2, 2, 1],
      affine: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      spacing_mm: [1, 1, 1],
    },
    output_classes: { "1": "TC", "2": "WT", "4": "ET" },
    label_counts: { "0": 1, "1": 1, "2": 1, "4": 1 },
    flair_preview_gzip_base64: preview,
    mask_array_gzip_base64: mask,
    mask_nifti_gzip_base64: nifti,
  };
  globalThis.fetch = async (_url, options) => {
    sentFields = [...options.body.keys()];
    return new Response(JSON.stringify(payload), { status: 200 });
  };
  try {
    const { segmentMri } = await vite.ssrLoadModule("/src/mriApi.js");
    const files = {
      t1c: new File(["t1c"], "t1c.nii"),
      t1: new File(["t1"], "t1.nii.gz"),
      t2: new File(["t2"], "t2.nii"),
      flair: new File(["flair"], "flair.nii.gz"),
    };
    const result = await segmentMri(files);

    assert.deepEqual(sentFields, ["t1c", "t1", "t2", "flair"]);
    assert.deepEqual([...result.volumePreview], [10, 20, 30, 40]);
    assert.deepEqual([...result.mask], [...bytes]);
    assert.deepEqual([...result.maskNifti], [31, 139, 8]);

    globalThis.fetch = async () => new Response(JSON.stringify({
      ...payload,
      mask_array_gzip_base64: gzipSync(Buffer.from([0, 1, 3, 4])).toString("base64"),
    }), { status: 200 });
    await assert.rejects(segmentMri(files), /unsupported segmentation label: 3/);

    globalThis.fetch = async () => new Response(JSON.stringify({
      ...payload,
      input_sequence_order: ["t1", "t1c", "t2", "flair"],
    }), { status: 200 });
    await assert.rejects(segmentMri(files), /incomplete segmentation response/);
  } finally {
    globalThis.fetch = originalFetch;
    await vite.close();
  }
});
