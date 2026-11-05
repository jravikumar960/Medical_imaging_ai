import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { inferImage } from "../web/src/api.js";

const [baseUrl, rasterPath, dicomPath] = process.argv.slice(2);
assert.ok(baseUrl?.startsWith("http://127.0.0.1:"), "smoke must use loopback");
assert.ok(rasterPath, "generated car PNG fixture path is required");
assert.ok(dicomPath, "generated candidate DICOM fixture path is required");

const nativeFetch = globalThis.fetch;
globalThis.fetch = (input, options) => nativeFetch(new URL(input, baseUrl), options);

const healthBefore = await fetch(`${baseUrl}/health`).then((response) => response.json());
assert.equal(healthBefore.status, "ok");
assert.equal(healthBefore.model_loaded, false);
assert.equal(healthBefore.model_status, "not_loaded");

const rasterBytes = await readFile(rasterPath);
const raster = new Blob([rasterBytes], { type: "image/png" });
Object.defineProperty(raster, "name", { value: "generated-car.png" });
await assert.rejects(inferImage(raster), /modality verifier|no validated image-modality classifier/i);

const dicomBytes = await readFile(dicomPath);
const dicom = new Blob([dicomBytes], { type: "application/dicom" });
Object.defineProperty(dicom, "name", { value: "generated-candidate.dcm" });
await assert.rejects(inferImage(dicom), /modality verifier|no validated image-modality classifier/i);

for (const [bytes, contentType, explanation] of [
  [rasterBytes, "image/png", "PNG image"],
  [dicomBytes, "application/dicom", "metadata-qualified DICOM"],
]) {
  const response = await fetch(`${baseUrl}/infer`, {
    method: "POST",
    headers: { "Content-Type": contentType },
    body: bytes,
  });
  assert.equal(response.status, 422, `${explanation} should be rejected`);
  const payload = await response.json();
  assert.match(payload.detail, /modality verifier|no validated image-modality classifier/i);
  assert.equal("predictions" in payload, false);
  assert.equal("gradcam" in payload, false);
}

const healthAfter = await fetch(`${baseUrl}/health`).then((response) => response.json());
assert.equal(healthAfter.model_loaded, false);
assert.equal(healthAfter.model_status, "not_loaded");

console.log(JSON.stringify({
  integration: "React API helper and loopback FastAPI fail-closed modality gate",
  fixtures: "generated car illustration PNG and generated metadata-qualified DICOM; neither is medical data",
  frontend: "the API helper surfaces modality-verification rejections",
  api: "both direct requests returned HTTP 422 without predictions or Grad-CAM",
  model_loaded: false,
}));
