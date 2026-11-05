import test from "node:test";
import assert from "node:assert/strict";
import { inferImage, rankPredictions } from "../src/api.js";

test("rankPredictions sorts scores without modifying source array", () => {
  const input = [
    { finding: "Atelectasis", score: 0.2 },
    { finding: "Pneumonia", score: 0.8 },
  ];
  assert.deepEqual(rankPredictions(input).map((entry) => entry.finding), ["Pneumonia", "Atelectasis"]);
  assert.equal(input[0].finding, "Atelectasis");
});

test("rankPredictions rejects non-finite and malformed scores", () => {
  assert.throws(() => rankPredictions([{ finding: "Pneumonia", score: NaN }]), /invalid scores/);
  assert.throws(() => rankPredictions([{}]), /invalid scores/);
});

test("inferImage sends supported images to the backend and surfaces backend errors", async () => {
  const originalFetch = globalThis.fetch;
  const requests = [];
  globalThis.fetch = async (...args) => {
    requests.push(args);
    return new Response(
      JSON.stringify({
        detail:
          "Chest X-ray finding inference is disabled for PNG/JPEG uploads because no " +
          "validated image-modality classifier is available. File format cannot establish " +
          "that pixels depict a chest X-ray; no scores or Grad-CAM were generated.",
      }),
      { status: 422, headers: { "Content-Type": "application/json" } },
    );
  };
  const cases = [
    ["car-photo.png", "image/png"],
    ["car-photo.jpg", "image/jpeg"],
    ["scan.dcm", "application/dicom"],
  ];
  try {
    for (const [name, type] of cases) {
      const file = new Blob(["synthetic non-medical image"], { type });
      file.name = name;
      await assert.rejects(inferImage(file), /no validated image-modality classifier/i);
    }
  } finally {
    globalThis.fetch = originalFetch;
  }
  assert.equal(requests.length, cases.length);
});

test("inferImage rejects unsupported formats without making a request", async () => {
  const file = new Blob(["text"], { type: "text/plain" });
  file.name = "notes.txt";
  await assert.rejects(inferImage(file), /Choose a PNG, JPEG, or DICOM image/);
});

test("inferImage with valid targetFinding appends encoded query parameter and returns response", async () => {
  const originalFetch = globalThis.fetch;
  let requestedUrl = null;
  const mockResponse = {
    model: "densenet121-res224-all",
    task: "Multi-label chest X-ray finding scoring",
    target_finding: "Infiltration",
    target_index: 2,
    target_score: 0.51234,
    target_rank: 2,
    predictions: [
      { finding: "Cardiomegaly", score: 0.9 },
      { finding: "Infiltration", score: 0.51234 },
    ],
    gradcam: {
      finding: "Infiltration",
      target_finding: "Infiltration",
      index: 2,
      score: 0.51234,
      rank: 2,
      overlay_png_base64: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNiAAAABgADNjd8qAAAAABJRU5ErkJggg==",
    },
  };

  globalThis.fetch = async (url) => {
    requestedUrl = url;
    return new Response(JSON.stringify(mockResponse), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };

  try {
    const file = new Blob(["cxr-bytes"], { type: "image/png" });
    file.name = "cxr.png";
    const result = await inferImage(file, "Infiltration");
    assert.equal(requestedUrl, "/infer?target_finding=Infiltration");
    assert.equal(result.target_finding, "Infiltration");
    assert.equal(result.target_index, 2);
    assert.equal(result.target_rank, 2);
    assert.equal(result.gradcam.finding, "Infiltration");
    assert.equal(result.gradcam.rank, 2);
    assert.equal(result.predictions.length, 2);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("inferImage falls back to /infer without query parameter when targetFinding is null, undefined, or empty", async () => {
  const originalFetch = globalThis.fetch;
  const requestedUrls = [];
  const mockResponse = {
    target_finding: "Cardiomegaly",
    target_index: 0,
    target_score: 0.9,
    target_rank: 1,
    predictions: [{ finding: "Cardiomegaly", score: 0.9 }],
    gradcam: { finding: "Cardiomegaly", index: 0, score: 0.9, rank: 1, overlay_png_base64: "abc" },
  };

  globalThis.fetch = async (url) => {
    requestedUrls.push(url);
    return new Response(JSON.stringify(mockResponse), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };

  try {
    const file = new Blob(["cxr-bytes"], { type: "image/png" });
    file.name = "cxr.png";

    await inferImage(file);
    await inferImage(file, null);
    await inferImage(file, undefined);
    await inferImage(file, "   ");
    await inferImage(file, "");

    assert.deepEqual(requestedUrls, [
      "/infer",
      "/infer",
      "/infer",
      "/infer",
      "/infer",
    ]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("inferImage properly URL-encodes target_finding parameter", async () => {
  const originalFetch = globalThis.fetch;
  let requestedUrl = null;
  const mockResponse = {
    target_finding: "Pleural Effusion",
    target_index: 5,
    target_score: 0.42,
    target_rank: 3,
    predictions: [{ finding: "Pleural Effusion", score: 0.42 }],
    gradcam: { finding: "Pleural Effusion", index: 5, score: 0.42, rank: 3, overlay_png_base64: "abc" },
  };

  globalThis.fetch = async (url) => {
    requestedUrl = url;
    return new Response(JSON.stringify(mockResponse), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };

  try {
    const file = new Blob(["cxr-bytes"], { type: "image/png" });
    file.name = "cxr.png";
    await inferImage(file, "Pleural Effusion");
    assert.equal(requestedUrl, "/infer?target_finding=Pleural%20Effusion");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("inferImage surfaces backend error when invalid target_finding is rejected (HTTP 400)", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    return new Response(
      JSON.stringify({
        detail: "Unknown target finding 'Tuberculosis'. Must be one of: Atelectasis, Cardiomegaly, Infiltration.",
      }),
      { status: 400, headers: { "Content-Type": "application/json" } },
    );
  };

  try {
    const file = new Blob(["cxr-bytes"], { type: "image/png" });
    file.name = "cxr.png";
    await assert.rejects(
      inferImage(file, "Tuberculosis"),
      /Unknown target finding 'Tuberculosis'/
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});
