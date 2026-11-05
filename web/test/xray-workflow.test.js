import test from "node:test";
import assert from "node:assert/strict";
import { processXrayStudy } from "../src/xrayWorkflow.js";

const predictions = Array.from({ length: 18 }, (_, index) => ({
  finding: `Finding ${index + 1}`,
  score: 0.9 - index / 100,
}));

function testFile(name) {
  return { name, arrayBuffer: async () => new ArrayBuffer(4) };
}

test("successful inference remains available and is distinct from a failed database save", async () => {
  let visibleInference = null;
  const file = testFile("cxr.png");
  const workflow = await processXrayStudy(file, null, "session", {
    infer: async () => ({ status: "completed", predictions }),
    hashFile: async () => "a".repeat(64),
    persist: async () => { throw new Error("database unavailable"); },
    onInference: (result) => { visibleInference = result; },
  });

  assert.equal(visibleInference.predictions.length, 18);
  assert.equal(workflow.result, visibleInference);
  assert.equal(workflow.studyId, "a".repeat(64));
  assert.equal(workflow.persistence.status, "warning");
  assert.match(workflow.persistence.message, /Inference completed; study save warning/);
  assert.match(workflow.persistence.message, /database unavailable/);
});

test("multiple distinct uploaded files receive distinct saved study identities", async () => {
  const persisted = [];
  const results = [];
  for (const [name, id] of [["first.png", "1"], ["second.png", "2"]]) {
    const workflow = await processXrayStudy(testFile(name), null, "session", {
      infer: async () => ({ status: "completed", predictions }),
      hashFile: async () => id.repeat(64),
      persist: async (studyId) => { persisted.push(studyId); return { existing: false }; },
      onInference: (result) => results.push(result),
    });
    assert.equal(workflow.persistence.status, "saved");
  }

  assert.deepEqual(persisted, ["1".repeat(64), "2".repeat(64)]);
  assert.equal(results.length, 2);
  assert.ok(results.every((result) => result.predictions.length === 18));
});

test("duplicate save response is represented as an existing immutable record", async () => {
  const workflow = await processXrayStudy(testFile("same.png"), null, "session", {
    infer: async () => ({ status: "completed", predictions }),
    hashFile: async () => "3".repeat(64),
    persist: async () => ({ existing: true }),
  });

  assert.equal(workflow.persistence.status, "existing");
  assert.match(workflow.persistence.message, /not overwritten/);
});

test("study identity failure does not hide a successful inference", async () => {
  let inferenceWasPublished = false;
  const workflow = await processXrayStudy(testFile("no-hash.png"), null, "session", {
    infer: async () => ({ status: "completed", predictions }),
    hashFile: async () => { throw new Error("SHA-256 unavailable"); },
    persist: async () => { throw new Error("must not be called"); },
    onInference: () => { inferenceWasPublished = true; },
  });

  assert.equal(inferenceWasPublished, true);
  assert.equal(workflow.result.predictions.length, 18);
  assert.equal(workflow.persistence.status, "warning");
  assert.match(workflow.persistence.message, /SHA-256 unavailable/);
});
