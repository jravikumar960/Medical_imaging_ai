import test from "node:test";
import assert from "node:assert/strict";
import { checkServiceHealth } from "../src/serviceHealth.js";

test("checkServiceHealth checks all three configured backend health endpoints", async () => {
  const requests = [];
  const statuses = await checkServiceHealth(async (url) => {
    requests.push(url);
    return { ok: true, status: 200 };
  });

  assert.equal(requests.length, 3);
  assert.deepEqual(requests.map((url) => new URL(url).pathname), ["/health", "/health", "/health"]);
  assert.deepEqual(statuses.map(({ id, status }) => [id, status]), [
    ["xray", "online"],
    ["ct", "online"],
    ["mri", "online"],
  ]);
});

test("checkServiceHealth reports offline services without hiding non-success responses", async () => {
  const statuses = await checkServiceHealth(async (url) => {
    if (url.includes(":8002")) return { ok: false, status: 503 };
    throw new Error("connection refused");
  });

  assert.deepEqual(statuses.map(({ status }) => status), ["offline", "offline", "offline"]);
  assert.equal(statuses[1].detail, "Health check returned HTTP 503");
  assert.equal(statuses[0].detail, "Service could not be reached");
});
