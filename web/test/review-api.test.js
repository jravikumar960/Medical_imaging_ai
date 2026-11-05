import test from "node:test";
import assert from "node:assert/strict";
import { registerUser, loginUser, logoutUser } from "../src/authApi.js";
import { createStudy, listStudies, getStudy, saveReview, getAuditHistory } from "../src/reviewApi.js";

const studyId = "a".repeat(64);
const baseUrl = "http://127.0.0.1:8000";

test("auth API sends requests to the backend", async () => {
  let calls = 0;
  const fetchImpl = async () => { calls += 1; return { ok: true, json: async () => ({}) }; };
  await registerUser("doc", "pass1234", fetchImpl);
  assert.equal(calls, 1);
  calls = 0;
  await loginUser("doc", "pass1234", fetchImpl);
  assert.equal(calls, 1);
  calls = 0;
  await logoutUser("tok", fetchImpl);
  assert.equal(calls, 1);
});

test("registerUser sends username and password without a role field", async () => {
  const requests = [];
  const fetchImpl = async (url, options) => {
    requests.push({ url, options });
    return { ok: true, json: async () => ({ user_id: 1, username: "doc", role: "clinician" }) };
  };
  await registerUser("doc", "pass1234", fetchImpl, baseUrl);
  assert.equal(requests[0].url, `${baseUrl}/doctor-reviews/auth/register`);
  assert.equal(requests[0].options.method, "POST");
  const body = JSON.parse(requests[0].options.body);
  assert.equal(body.username, "doc");
  assert.equal(body.password, "pass1234");
  assert.equal(body.role, undefined);
});

test("auth login returns token, username, and role", async () => {
  const requests = [];
  const fetchImpl = async (url, options) => {
    requests.push({ url, options });
    return {
      ok: true,
      json: async () => ({ token: "tok", username: "doc", role: "clinician" }),
    };
  };
  const result = await loginUser("doc", "pass1234", fetchImpl, baseUrl);
  assert.equal(requests[0].url, `${baseUrl}/doctor-reviews/auth/login`);
  assert.equal(requests[0].options.method, "POST");
  assert.equal(result.token, "tok");
  assert.equal(result.username, "doc");
  assert.equal(result.role, "clinician");
});

test("logout sends bearer token", async () => {
  const requests = [];
  const fetchImpl = async (url, options) => {
    requests.push({ url, options });
    return { ok: true, json: async () => ({ status: "logged_out" }) };
  };
  await logoutUser("tok123", fetchImpl, baseUrl);
  assert.equal(requests[0].url, `${baseUrl}/doctor-reviews/auth/logout`);
  assert.equal(requests[0].options.method, "POST");
  assert.equal(requests[0].options.headers.Authorization, "Bearer tok123");
});

test("createStudy validates study hash and sends correct body", async () => {
  const requests = [];
  const fetchImpl = async (url, options) => {
    requests.push({ url, options });
    return { ok: true, json: async () => ({ study_id: 1, study_sha256: studyId }) };
  };
  await createStudy(studyId, "xray", JSON.stringify([{ finding: "x", score: 0.5 }]), "completed", "tok", fetchImpl, baseUrl);
  assert.equal(requests[0].url, `${baseUrl}/doctor-reviews/studies`);
  assert.equal(requests[0].options.method, "POST");
  assert.equal(requests[0].options.headers.Authorization, "Bearer tok");
  const body = JSON.parse(requests[0].options.body);
  assert.equal(body.study_sha256, studyId);
  assert.equal(body.modality, "xray");
  assert.equal(body.ai_status, "completed");
});

test("createStudy rejects invalid study hash", async () => {
  let calls = 0;
  const fetchImpl = async () => { calls += 1; return { ok: true, json: async () => ({}) }; };
  assert.throws(() => createStudy("not-a-hash", "xray", "{}", "completed", "tok", fetchImpl, baseUrl), /SHA-256/);
  assert.equal(calls, 0);
});

test("listStudies supports optional status and modality filters", async () => {
  const requests = [];
  const fetchImpl = async (url) => {
    requests.push(url);
    return { ok: true, json: async () => [] };
  };
  await listStudies("tok", "Pending", "xray", fetchImpl, baseUrl);
  assert.equal(requests[0], `${baseUrl}/doctor-reviews/studies?status=Pending&modality=xray`);
});

test("getStudy returns study and review", async () => {
  const requests = [];
  const fetchImpl = async (url) => {
    requests.push(url);
    return { ok: true, json: async () => ({ id: 1, review: { status: "Reviewed" } }) };
  };
  const data = await getStudy(studyId, "tok", fetchImpl, baseUrl);
  assert.equal(requests[0], `${baseUrl}/doctor-reviews/studies/${studyId}`);
  assert.equal(data.review.status, "Reviewed");
});

test("saveReview sends status, assessment, notes", async () => {
  const requests = [];
  const fetchImpl = async (url, options) => {
    requests.push({ url, options });
    return { ok: true, json: async () => ({ id: 1, status: "Reviewed" }) };
  };
  await saveReview(studyId, "Reviewed", "assessment text", "notes text", "tok", fetchImpl, baseUrl);
  assert.equal(requests[0].url, `${baseUrl}/doctor-reviews/studies/${studyId}/review`);
  assert.equal(requests[0].options.method, "PUT");
  const body = JSON.parse(requests[0].options.body);
  assert.equal(body.status, "Reviewed");
  assert.equal(body.assessment, "assessment text");
  assert.equal(body.notes, "notes text");
});

test("getAuditHistory requests audit endpoint", async () => {
  const requests = [];
  const fetchImpl = async (url) => {
    requests.push(url);
    return { ok: true, json: async () => [] };
  };
  await getAuditHistory(studyId, "tok", fetchImpl, baseUrl);
  assert.equal(requests[0], `${baseUrl}/doctor-reviews/studies/${studyId}/audit`);
});
