const DEFAULT_REVIEW_API_URL = import.meta.env?.VITE_XRAY_API_URL || "http://127.0.0.1:8000";

async function reviewRequest(path, options = {}, token, fetchImpl = fetch, apiUrl = DEFAULT_REVIEW_API_URL) {
  const headers = {
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(options.body ? { "Content-Type": "application/json" } : {}),
    ...options.headers,
  };
  let response;
  try {
    response = await fetchImpl(`${apiUrl}${path}`, { ...options, headers });
  } catch {
    throw new Error("Could not reach the local review API on port 8000.");
  }

  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error("The review API returned an unreadable response.");
  }
  if (!response.ok) {
    throw new Error(typeof payload.detail === "string" ? payload.detail : `Request failed with status ${response.status}.`);
  }
  return payload;
}

export function createStudy(studySha256, modality, aiResultsJson, aiStatus, token, fetchImpl = fetch, apiUrl = DEFAULT_REVIEW_API_URL) {
  if (!/^[0-9a-f]{64}$/.test(studySha256 || "")) {
    throw new Error("A valid analyzed-study SHA-256 is required to create a study.");
  }
  return reviewRequest("/doctor-reviews/studies", {
    method: "POST",
    body: JSON.stringify({ study_sha256: studySha256, modality, ai_results_json: aiResultsJson, ai_status: aiStatus }),
  }, token, fetchImpl, apiUrl);
}

export function listStudies(token, statusFilter, modalityFilter, fetchImpl = fetch, apiUrl = DEFAULT_REVIEW_API_URL) {
  const params = new URLSearchParams();
  if (statusFilter) params.set("status", statusFilter);
  if (modalityFilter) params.set("modality", modalityFilter);
  const query = params.toString() ? `?${params.toString()}` : "";
  return reviewRequest(`/doctor-reviews/studies${query}`, {}, token, fetchImpl, apiUrl);
}

export function getStudy(studyId, token, fetchImpl = fetch, apiUrl = DEFAULT_REVIEW_API_URL) {
  return reviewRequest(`/doctor-reviews/studies/${studyId}`, {}, token, fetchImpl, apiUrl);
}

export function saveReview(studyId, status, assessment, notes, token, fetchImpl = fetch, apiUrl = DEFAULT_REVIEW_API_URL) {
  return reviewRequest(`/doctor-reviews/studies/${studyId}/review`, {
    method: "PUT",
    body: JSON.stringify({ status, assessment, notes }),
  }, token, fetchImpl, apiUrl);
}

export function getAuditHistory(studyId, token, fetchImpl = fetch, apiUrl = DEFAULT_REVIEW_API_URL) {
  return reviewRequest(`/doctor-reviews/studies/${studyId}/audit`, {}, token, fetchImpl, apiUrl);
}
