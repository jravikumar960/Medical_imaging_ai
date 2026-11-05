const DEFAULT_AUTH_API_URL = import.meta.env?.VITE_XRAY_API_URL || "http://127.0.0.1:8000";

async function authRequest(path, options = {}, fetchImpl = fetch, apiUrl = DEFAULT_AUTH_API_URL) {
  let response;
  try {
    response = await fetchImpl(`${apiUrl}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    });
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

export function registerUser(username, password, fetchImpl = fetch, apiUrl = DEFAULT_AUTH_API_URL) {
  return authRequest("/doctor-reviews/auth/register", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  }, fetchImpl, apiUrl);
}

export function loginUser(username, password, fetchImpl = fetch, apiUrl = DEFAULT_AUTH_API_URL) {
  return authRequest("/doctor-reviews/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  }, fetchImpl, apiUrl);
}

export function logoutUser(token, fetchImpl = fetch, apiUrl = DEFAULT_AUTH_API_URL) {
  return authRequest("/doctor-reviews/auth/logout", {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
  }, fetchImpl, apiUrl);
}
