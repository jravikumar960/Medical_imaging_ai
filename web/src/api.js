export function rankPredictions(predictions) {
  if (!Array.isArray(predictions) || predictions.some((item) =>
    typeof item?.finding !== "string" || item.finding.length === 0 ||
    typeof item.score !== "number" || !Number.isFinite(item.score))) {
    throw new Error("The inference service returned invalid scores.");
  }
  return [...predictions].sort((left, right) => right.score - left.score);
}

export async function inferImage(file, targetFinding = null) {
  const contentType = file.name.toLowerCase().endsWith(".dcm") ? "application/dicom" : file.type;
  if (!["image/png", "image/jpeg", "application/dicom"].includes(contentType)) {
    throw new Error("Choose a PNG, JPEG, or DICOM image.");
  }

  const trimmed = typeof targetFinding === "string" ? targetFinding.trim() : "";
  const endpoint = trimmed
    ? `/infer?target_finding=${encodeURIComponent(trimmed)}`
    : "/infer";

  let response;
  try {
    response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": contentType },
      body: file,
    });
  } catch (error) {
    throw new Error("Could not reach the inference service. Is the backend running?");
  }

  if (!response.ok) {
    let detail = `Inference failed with status ${response.status}.`;
    try {
      const payload = await response.json();
      if (typeof payload?.detail === "string" && payload.detail.length > 0) {
        detail = payload.detail;
      }
    } catch {
      // Keep the status-based message when the response is not JSON.
    }
    throw new Error(detail);
  }

  const result = await response.json();
  if (!Array.isArray(result?.predictions) || result.predictions.length === 0) {
    throw new Error("The inference service returned no predictions.");
  }
  return result;
}
