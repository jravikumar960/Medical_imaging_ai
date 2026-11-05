import { createStudy } from "./reviewApi.js";
import { inferImage } from "./api.js";

async function hashUploadedFile(file) {
  const digest = await globalThis.crypto.subtle.digest("SHA-256", await file.arrayBuffer());
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function persistenceWarning(reason) {
  const detail = reason instanceof Error ? reason.message : "The study record could not be saved.";
  return {
    status: "warning",
    message: `Inference completed; study save warning: ${detail}. The actual inference results remain available.`,
  };
}

export async function processXrayStudy(
  file,
  targetFinding,
  token,
  { infer = inferImage, persist = createStudy, hashFile = hashUploadedFile, onInference = () => {} } = {},
) {
  const result = await infer(file, targetFinding || null);
  let studyId = "";
  let hashError = null;
  try {
    studyId = await hashFile(file);
  } catch (reason) {
    hashError = reason;
  }

  onInference(result, studyId);

  if (hashError) {
    return { result, studyId, persistence: persistenceWarning(hashError) };
  }

  try {
    const saved = await persist(
      studyId,
      "xray",
      JSON.stringify(result.predictions),
      result.status || "completed",
      token,
    );
    return {
      result,
      studyId,
      persistence: saved?.existing
        ? {
          status: "existing",
          message: "An authorized existing study record was reused; its saved inference results were not overwritten.",
        }
        : { status: "saved", message: "Study record saved." },
    };
  } catch (reason) {
    return { result, studyId, persistence: persistenceWarning(reason) };
  }
}
