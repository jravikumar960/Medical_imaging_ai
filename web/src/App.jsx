import { useEffect, useMemo, useRef, useState } from "react";
import { inferImage, rankPredictions } from "./api.js";
import { processXrayStudy } from "./xrayWorkflow.js";
import { registerUser, loginUser, logoutUser } from "./authApi.js";
import {
  createNiftiOriginRecord,
  inspectCtVolume,
  orientationReviewIssues,
  readOrientationReview,
  segmentCtVolume,
  uploadCtEvidence,
  uploadNiftiOriginEvidence,
} from "./ctApi.js";
import {
  createNiftiOriginReviewTemplate,
  getCtAuthorizationPresentation,
} from "./ctWorkflow.js";
import { listStudies } from "./reviewApi.js";
import CTSliceViewer from "./CTSliceViewer.jsx";
import StatusMessage from "./StatusMessage.jsx";
import { MRI_SEQUENCES, segmentMri } from "./mriApi.js";
import MRISliceViewer from "./MRISliceViewer.jsx";
import DoctorReviewPanel from "./DoctorReviewPanel.jsx";
import ImagePreview from "./ImagePreview.jsx";
import DashboardOverview from "./DashboardOverview.jsx";
import { checkServiceHealth } from "./serviceHealth.js";

function formatElapsedTime(totalSeconds) {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`
    : `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function MRIInferenceProgress() {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    const startedAt = Date.now();
    const interval = window.setInterval(
      () => setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000)),
      1000,
    );
    return () => window.clearInterval(interval);
  }, []);

  return (
    <div className="inference-progress">
      <progress aria-label="MRI segmentation progress" />
      <p role="status" aria-live="polite">
        Running the 3D MONAI model locally. This may take several minutes or longer,
        depending on volume size and CPU performance. The service does not provide a percentage.
      </p>
      <p className="field-note">Elapsed upload and processing time: <output aria-live="off">{formatElapsedTime(elapsedSeconds)}</output></p>
    </div>
  );
}

export default function App() {
  const [modality, setModality] = useState("xray");
  const [activeView, setActiveView] = useState("overview");
  const [services, setServices] = useState([
    { id: "xray", label: "X-ray & review API", status: "checking", detail: "Checking connection…" },
    { id: "ct", label: "CT segmentation API", status: "checking", detail: "Checking connection…" },
    { id: "mri", label: "MRI segmentation API", status: "checking", detail: "Checking connection…" },
  ]);
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [xrayModel, setXrayModel] = useState("");
  const [xrayTask, setXrayTask] = useState("");
  const [gradcam, setGradcam] = useState(null);
  const [xrayScoreType, setXrayScoreType] = useState("");
  const [xrayInputWarning, setXrayInputWarning] = useState("");
  const [error, setError] = useState("");
  const [analyzedImage, setAnalyzedImage] = useState(null);
  const [xrayPreviewPngBase64, setXrayPreviewPngBase64] = useState("");
  const [xrayHistory, setXrayHistory] = useState([]);
  const [xrayPersistence, setXrayPersistence] = useState(null);
  const [xrayStudyId, setXrayStudyId] = useState("");
  const [busy, setBusy] = useState(false);
  const [targetFinding, setTargetFinding] = useState("");
  const [gradcamBusy, setGradcamBusy] = useState(false);
  const [gradcamError, setGradcamError] = useState("");
  const fileInput = useRef(null);

  const [token, setToken] = useState("");
  const [user, setUser] = useState(null);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [authTab, setAuthTab] = useState("signin");
  const [authError, setAuthError] = useState("");
  const [authMessage, setAuthMessage] = useState("");
  const [authBusy, setAuthBusy] = useState(false);
  const [studies, setStudies] = useState([]);
  const [studiesBusy, setStudiesBusy] = useState(false);
  const [studiesError, setStudiesError] = useState("");

  const [ctFile, setCtFile] = useState(null);
  const [ctInspection, setCtInspection] = useState(null);
  const [ctProvenanceMode, setCtProvenanceMode] = useState("nifti_origin");
  const [ctSourceMetadataFile, setCtSourceMetadataFile] = useState(null);
  const [ctConversionRecordFile, setCtConversionRecordFile] = useState(null);
  const [ctDatasetMetadataFile, setCtDatasetMetadataFile] = useState(null);
  const [ctOriginRecord, setCtOriginRecord] = useState(null);
  const [ctEvidence, setCtEvidence] = useState(null);
  const [ctEvidenceStatus, setCtEvidenceStatus] = useState("");
  const [ctAuthorizationState, setCtAuthorizationState] = useState("awaiting_review");
  const [ctAuthorizationMessage, setCtAuthorizationMessage] = useState("");
  const [ctReview, setCtReview] = useState(null);
  const [ctReviewName, setCtReviewName] = useState("");
  const [ctResult, setCtResult] = useState(null);
  const [ctStudyId, setCtStudyId] = useState("");
  const [ctBusy, setCtBusy] = useState(false);
  const [ctError, setCtError] = useState("");
  const ctFileInput = useRef(null);

  const [mriFiles, setMriFiles] = useState({ t1c: null, t1: null, t2: null, flair: null });
  const [mriResult, setMriResult] = useState(null);
  const [mriStudyId, setMriStudyId] = useState("");
  const [mriBusy, setMriBusy] = useState(false);
  const [mriError, setMriError] = useState("");
  const mriFileInputs = useRef({});

  const ctStatistics = useMemo(() => {
    if (!ctResult?.mask || !ctResult.geometry?.spacing_mm) return null;
    const labelCounts = Object.fromEntries((ctResult.labels || []).map((label) => [label, 0]));
    for (const label of ctResult.mask) {
      labelCounts[label] = (labelCounts[label] || 0) + 1;
    }
    const predictedVoxels = labelCounts[1] || 0;
    const voxelVolumeMm3 = ctResult.geometry.spacing_mm.reduce((volume, spacing) => volume * spacing, 1);
    return {
      labelCounts,
      totalVoxels: ctResult.mask.length,
      predictedVoxels,
      predictedVolumeMl: predictedVoxels * voxelVolumeMm3 / 1000,
    };
  }, [ctResult]);
  const ctAuthorizationPresentation = getCtAuthorizationPresentation(ctAuthorizationState);
  const ctReviewIssues = ctReview
    ? orientationReviewIssues(
      ctReview,
      ctInspection,
      ctProvenanceMode === "nifti_origin" ? ctOriginRecord?.origin_record_sha256 : null,
    )
    : [];
  const ctValidationBlockers = [
    !ctInspection && "Inspect the exact CT volume first.",
    !ctReview && "Load a complete, signed reviewer JSON that passes local field/binding checks.",
    ctProvenanceMode === "nifti_origin" && !ctDatasetMetadataFile && "Select the Task09 dataset.json.",
    ctProvenanceMode === "nifti_origin" && !ctOriginRecord && "Generate the NIfTI-origin record for this inspection.",
    ctProvenanceMode === "dicom" && !ctSourceMetadataFile && "Select actual source DICOM metadata.",
    ctProvenanceMode === "dicom" && !ctConversionRecordFile && "Select the actual NIfTI conversion record.",
    ctReviewIssues.length > 0 && "Fix the reviewer JSON issues listed above.",
  ].filter(Boolean);
  const ctSegmentationBlockers = [
    !ctFile && "Select the inspected NIfTI volume.",
    !ctInspection && "Inspect the exact volume; its short-lived geometry token is required.",
    !ctReview && "Load valid, current reviewer evidence.",
    !ctEvidence?.provenance_token && "Submit evidence and wait for successful CT backend validation.",
    ctEvidence?.expires_at_unix * 1000 <= Date.now() && "The backend evidence token has expired; inspect and validate again.",
    ctAuthorizationState !== "authorization_validated" && "The CT backend has not established authorization for this inspection.",
  ].filter(Boolean);

  async function refreshServices() {
    setServices((current) => current.map((service) => ({
      ...service,
      status: "checking",
      detail: "Checking connection…",
    })));
    setServices(await checkServiceHealth());
  }

  useEffect(() => {
    void refreshServices();
    const interval = window.setInterval(() => { void refreshServices(); }, 30000);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    if (ctResult) return undefined;
    const reviewExpiresAt = Date.parse(ctReview?.expires_at || "");
    const evidenceExpiresAt = Number(ctEvidence?.expires_at_unix) * 1000;
    const deadlines = [reviewExpiresAt, evidenceExpiresAt].filter(Number.isFinite);
    if (deadlines.length === 0) return undefined;
    const deadline = Math.min(...deadlines);
    const timeout = window.setTimeout(() => {
      const now = Date.now();
      if (Number.isFinite(evidenceExpiresAt) && evidenceExpiresAt <= now) {
        setCtInspection(null);
        setCtOriginRecord(null);
        setCtEvidence(null);
        setCtEvidenceStatus("The short-lived inspection and authorization token expired.");
        setCtReview(null);
        setCtReviewName("");
        setCtAuthorizationState("review_rejected");
        setCtAuthorizationMessage("The inspection token expired. Inspect the exact volume again and regenerate its origin record, then upload a still-current signed review matching the new hashes (obtain a fresh review if the old one expired or the hashes changed).");
      } else if (Number.isFinite(reviewExpiresAt) && reviewExpiresAt <= now) {
        setCtEvidence(null);
        setCtEvidenceStatus("The reviewer evidence has expired.");
        setCtReview(null);
        setCtReviewName("");
        setCtAuthorizationState("review_rejected");
        setCtAuthorizationMessage("The reviewer evidence expired. Obtain a fresh independent review and signature for the current inspection.");
      }
    }, Math.max(0, deadline - Date.now()));
    return () => window.clearTimeout(timeout);
  }, [ctEvidence, ctReview, ctResult]);

  function changeModality(event) {
    setModality(event.target.value);
    setActiveView("workspace");
  }

  function selectMriFile(sequence, selectedFile) {
    setMriFiles((current) => ({ ...current, [sequence]: selectedFile }));
    setMriResult(null);
    setMriStudyId("");
    setMriError("");
  }

  async function runMriSegmentation(event) {
    event.preventDefault();
    if (mriBusy || modality !== "mri") return;
    setMriBusy(true);
    setMriError("");
    setMriResult(null);
    setMriStudyId("");
    try {
      const result = await segmentMri(mriFiles);
      setMriResult(result);
      setMriStudyId(result.study_id);
    } catch (reason) {
      setMriError(reason instanceof Error ? reason.message : "Pretrained brain MRI segmentation failed.");
    } finally {
      setMriBusy(false);
    }
  }

  function selectXrayFile(selectedFile) {
    setFile(selectedFile);
    setError("");
  }

  function selectCtFile(selectedFile) {
    setCtFile(selectedFile);
    setCtInspection(null);
    setCtSourceMetadataFile(null);
    setCtConversionRecordFile(null);
    setCtDatasetMetadataFile(null);
    setCtOriginRecord(null);
    setCtEvidence(null);
    setCtEvidenceStatus("");
    setCtAuthorizationState("awaiting_review");
    setCtAuthorizationMessage("Inspect the selected volume to begin its independent review.");
    setCtReview(null);
    setCtReviewName("");
    setCtResult(null);
    setCtStudyId("");
    setCtError("");
  }

  function selectCtProvenanceMode(mode) {
    setCtProvenanceMode(mode);
    setCtSourceMetadataFile(null);
    setCtConversionRecordFile(null);
    setCtDatasetMetadataFile(null);
    setCtOriginRecord(null);
    setCtEvidence(null);
    setCtEvidenceStatus("");
    setCtAuthorizationState("awaiting_review");
    setCtAuthorizationMessage("Select the provenance route that truthfully describes this volume.");
    setCtReview(null);
    setCtReviewName("");
    setCtResult(null);
    setCtError("");
  }

  async function handleLogin(event) {
    event.preventDefault();
    setAuthError("");
    setAuthMessage("");
    setAuthBusy(true);
    try {
      const result = await loginUser(username, password);
      setToken(result.token);
      setUser({ username: result.username, role: result.role });
      setActiveView("overview");
      await loadStudies(result.token);
    } catch (reason) {
      setAuthError(reason instanceof Error ? reason.message : "Login failed.");
    } finally {
      setAuthBusy(false);
    }
  }

  async function handleRegister(event) {
    event.preventDefault();
    setAuthError("");
    setAuthMessage("");
    setAuthBusy(true);
    try {
      await registerUser(username, password);
      setAuthMessage("Account created. Please log in.");
    } catch (reason) {
      setAuthError(reason instanceof Error ? reason.message : "Registration failed.");
    } finally {
      setAuthBusy(false);
    }
  }

  async function handleLogout() {
    try {
      await logoutUser(token);
    } catch {
      // Ignore logout errors; clear local state regardless.
    }
    setToken("");
    setUser(null);
    setStudies([]);
    setActiveView("overview");
  }

  async function loadStudies(accessToken = token) {
    if (!accessToken) return;
    setStudiesBusy(true);
    setStudiesError("");
    try {
      const data = await listStudies(accessToken);
      setStudies(data || []);
    } catch (reason) {
      setStudies([]);
      setStudiesError(reason instanceof Error ? reason.message : "Could not load saved research sessions.");
    } finally {
      setStudiesBusy(false);
    }
  }

  async function submitXray(event) {
    event.preventDefault();
    if (!file || busy || modality !== "xray") return;
    if (!token) {
      setError("Please log in before running inference.");
      return;
    }
    setBusy(true);
    setError("");
    setGradcamError("");
    try {
      const workflow = await processXrayStudy(file, targetFinding || null, token, {
        onInference: (result, studyId) => {
          if (analyzedImage && xrayStudyId && xrayStudyId !== studyId && predictions.length > 0) {
            const previous = {
              studyId: xrayStudyId,
              file: analyzedImage,
              previewPngBase64: xrayPreviewPngBase64,
              predictions,
              model: xrayModel,
              task: xrayTask,
              gradcam,
              scoreType: xrayScoreType,
              inputWarning: xrayInputWarning,
              persistence: xrayPersistence,
            };
            setXrayHistory((current) => [
              previous,
              ...current.filter((item) => item.studyId !== previous.studyId),
            ]);
          }
          setXrayPersistence({ status: "pending", message: "Saving the study record…" });
          setPredictions(rankPredictions(result.predictions));
          setXrayModel(result.model);
          setXrayTask(result.task);
          setGradcam(result.gradcam);
          if (result.gradcam?.finding) setTargetFinding(result.gradcam.finding);
          setXrayScoreType(result.scoreType || result.score_type || "");
          setXrayInputWarning(result.inputWarning || result.input_warning || "");
          setAnalyzedImage(file);
          setXrayPreviewPngBase64(result.input_preview_png_base64 || "");
          setXrayStudyId(studyId);
        },
      });
      setXrayPersistence(workflow.persistence);
      if (workflow.persistence.status !== "warning") await loadStudies(token);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Inference failed.");
    } finally {
      setBusy(false);
      setFile(null);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  async function selectTargetFinding(findingName) {
    if (!analyzedImage || !findingName || gradcamBusy || busy) return;
    setTargetFinding(findingName);
    setGradcamBusy(true);
    setGradcamError("");
    try {
      const result = await inferImage(analyzedImage, findingName);
      setGradcam(result.gradcam);
      if (Array.isArray(result.predictions) && result.predictions.length > 0) {
        setPredictions(rankPredictions(result.predictions));
      }
    } catch (reason) {
      setGradcamError(reason instanceof Error ? reason.message : "Failed to generate Grad-CAM for selected finding.");
    } finally {
      setGradcamBusy(false);
    }
  }

  async function inspectSelectedCt() {
    if (!ctFile || ctBusy) return;
    setCtBusy(true);
    setCtError("");
    setCtInspection(null);
    setCtOriginRecord(null);
    setCtEvidence(null);
    setCtEvidenceStatus("");
    setCtResult(null);
    setCtStudyId("");
    setCtReview(null);
    setCtReviewName("");
    setCtAuthorizationState("awaiting_review");
    setCtAuthorizationMessage("Inspecting the selected volume; independent orientation review is still required.");
    try {
      setCtInspection(await inspectCtVolume(ctFile));
      setCtAuthorizationMessage("Inspection completed. Generate the bound origin record, then request independent review.");
    } catch (reason) {
      setCtError(reason instanceof Error ? reason.message : "CT inspection failed.");
    } finally {
      setCtBusy(false);
    }
  }

  async function loadCtReview(event) {
    const reviewFile = event.target.files?.[0];
    setCtEvidence(null);
    setCtEvidenceStatus("");
    setCtResult(null);
    setCtReview(null);
    setCtReviewName(reviewFile?.name || "");
    setCtError("");
    setCtAuthorizationState(reviewFile ? "review_submitted" : "awaiting_review");
    setCtAuthorizationMessage(reviewFile
      ? "Reading reviewer evidence; it is not authorized until the CT backend validates it."
      : "No reviewer evidence is loaded.");
    if (!reviewFile) return;
    if (!ctInspection) {
      const message = "Inspect the exact CT volume before loading its review.";
      setCtAuthorizationState("review_rejected");
      setCtAuthorizationMessage(message);
      setCtError(message);
      return;
    }
    try {
      setCtReview(await readOrientationReview(
        reviewFile,
        ctInspection,
        ctProvenanceMode === "nifti_origin" ? ctOriginRecord?.origin_record_sha256 : null,
      ));
      setCtAuthorizationState("review_submitted");
      setCtAuthorizationMessage("Reviewer evidence is loaded and bound to this inspection. Submit it for CT backend validation.");
    } catch (reason) {
      setCtReview(null);
      setCtAuthorizationState("review_rejected");
      setCtAuthorizationMessage("The selected review is malformed, expired, or does not match this inspection.");
      setCtError(reason instanceof Error ? reason.message : "Orientation review could not be read.");
    }
  }

  async function validateCtEvidence() {
    const niftiOrigin = ctProvenanceMode === "nifti_origin";
    if (!ctInspection || !ctReview || ctBusy ||
        (niftiOrigin && (!ctDatasetMetadataFile || !ctOriginRecord)) ||
        (!niftiOrigin && (!ctSourceMetadataFile || !ctConversionRecordFile))) return;
    setCtBusy(true);
    setCtError("");
    setCtEvidence(null);
    setCtEvidenceStatus(niftiOrigin
      ? "Validating NIfTI-origin metadata and review against this inspection…"
      : "Validating the supplied DICOM source records against this inspection…");
    setCtAuthorizationState("review_submitted");
    setCtAuthorizationMessage("The CT backend is validating the signed review, provenance, expiry, and inspection bindings.");
    setCtResult(null);
    try {
      const evidence = niftiOrigin
        ? await uploadNiftiOriginEvidence(ctInspection, ctReview, ctDatasetMetadataFile)
        : await uploadCtEvidence(ctInspection, ctReview, ctSourceMetadataFile, ctConversionRecordFile);
      setCtEvidence(evidence);
      setCtEvidenceStatus(niftiOrigin
        ? "The API bound the Task09 NIfTI-origin record and independent review to this inspection."
        : "The API confirmed record integrity and geometric consistency for this inspection.");
      setCtAuthorizationState("authorization_validated");
      setCtAuthorizationMessage("Authorization was accepted by the CT backend; segmentation is enabled while the one-use token remains valid.");
    } catch (reason) {
      setCtEvidenceStatus("Source evidence was not validated; segmentation remains blocked.");
      setCtAuthorizationState("review_rejected");
      setCtAuthorizationMessage("The CT backend rejected or could not validate the supplied review. No segmentation was run.");
      setCtError(reason instanceof Error ? reason.message : "CT source evidence validation failed.");
    } finally {
      setCtBusy(false);
    }
  }

  async function generateCtOriginRecord() {
    if (!ctInspection || !ctDatasetMetadataFile || ctBusy) return;
    setCtBusy(true);
    setCtError("");
    setCtOriginRecord(null);
    setCtReview(null);
    setCtEvidence(null);
    setCtEvidenceStatus("");
    setCtResult(null);
    setCtAuthorizationState("awaiting_review");
    setCtAuthorizationMessage("Origin record generation does not establish orientation; an independent reviewer is still required.");
    try {
      setCtOriginRecord(await createNiftiOriginRecord(ctInspection, ctDatasetMetadataFile));
    } catch (reason) {
      setCtError(reason instanceof Error ? reason.message : "NIfTI-origin metadata could not be generated.");
    } finally {
      setCtBusy(false);
    }
  }

  function downloadCtOriginRecord() {
    if (!ctOriginRecord?.origin_record) return;
    const record = new Blob([`${JSON.stringify(ctOriginRecord.origin_record, null, 2)}\n`], {
      type: "application/json",
    });
    const url = URL.createObjectURL(record);
    const link = document.createElement("a");
    link.href = url;
    link.download = "task09-spleen_1-nifti-origin.json";
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function downloadCtReviewTemplate() {
    if (!ctInspection || !ctOriginRecord?.origin_record_sha256) return;
    const template = createNiftiOriginReviewTemplate(ctInspection, ctOriginRecord.origin_record_sha256);
    const file = new Blob([`${JSON.stringify(template, null, 2)}\n`], { type: "application/json" });
    const url = URL.createObjectURL(file);
    const link = document.createElement("a");
    link.href = url;
    link.download = "task09-spleen_1-orientation-review-template.json";
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function runCtSegmentation() {
    if (!ctFile || !ctInspection || !ctReview || !ctEvidence?.provenance_token ||
        ctAuthorizationState !== "authorization_validated" || ctBusy) return;
    if (ctEvidence.expires_at_unix * 1000 <= Date.now()) {
      setCtInspection(null);
      setCtOriginRecord(null);
      setCtEvidence(null);
      setCtEvidenceStatus("The short-lived inspection and authorization token expired.");
      setCtReview(null);
      setCtReviewName("");
      setCtAuthorizationState("review_rejected");
      setCtAuthorizationMessage("The inspection token expired. Inspect the exact volume again and regenerate its origin record, then upload a still-current signed review matching the new hashes (obtain a fresh review if the old one expired or the hashes changed).");
      return;
    }
    setCtBusy(true);
    setCtError("");
    setCtResult(null);
    try {
      const result = await segmentCtVolume(
        ctFile,
        ctInspection,
        ctReview,
        ctEvidence.provenance_token,
      );
      setCtResult(result);
      setCtStudyId(ctInspection.file_sha256);
      setCtAuthorizationState("segmentation_available");
      setCtAuthorizationMessage("The CT service returned an actual pretrained-model mask for this authorized volume, and the UI validated its geometry and labels.");
    } catch (reason) {
      setCtEvidence(null);
      setCtEvidenceStatus("The one-use authorization token cannot be reused. Validate the current signed review again before retrying.");
      setCtAuthorizationState("review_submitted");
      setCtAuthorizationMessage("Segmentation did not complete with a valid mask. Revalidate the submitted review to obtain a fresh one-use token.");
      setCtError(reason instanceof Error ? reason.message : "Pretrained CT segmentation failed.");
    } finally {
      setCtBusy(false);
    }
  }

  function openResults() {
    setActiveView("workspace");
    window.setTimeout(() => {
      document.getElementById("analysis-results")?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 0);
  }

  return (!user) ? (
    <main className="auth-page">
      <section className="auth-intro" aria-label="Medical Imaging AI research workspace">
        <a className="brand auth-brand" href="#dashboard-home" aria-label="Medical Imaging AI home">
          <span className="brand-mark" aria-hidden="true">+</span>
          <span>MedView <small>IMAGING WORKSPACE</small></span>
        </a>
        <div className="auth-intro-copy">
          <p className="eyebrow">Medical Imaging AI</p>
          <h1>Research with clarity.</h1>
          <p>A focused workspace for exploring imaging models, reviewing results, and understanding the safeguards behind every workflow.</p>
          <div className="auth-modality-list" aria-label="Available research modalities">
            <span><i aria-hidden="true">XR</i> Chest X-ray</span>
            <span><i aria-hidden="true">CT</i> CT · Spleen</span>
            <span><i aria-hidden="true">MR</i> MRI · Brain</span>
          </div>
        </div>
        <p className="auth-intro-footnote">Research demonstration only · Not for diagnosis or treatment</p>
      </section>
      <section className="auth-form-side">
        <section className="auth-panel" aria-labelledby="auth-heading">
          <div className="auth-heading">
            <p className="eyebrow">Secure workspace access</p>
            <h2 id="auth-heading">{authTab === "signin" ? "Welcome back" : "Create an account"}</h2>
            <p>Sign in to access saved research sessions and reviewer tools.</p>
          </div>
          <div className="auth-tabs">
            <button
              type="button"
              className={`auth-tab ${authTab === "signin" ? "auth-tab-active" : ""}`}
              onClick={() => setAuthTab("signin")}
              aria-pressed={authTab === "signin"}
            >
              Sign In
            </button>
            <button
              type="button"
              className={`auth-tab ${authTab === "register" ? "auth-tab-active" : ""}`}
              onClick={() => setAuthTab("register")}
              aria-pressed={authTab === "register"}
            >
              Register
            </button>
          </div>
          {authError && <p className="error" role="alert">{authError}</p>}
          {authMessage && <p className="success-message" role="status">{authMessage}</p>}
          {authTab === "signin" && (
            <form onSubmit={handleLogin} className="auth-form signin-form" autoComplete="on">
              <label htmlFor="login-username">Username</label>
              <input id="login-username" autoComplete="username" required maxLength={64} value={username}
                onChange={(event) => setUsername(event.target.value)} />
              <label htmlFor="login-password">Password</label>
              <input id="login-password" type="password" autoComplete="current-password" required maxLength={128} value={password}
                onChange={(event) => setPassword(event.target.value)} />
              <div className="auth-actions">
                <button type="submit" disabled={authBusy}>{authBusy ? "Signing in…" : "Sign in"}</button>
              </div>
              <p className="field-note auth-switch">
                No account yet? <button type="button" className="auth-link" onClick={() => setAuthTab("register")}>Register</button>
              </p>
            </form>
          )}
          {authTab === "register" && (
            <form onSubmit={handleRegister} className="auth-form register-form" autoComplete="on">
              <label htmlFor="reg-username">Username</label>
              <input id="reg-username" autoComplete="username" required maxLength={64} value={username}
                onChange={(event) => setUsername(event.target.value)} />
              <label htmlFor="reg-password">Password (min 8 chars)</label>
              <input id="reg-password" type="password" autoComplete="new-password" required minLength={8} maxLength={128} value={password}
                onChange={(event) => setPassword(event.target.value)} />
              <div className="auth-actions">
                <button type="submit" disabled={authBusy}>{authBusy ? "Registering…" : "Register"}</button>
              </div>
              <p className="field-note auth-switch">
                Already have an account? <button type="button" className="auth-link" onClick={() => setAuthTab("signin")}>Sign in</button>
              </p>
            </form>
          )}
          <p className="auth-footnote">Access uses the existing local review service. User sessions are held in memory and end when the service restarts.</p>
        </section>
      </section>
    </main>
  ) : (
    <div className="dashboard">
      <aside className="sidebar" aria-label="Application navigation">
        <button type="button" className="brand" onClick={() => setActiveView("overview")} aria-label="Medical Imaging AI home">
          <span className="brand-mark" aria-hidden="true">+</span>
          <span>MedView <small>IMAGING WORKSPACE</small></span>
        </button>
        <p className="sidebar-label">OVERVIEW</p>
        <nav className="modality-nav primary-nav" aria-label="Primary navigation">
          <button type="button" className={`nav-item ${activeView === "overview" ? "active" : ""}`}
            aria-current={activeView === "overview" ? "page" : undefined}
            onClick={() => setActiveView("overview")}>
            <span className="nav-icon" aria-hidden="true">⌂</span><span>Dashboard</span>
          </button>
          <button type="button" className={`nav-item ${activeView === "sessions" ? "active" : ""}`}
            aria-current={activeView === "sessions" ? "page" : undefined}
            onClick={() => setActiveView("sessions")}>
            <span className="nav-icon" aria-hidden="true">≡</span><span>Research sessions</span>
          </button>
          <button type="button" className="nav-item" onClick={openResults}>
            <span className="nav-icon" aria-hidden="true">▧</span><span>Current results</span>
          </button>
        </nav>
        <p className="sidebar-label">MODALITIES</p>
        <nav className="modality-nav" aria-label="Imaging modality">
          {[
            ["xray", "Chest X-ray", "XR"],
            ["ct", "CT · Spleen", "CT"],
            ["mri", "MRI", "MR"],
          ].map(([key, label, abbreviation]) => (
            <button key={key} type="button" className={modality === key && activeView === "workspace" ? "nav-item active" : "nav-item"}
              aria-current={modality === key && activeView === "workspace" ? "page" : undefined}
              disabled={busy || ctBusy || mriBusy}
              onClick={() => changeModality({ target: { value: key } })}>
              <span className="nav-icon">{abbreviation}</span><span>{label}</span>
            </button>
          ))}
        </nav>
        <button type="button" className="sidebar-upload" onClick={() => {
          if (modality === "xray") fileInput.current?.click();
          if (modality === "ct") ctFileInput.current?.click();
          if (modality === "mri") document.getElementById("mri-t1c")?.click();
        }} disabled={busy || ctBusy || mriBusy}>
          <span aria-hidden="true">＋</span> Upload study
        </button>
        <div className="sidebar-account">
          <span className="avatar" aria-hidden="true">{user.username.slice(0, 1).toUpperCase()}</span>
          <span className="account-copy"><strong>{user.username}</strong><small>{user.role}</small></span>
          <button type="button" className="account-settings" aria-label="Open account and service settings"
            title="Settings" onClick={() => setActiveView("settings")}>
            <span aria-hidden="true">⚙</span>
          </button>
        </div>
        <div className="sidebar-note">
          <button type="button" className="sidebar-logout" onClick={handleLogout} disabled={busy}>
            Sign out
          </button>
          <p>Research use only · Not for diagnosis or treatment.</p>
        </div>
      </aside>

      <main className="shell" id="dashboard-home">
      {activeView === "overview" && (
        <DashboardOverview username={user.username} role={user.role} services={services} studies={studies}
          studiesBusy={studiesBusy} studiesError={studiesError} onNavigate={(destination) => {
            if (destination === "sessions") {
              setActiveView("sessions");
            } else {
              changeModality({ target: { value: destination } });
            }
          }} onRefreshServices={() => void refreshServices()} onRefreshStudies={() => void loadStudies()} />
      )}
      {activeView === "sessions" && (
        <section className="surface-panel sessions-page" aria-labelledby="sessions-heading">
          <div className="page-heading">
            <div><p className="eyebrow">Saved records</p><h1 id="sessions-heading">Research sessions</h1>
              <p className="header-subtitle">Studies saved to the authenticated review service.</p></div>
            <button type="button" className="button-secondary" onClick={() => void loadStudies()}
              disabled={studiesBusy}>{studiesBusy ? "Refreshing…" : "Refresh sessions"}</button>
          </div>
          {studiesError && <p className="inline-error" role="alert">{studiesError}</p>}
          {studiesBusy ? <p className="empty-state" role="status">Loading saved sessions…</p>
            : studies.length ? (
              <div className="session-table-wrap">
                <table className="session-table">
                  <thead><tr><th scope="col">Modality</th><th scope="col">Study hash</th>
                    <th scope="col">AI status</th><th scope="col">Review status</th><th scope="col">Created</th></tr></thead>
                  <tbody>{studies.map((study) => (
                    <tr key={study.id || study.study_sha256}>
                      <td>{study.modality || "—"}</td>
                      <td><code>{study.study_sha256 || "—"}</code></td>
                      <td>{study.ai_status || "—"}</td>
                      <td>{study.review_status || "Pending"}</td>
                      <td>{study.created_at ? new Date(study.created_at).toLocaleString() : "—"}</td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
            ) : <p className="empty-state">No research sessions have been saved to this account yet.</p>}
        </section>
      )}
      {activeView === "settings" && (
        <section className="surface-panel settings-page" aria-labelledby="settings-heading">
          <div className="page-heading">
            <div><p className="eyebrow">Workspace settings</p><h1 id="settings-heading">Account &amp; services</h1>
              <p className="header-subtitle">Your current account and live backend connection status.</p></div>
            <button type="button" className="button-secondary" onClick={() => void refreshServices()}>Refresh services</button>
          </div>
          <div className="settings-grid">
            <section className="settings-block" aria-labelledby="account-heading">
              <h2 id="account-heading">Signed-in account</h2>
              <p><span>Username</span><strong>{user.username}</strong></p>
              <p><span>Role</span><strong>{user.role}</strong></p>
              <p className="field-note">Account permissions are managed by the existing review API.</p>
            </section>
            <section className="settings-block" aria-labelledby="settings-services-heading">
              <h2 id="settings-services-heading">Backend connections</h2>
              <div className="service-list">{services.map((service) => (
                <div className="service-row" key={service.id}>
                  <span className={`service-indicator ${service.status}`} aria-hidden="true" />
                  <div className="service-copy"><strong>{service.label}</strong><span>{service.detail}</span></div>
                  <span className={`service-status ${service.status}`}>{service.status === "online" ? "Online" : service.status === "checking" ? "Checking" : "Offline"}</span>
                </div>
              ))}</div>
              <p className="field-note">Health checks confirm service reachability only, not model availability or clinical validation.</p>
            </section>
          </div>
        </section>
      )}
      {activeView === "workspace" && <>
      <header className="dashboard-header">
        <p className="eyebrow">{modality === "xray" ? "Chest X-ray" : modality === "ct" ? "Computed tomography" : "Magnetic resonance imaging"} · Research workspace</p>
        <div className="header-title-row">
          <div><h1>{modality === "xray" ? "X-ray analysis" : modality === "ct" ? "CT volume workspace" : "MRI segmentation workspace"}</h1>
            <p className="header-subtitle">Review model outputs with modality-specific safeguards.</p></div>
          <span className="research-chip"><span className="chip-dot" /> RESEARCH ONLY</span>
        </div>
      </header>
      <aside className="research-notice" aria-label="Research use disclaimer">
        <span className="research-notice-mark" aria-hidden="true">i</span>
        <p><strong>Research demonstration only · Not clinically validated.</strong>
          {" "}Outputs are not diagnoses or treatment recommendations.
          {modality === "ct" && " CT segmentation remains pending independent orientation review."}
        </p>
      </aside>

      {modality === "xray" && (
        <>
          <section className="panel workspace-panel" aria-labelledby="upload-heading" aria-busy={busy}>
            <div className="section-heading">
              <div><p className="eyebrow">Image input</p><h2 id="upload-heading">Chest X-ray research analysis</h2></div>
              <span className="status-badge neutral">PNG · JPEG · DICOM</span>
            </div>
            <p>A local image-modality classifier must verify chest X-ray content before the finding model runs. Unsupported or uncertain images are rejected; file format and DICOM metadata alone cannot verify image content.</p>
            <form onSubmit={submitXray} onDragOver={(event) => event.preventDefault()} onDrop={(event) => {
              event.preventDefault();
              selectXrayFile(event.dataTransfer.files?.[0] || null);
            }} className="upload-workflow">
              <label className="upload-dropzone" htmlFor="image-file">
                <span className="upload-symbol" aria-hidden="true">↑</span>
                <strong>{file ? file.name : "Drop a study image here"}</strong>
                <span>or select a PNG, JPEG, or DICOM file</span>
                <input ref={fileInput} id="image-file" type="file" accept="image/png,image/jpeg,.dcm,application/dicom"
                  disabled={busy}
                  onChange={(event) => selectXrayFile(event.target.files?.[0] || null)} />
              </label>
              <div className="xray-controls">
                <label htmlFor="target-finding-input">Target finding <span className="field-note">Optional</span></label>
                <input id="target-finding-input" type="text"
                  placeholder="Defaults to the model's highest-scoring finding"
                  value={targetFinding} disabled={busy}
                  onChange={(event) => setTargetFinding(event.target.value)} />
                <p className="field-note">Model scores are not calibrated probabilities or diagnostic certainty.</p>
                <div className="form-actions">
                  <button type="submit" disabled={!file || busy}>{busy ? "Running analysis…" : "Run inference"}</button>
                  {file && <button type="button" className="button-secondary" disabled={busy}
                    onClick={() => selectXrayFile(null)}>Remove selected image</button>}
                </div>
              </div>
            </form>
            {file && <ImagePreview file={file} />}
            {busy && <><progress aria-label="X-ray analysis progress" /><p role="status" aria-live="polite">Processing the selected image locally…</p></>}
            {error && <StatusMessage kind="error">{error}</StatusMessage>}
            {xrayInputWarning && <StatusMessage kind="success">{xrayInputWarning}</StatusMessage>}
          </section>
          {predictions.length > 0 && (
            <section id="analysis-results" className="panel result-grid results-panel" aria-labelledby="results-heading">
              <div><p className="eyebrow">Analysis output</p><h2 id="results-heading">Original image</h2>
                <ImagePreview file={analyzedImage} analyzed previewPngBase64={xrayPreviewPngBase64} />
              </div>
              <div className="result-content">
              <p className="eyebrow">Research results</p><h2>Chest X-ray classification</h2>
              <StatusMessage kind="success"><strong>Inference status:</strong> Completed; {predictions.length === 18
                ? "all 18 model outputs returned."
                : `${predictions.length} model outputs returned.`}</StatusMessage>
              {xrayPersistence && <StatusMessage kind={xrayPersistence.status === "warning" ? "warning" : "info"}>
                <strong>Study persistence:</strong> {xrayPersistence.message}
              </StatusMessage>}
              <p><strong>Model:</strong> {xrayModel || "Model identifier unavailable"}</p>
              <p><strong>Task:</strong> {xrayTask || "Task metadata unavailable"}.</p>
              <p>{xrayScoreType === "operating_point_normalized_score"
                ? "TorchXRayVision operating-point-normalized scores; these are not calibrated probabilities or diagnostic certainty."
                : "Model output scores; these are not calibrated probabilities or diagnostic certainty."}</p>
              <h3 className="output-list-heading">All model output scores <span>({predictions.length})</span></h3>
              <ol className="predictions-list" aria-label={`All ${predictions.length} model output scores`}>{predictions.map(({ finding, score }) => (
                <li key={finding} className={gradcam?.finding === finding ? "prediction-selected" : ""}>
                  <button
                    type="button"
                    className={`finding-select-button ${gradcam?.finding === finding ? "active" : ""}`}
                    onClick={() => selectTargetFinding(finding)}
                    disabled={gradcamBusy || busy}
                    title={`Generate Grad-CAM for ${finding}`}
                  >
                    <span>{finding}</span>
                    <output>{score.toFixed(5)}</output>
                  </button>
                </li>
              ))}</ol>
              {xrayInputWarning && <StatusMessage kind="success">{xrayInputWarning}</StatusMessage>}
              <p className="results-disclaimer">Not clinically validated. Scores are not probabilities, diagnoses, or treatment recommendations.</p>
              </div>
            </section>
          )}
          {gradcam && (
            <section className="panel" aria-labelledby="cam-heading" aria-busy={gradcamBusy}>
              <h2 id="cam-heading">Grad-CAM overlay</h2>
              <div className="gradcam-controls">
                <label htmlFor="target-finding-select">Target finding:</label>
                <select
                  id="target-finding-select"
                  value={gradcam.finding}
                  disabled={gradcamBusy || busy}
                  onChange={(event) => selectTargetFinding(event.target.value)}
                >
                  {predictions.map(({ finding, score }) => (
                    <option key={finding} value={finding}>
                      {finding} (score: {score.toFixed(5)})
                    </option>
                  ))}
                </select>
                {gradcamBusy && <span className="field-note" role="status" aria-live="polite">Updating Grad-CAM overlay…</span>}
              </div>
              {gradcamError && <p className="error" role="alert">{gradcamError}</p>}
              <p>
                <strong>{gradcam.finding}</strong>
                {typeof gradcam.rank === "number" && ` · Rank ${gradcam.rank} of ${predictions.length}`}
                {typeof gradcam.index === "number" && ` (index ${gradcam.index})`}
                {` · model score ${gradcam.score.toFixed(5)}.`}
                {" "}This sensitivity visualization is not proof of pathology, clinical accuracy, or localization.
              </p>
              <img className="gradcam" src={`data:image/png;base64,${gradcam.overlay_png_base64}`}
                alt={`Grad-CAM sensitivity overlay for ${gradcam.finding}`} />
            </section>
          )}
          {predictions.length > 0 && xrayStudyId
            && ["saved", "existing"].includes(xrayPersistence?.status) && (
            <DoctorReviewPanel key={xrayStudyId} studyId={xrayStudyId} token={token}
              aiSummary={`TorchXRayVision DenseNet121 returned ${predictions.length} raw finding scores. Scores are not calibrated probabilities or diagnostic conclusions.`} />
          )}
          {xrayHistory.length > 0 && (
            <section className="panel" aria-labelledby="xray-history-heading">
              <p className="eyebrow">Session history</p><h2 id="xray-history-heading">Previous X-ray analyses</h2>
              {xrayHistory.map((entry) => (
                <details key={entry.studyId} className="xray-history-entry">
                  <summary>{entry.file.name} · {entry.predictions.length} model outputs</summary>
                  <ImagePreview file={entry.file} analyzed previewPngBase64={entry.previewPngBase64} />
                  <p><strong>Inference status:</strong> Completed.</p>
                  {entry.persistence && <p><strong>Study persistence:</strong> {entry.persistence.message}</p>}
                  <p><strong>Model:</strong> {entry.model || "Model identifier unavailable"}</p>
                  <ol className="predictions-list">{entry.predictions.map(({ finding, score }) => (
                    <li key={finding}><span>{finding}</span><output>{score.toFixed(5)}</output></li>
                  ))}</ol>
                  {entry.gradcam?.overlay_png_base64 && (
                    <img className="gradcam" src={`data:image/png;base64,${entry.gradcam.overlay_png_base64}`}
                      alt={`Grad-CAM sensitivity overlay for ${entry.gradcam.finding}`} />
                  )}
                </details>
              ))}
            </section>
          )}
        </>
      )}

      {modality === "ct" && (
        <>
          <section className="panel workspace-panel" aria-labelledby="ct-heading" aria-busy={ctBusy}>
          <div className="section-heading">
            <div><p className="eyebrow">3D volume input</p><h2 id="ct-heading">CT spleen segmentation</h2></div>
            <span className={`status-badge ${ctAuthorizationPresentation.badge}`}>
              {ctAuthorizationPresentation.label}
            </span>
          </div>
          <p className="status-pill">Separate pretrained MONAI workflow · reviewer-signed, volume-specific orientation evidence required</p>
          <div className={`authorization-banner ${ctAuthorizationState.replace(/_/g, "-")}`} role="status" aria-live="polite">
            <strong>{ctAuthorizationPresentation.label}</strong>
            <span>{ctAuthorizationMessage || ctAuthorizationPresentation.detail}</span>
          </div>
            <p>Inspection checks header geometry and returns an exact-file SHA-256-bound, short-lived token; it does not run inference. Select the provenance path that truthfully describes the distributed input. Header axis codes alone never establish anatomical orientation.</p>
            <p>Both paths require an independent human review and a valid HMAC signature supplied by the reviewer; this interface never approves or signs reviews. The service checks review integrity, identity binding and expiry, but cannot independently authenticate the reviewer or establish the truth of their claims. HMAC uses a shared secret, so its custody must be controlled outside this UI.</p>
            <form onSubmit={(event) => { event.preventDefault(); void inspectSelectedCt(); }}
              onDragOver={(event) => event.preventDefault()} onDrop={(event) => {
                event.preventDefault();
                selectCtFile(event.dataTransfer.files?.[0] || null);
              }}>
              <label htmlFor="ct-volume">3D CT volume (.nii or .nii.gz)</label>
              <input ref={ctFileInput} id="ct-volume" type="file" accept=".nii,.nii.gz,application/octet-stream" disabled={ctBusy}
                onChange={(event) => selectCtFile(event.target.files?.[0] || null)} />
              <label htmlFor="ct-provenance-mode">CT origin/provenance</label>
              <select id="ct-provenance-mode" value={ctProvenanceMode} disabled={ctBusy}
                onChange={(event) => selectCtProvenanceMode(event.target.value)}>
                <option value="dicom">DICOM-origin series with source and conversion records</option>
                <option value="nifti_origin">NIfTI-origin Task09 Spleen research distribution</option>
              </select>
              <button type="submit" disabled={!ctFile || ctBusy}>{ctBusy ? "Inspecting…" : "Inspect volume geometry"}</button>
            </form>
            {ctBusy && <><progress aria-label="CT workflow progress" /><p role="status" aria-live="polite">Processing the CT locally in the separate MONAI service…</p></>}
            {ctError && <p className="error" role="alert">{ctError}</p>}
            {ctInspection && (
              <div className="ct-inspection" aria-live="polite">
                <h3>Header inspection only — no inference has run</h3>
                <p><strong>Dimensions:</strong> {ctInspection.geometry.shape.join(" × ")}</p>
                <p><strong>Spacing:</strong> {ctInspection.geometry.spacing_mm.map((value) => `${value.toFixed(3)} mm`).join(" × ")}</p>
                <p><strong>Affine axis codes:</strong> {ctInspection.geometry.axis_codes.join(" / ")}</p>
                <p><strong>Exact file SHA-256:</strong> <code className="hash">{ctInspection.file_sha256}</code></p>
                <p><strong>Exact geometry SHA-256:</strong> <code className="hash">{ctInspection.geometry_sha256}</code></p>
                <p><strong>qform/sform codes:</strong> {ctInspection.geometry.qform_code} / {ctInspection.geometry.sform_code}</p>
                <p><strong>Orientation verification:</strong> not established by header inspection.</p>
                <details>
                  <summary>Exact inspected geometry object</summary>
                  <pre className="ct-geometry-json">{JSON.stringify(ctInspection.geometry, null, 2)}</pre>
                </details>
                {ctProvenanceMode === "nifti_origin" ? (
                  <>
                    <h3>NIfTI-origin research dataset</h3>
                    <p>This path does not claim DICOM source metadata or a DICOM conversion history. For the selected candidate, choose the original Task09_Spleen <code>dataset.json</code>; the API checks its test-list membership and generates a record bound to this inspected volume hash and geometry. That dataset declaration does not independently authenticate the uploaded image bytes.</p>
                    <label htmlFor="ct-dataset-metadata">Task09 Spleen dataset.json</label>
                    <input id="ct-dataset-metadata" type="file" accept="application/json,.json" disabled={ctBusy}
                      onChange={(event) => {
                        setCtDatasetMetadataFile(event.target.files?.[0] || null);
                        setCtOriginRecord(null);
                        setCtEvidence(null);
                        setCtEvidenceStatus("");
                        setCtReview(null);
                        setCtReviewName("");
                        setCtResult(null);
                        setCtAuthorizationState("awaiting_review");
                        setCtAuthorizationMessage("Generate a fresh NIfTI-origin record before requesting reviewer evidence.");
                      }} />
                    {ctDatasetMetadataFile && <p>Dataset metadata: {ctDatasetMetadataFile.name}</p>}
                    <button type="button" onClick={() => void generateCtOriginRecord()}
                      disabled={!ctDatasetMetadataFile || !ctInspection || ctBusy}>
                      Generate NIfTI-origin record
                    </button>
                    {ctOriginRecord && (
                      <div className="ct-review-summary" aria-label="Generated NIfTI-origin record">
                        <h4>Server-generated origin record (not an orientation approval)</h4>
                        <p><strong>Dataset/task/release:</strong> Medical Segmentation Decathlon · Task09_Spleen · 1.0 06/08/2018</p>
                        <p><strong>Split/image:</strong> test · ./imagesTs/spleen_1.nii.gz</p>
                        <p><strong>Distributed test label:</strong> not provided for this test image</p>
                        <p><strong>Image-byte authenticity:</strong> not independently authenticated by dataset.json membership alone</p>
                        <p><strong>Original DICOM metadata:</strong> not supplied in the dataset distribution</p>
                        <p><strong>Conversion history:</strong> unavailable</p>
                        <p><strong>Record SHA-256:</strong> <code className="hash">{ctOriginRecord.origin_record_sha256}</code></p>
                        <p><strong>Record volume geometry SHA-256:</strong> <code className="hash">{ctOriginRecord.origin_record.volume.geometry_sha256}</code></p>
                        <p>Independent human orientation review is still required. RAS header codes are not an anatomical orientation verification.</p>
                        <button type="button" onClick={downloadCtOriginRecord}>Download generated provenance record</button>
                        <button type="button" className="button-secondary" onClick={downloadCtReviewTemplate}>
                          Download unsigned review template bound to these hashes
                        </button>
                        <details>
                          <summary>Exact inspected geometry</summary>
                          <pre className="ct-geometry-json">{JSON.stringify(ctInspection.geometry, null, 2)}</pre>
                        </details>
                      </div>
                    )}
                    <div className="ct-review-requirements">
                      <h4>Independent reviewer requirements</h4>
                      <ol>
                        <li>Reviewer identity must be accountable and independent of the person preparing the submission.</li>
                        <li>Verify laterality and physical slice direction against an authoritative reference tied to this exact volume; RAS header codes alone are insufficient.</li>
                        <li>Record the reference, comparison method, explicit physical direction, review time, expiry (within seven days), and rationale.</li>
                        <li>The service operator must securely configure <code>CT_ORIENTATION_REVIEW_HMAC_KEY</code> out of band. Sign through the authorized reviewer process using that trusted key; it must never be exposed to the browser or this application.</li>
                        <li>Upload the signed JSON and wait for successful CT backend validation. Only then is segmentation enabled.</li>
                      </ol>
                      <p>The backend validates signature and exact file, geometry, origin-record, and expiry bindings. It cannot independently authenticate the named reviewer or prove the reference claims are truthful. Do not invent DICOM metadata or sign if no authoritative reference is available.</p>
                    </div>
                  </>
                ) : (
                  <>
                    <h3>DICOM-origin CT series</h3>
                    <p>Provide the actual source DICOM metadata and conversion record for this exact NIfTI. If those original records are unavailable, this DICOM-origin path must remain blocked; do not represent the Task09 NIfTI distribution as a DICOM conversion.</p>
                    <label htmlFor="ct-source-metadata">Source DICOM metadata JSON</label>
                    <input id="ct-source-metadata" type="file" accept="application/json,.json" disabled={ctBusy}
                      onChange={(event) => {
                        setCtSourceMetadataFile(event.target.files?.[0] || null);
                        setCtEvidence(null);
                        setCtEvidenceStatus("");
                        setCtReview(null);
                        setCtReviewName("");
                        setCtResult(null);
                        setCtAuthorizationState("awaiting_review");
                        setCtAuthorizationMessage("Submit matching source records and a fresh independent review for backend validation.");
                      }} />
                    {ctSourceMetadataFile && <p>Source metadata: {ctSourceMetadataFile.name}</p>}
                    <label htmlFor="ct-conversion-record">NIfTI conversion record JSON</label>
                    <input id="ct-conversion-record" type="file" accept="application/json,.json" disabled={ctBusy}
                      onChange={(event) => {
                        setCtConversionRecordFile(event.target.files?.[0] || null);
                        setCtEvidence(null);
                        setCtEvidenceStatus("");
                        setCtReview(null);
                        setCtReviewName("");
                        setCtResult(null);
                        setCtAuthorizationState("awaiting_review");
                        setCtAuthorizationMessage("Submit matching source records and a fresh independent review for backend validation.");
                      }} />
                    {ctConversionRecordFile && <p>Conversion record: {ctConversionRecordFile.name}</p>}
                  </>
                )}
                <label htmlFor="ct-review">Reviewer-signed orientation evidence (.json)</label>
                <input id="ct-review" type="file" accept="application/json,.json" disabled={ctBusy}
                  onChange={(event) => { void loadCtReview(event); }} />
                {ctProvenanceMode === "nifti_origin" && !ctOriginRecord &&
                  <p className="field-note">Generate the NIfTI-origin record first; the review must include its hash.</p>}
                {ctReviewName && <p>Review file: {ctReviewName}</p>}
                {ctReviewIssues.length > 0 && (
                  <div className="ct-review-issues" role="alert" aria-label="Reviewer evidence issues">
                    <strong>Evidence cannot be submitted:</strong>
                    <ul>{ctReviewIssues.map((issue) => <li key={issue}>{issue}</li>)}</ul>
                    <p>The browser checks field shape and bindings only. It cannot verify the HMAC without exposing the shared key; the CT backend performs cryptographic verification.</p>
                  </div>
                )}
                {ctReview && (
                  <div className="ct-review-summary" aria-label="Signed orientation evidence summary">
                    <h4>Review manifest summary</h4>
                    {ctReview.schema === "medical-imaging-nifti-origin-orientation-review/v1" ? (
                      <>
                        <p><strong>Provenance:</strong> NIfTI-origin · Task09_Spleen test</p>
                        <p><strong>Independent reviewer:</strong> {ctReview.reviewer_id}; reviewed {ctReview.reviewed_at}</p>
                        <p><strong>Orientation reference/method:</strong> {ctReview.orientation_reference} · {ctReview.orientation_method}</p>
                        <p><strong>Physical slice direction:</strong> {ctReview.physical_slice_direction}</p>
                        <p><strong>Review notes:</strong> {ctReview.review_notes}</p>
                      </>
                    ) : (
                      <>
                        <p><strong>Series UID:</strong> <code>{ctReview.series_instance_uid}</code></p>
                        <p><strong>Frame of Reference UID:</strong> <code>{ctReview.frame_of_reference_uid}</code></p>
                        <p><strong>Reviewer:</strong> {ctReview.reviewer_id}; reviewed {ctReview.reviewed_at}</p>
                        <p><strong>Converter:</strong> {ctReview.conversion_software} {ctReview.conversion_version}</p>
                        <p><strong>DICOM orientation:</strong> {ctReview.image_orientation_patient.join(", ")}</p>
                        <p><strong>First/last slice positions:</strong> {ctReview.first_image_position_patient.join(", ")} / {ctReview.last_image_position_patient.join(", ")}</p>
                        <p><strong>Review notes:</strong> {ctReview.review_notes}</p>
                      </>
                    )}
                    <p>The service checks the signature and file/geometry binding. It cannot independently prove the source evidence is truthful.</p>
                    {ctReview.schema === "medical-imaging-nifti-origin-orientation-review/v1" && ctInspection && ctOriginRecord && (
                      <p><strong>Bound volume SHA-256:</strong> <code className="hash">{ctInspection.file_sha256}</code><br />
                        <strong>Bound geometry SHA-256:</strong> <code className="hash">{ctInspection.geometry_sha256}</code><br />
                        <strong>Bound origin-record SHA-256:</strong> <code className="hash">{ctOriginRecord.origin_record_sha256}</code></p>
                    )}
                  </div>
                )}
                <button type="button" onClick={() => void validateCtEvidence()}
                  disabled={ctValidationBlockers.length > 0 || ctBusy}>
                  {ctBusy && ctEvidenceStatus.startsWith("Validating")
                    ? "Validating source evidence…"
                    : ctProvenanceMode === "nifti_origin"
                      ? "Validate NIfTI-origin evidence"
                      : "Validate DICOM source evidence"}
                </button>
                {ctValidationBlockers.length > 0 && (
                  <p className="field-note" role="status">Validation disabled: {ctValidationBlockers.join(" ")}</p>
                )}
                {ctEvidenceStatus && <p role="status" aria-live="polite">{ctEvidenceStatus}</p>}
                {ctEvidence && <p><strong>Evidence status:</strong> validated by the CT API for this inspection; expires {new Date(ctEvidence.expires_at_unix * 1000).toLocaleString()}. The token is one-use and does not prove source authenticity or anatomical correctness.</p>}
                <button type="button" onClick={() => void runCtSegmentation()}
                  disabled={ctSegmentationBlockers.length > 0 || ctBusy || Boolean(ctResult)}>
                  {ctBusy ? "Running pretrained segmentation…" : "Run pretrained spleen segmentation"}
                </button>
                {ctSegmentationBlockers.length > 0 && (
                  <p className="field-note" role="status">Segmentation disabled: {ctSegmentationBlockers.join(" ")}</p>
                )}
              </div>
            )}
          </section>
          {ctResult && (
            <section id="analysis-results" className="panel results-panel" aria-labelledby="ct-result-heading">
              <p className="eyebrow">Research results</p><h2 id="ct-result-heading">Pretrained spleen segmentation</h2>
              <p><strong>Model:</strong> {ctResult.model} · pretrained MONAI spleen segmentation.</p>
              <p><strong>HTTP response:</strong> {ctResult.httpStatus ?? "status unavailable"} · inference status: {ctResult.status}.</p>
              <p><strong>Task:</strong> Binary spleen-mask prediction. A segmentation mask is not a diagnosis.</p>
              <p>{ctResult.warning}</p>
              {ctStatistics && (
                <dl className="ct-metadata segmentation-statistics">
                  <div><dt>Mask voxels</dt><dd>{ctStatistics.totalVoxels.toLocaleString()}</dd></div>
                  <div><dt>Predicted spleen-label voxels</dt><dd>{ctStatistics.predictedVoxels.toLocaleString()}</dd></div>
                  <div><dt>Output labels</dt><dd>{ctResult.labels.join(", ")}</dd></div>
                  <div><dt>Background voxels · label 0</dt><dd>{(ctStatistics.labelCounts[0] || 0).toLocaleString()}</dd></div>
                  <div><dt>Derived mask volume</dt><dd>{ctStatistics.predictedVolumeMl.toFixed(2)} mL</dd></div>
                  <div><dt>Calculation basis</dt><dd>Predicted label count × reported voxel spacing</dd></div>
                </dl>
              )}
              <details>
                <summary>Exact output geometry</summary>
                <pre className="ct-geometry-json">{JSON.stringify(ctResult.geometry, null, 2)}</pre>
              </details>
              <CTSliceViewer geometry={ctResult.geometry} volumePreview={ctResult.volumePreview} mask={ctResult.mask} />
              <p className="results-disclaimer">Not clinically validated. Review the source orientation evidence and segmentation output independently.</p>
            </section>
          )}
          {ctResult && ctStudyId && (
            <DoctorReviewPanel key={ctStudyId} studyId={ctStudyId} token={token}
              aiSummary={`MONAI spleen segmentation completed for input SHA-256 ${ctStudyId}. A predicted binary mask is available; it is not a diagnosis.`} />
          )}
        </>
      )}

      {modality === "mri" && (
        <>
          <section className="panel workspace-panel" aria-labelledby="mri-heading" aria-busy={mriBusy}>
          <div className="section-heading">
            <div><p className="eyebrow">Four-sequence input</p><h2 id="mri-heading">3D brain MRI segmentation</h2></div>
            <span className="status-badge neutral">T1 · T1c · T2 · FLAIR</span>
          </div>
            <p className="status-pill">Task-specific pretrained MONAI BraTS workflow · research demonstration only</p>
            <p>This model supports only BraTS 2018-style primary-glioma segmentation from four aligned, skull-stripped, 1 mm isotropic NIfTI sequences. It does not diagnose, identify other tumors, or establish that the uploaded sequences meet that protocol. Do not use the output for patient-care decisions.</p>
            <p className="mri-input-order"><strong>Required model channel order:</strong> T1c (post-contrast) → T1 → T2 → FLAIR. Select one NIfTI volume for each sequence; all four must be co-registered.</p>
            <form onSubmit={runMriSegmentation} className="mri-upload-form">
              {MRI_SEQUENCES.map((sequence) => (
                <label key={sequence} className="mri-sequence-input" htmlFor={`mri-${sequence}`}>
                  {sequence === "t1c" ? "T1c (post-contrast)" : sequence.toUpperCase()} volume
                  <input ref={(input) => { mriFileInputs.current[sequence] = input; }}
                    id={`mri-${sequence}`} type="file" accept=".nii,.nii.gz,application/octet-stream"
                    disabled={mriBusy}
                    onChange={(event) => selectMriFile(sequence, event.target.files?.[0] || null)} />
                  <span className="field-note">{mriFiles[sequence]?.name || "Select a .nii or .nii.gz volume"}</span>
                </label>
              ))}
              <p className="field-note mri-upload-note">
                {MRI_SEQUENCES.filter((sequence) => mriFiles[sequence]).length} of 4 sequence files selected.
                {" "}Each file is limited to 128 MiB and the combined upload to 512 MiB. The service checks readability, dimensions, affine, and voxel spacing, but cannot verify sequence identity or skull stripping.
              </p>
              <button type="submit" disabled={mriBusy || MRI_SEQUENCES.some((sequence) => !mriFiles[sequence])}>
                {mriBusy ? "Running pretrained segmentation…" : "Run brain tumor segmentation"}
              </button>
            </form>
            {mriBusy && <MRIInferenceProgress />}
            {mriError && <p className="error" role="alert">{mriError}</p>}
          </section>
          {mriResult && (
            <section id="analysis-results" className="panel results-panel" aria-labelledby="mri-result-heading">
              <p className="eyebrow">Research results</p><h2 id="mri-result-heading">Predicted brain tumor segmentation</h2>
              <StatusMessage kind="success">Inference completed. The displayed mask and slice overlay are the model output for these four input volumes.</StatusMessage>
              <p><strong>Model:</strong> {mriResult.model}</p>
              <p><strong>Task:</strong> {mriResult.task}</p>
              <p><strong>Threshold:</strong> sigmoid probability ≥ {mriResult.threshold}; this is a segmentation cutoff, not a confidence estimate.</p>
              <p>{mriResult.warning}</p>
              <MRISliceViewer geometry={mriResult.geometry} volumePreview={mriResult.volumePreview} mask={mriResult.mask} />
              <dl className="ct-metadata mri-label-counts">
                {Object.entries(mriResult.output_classes).map(([label, title]) => (
                  <div key={label}><dt>{title} · label {label}</dt><dd>{mriResult.label_counts[label].toLocaleString()} voxels</dd></div>
                ))}
              </dl>
              <button type="button" className="button-secondary" onClick={() => {
                const url = URL.createObjectURL(new Blob([mriResult.maskNifti], { type: "application/gzip" }));
                const link = document.createElement("a");
                link.href = url;
                link.download = "brats_tumor_segmentation.nii.gz";
                link.click();
                URL.revokeObjectURL(url);
              }}>Download predicted NIfTI mask</button>
              <p><strong>Study input hash:</strong> <code className="hash">{mriStudyId}</code></p>
              <p className="results-disclaimer">Not clinically validated. The label map is a research output, not a diagnosis or treatment recommendation.</p>
            </section>
          )}
          {mriResult && mriStudyId && (
            <DoctorReviewPanel key={mriStudyId} studyId={mriStudyId} token={token}
              aiSummary={`${mriResult.model} returned a 3D brain-tumor subregion label map for the four supplied inputs. The model is research-only, not clinically validated, and the result is not a diagnosis.`} />
          )}
        </>
      )}

      {!(modality === "xray" && predictions.length) && !(modality === "ct" && ctResult)
        && !(modality === "mri" && mriResult) && (
          <section id="analysis-results" className="panel empty-results" aria-labelledby="empty-results-heading">
            <p className="eyebrow">Results</p>
            <h2 id="empty-results-heading">No analysis results yet</h2>
            <p>Run a modality workflow to view its actual model output here. No sample or placeholder results are shown.</p>
          </section>
        )}
      <footer className="workspace-footer">Images are processed by local services. Do not upload identifying data.</footer>
      </>}
      </main>
    </div>
    );
}
