import { useState } from "react";
import { getStudy, saveReview, getAuditHistory } from "./reviewApi.js";

const VALID_STATUSES = ["Pending", "In Review", "Reviewed", "Requires Further Assessment"];

export default function DoctorReviewPanel({ studyId, aiSummary, token }) {
  const [review, setReview] = useState({ status: "Pending", assessment: "", notes: "" });
  const [savedReview, setSavedReview] = useState(null);
  const [audit, setAudit] = useState([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [showAudit, setShowAudit] = useState(false);

  async function retrieve() {
    if (!token) {
      setError("Sign in to retrieve a review.");
      return;
    }
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const data = await getStudy(studyId, token);
      setReview({
        status: data.review?.status || "Pending",
        assessment: data.review?.assessment || "",
        notes: data.review?.notes || "",
      });
      setSavedReview(data.review);
      if (data.review) {
        const history = await getAuditHistory(studyId, token);
        setAudit(history || []);
        setMessage("Saved review retrieved for this study.");
      } else {
        setMessage("No saved review exists for this study yet.");
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not retrieve the review.");
    } finally {
      setBusy(false);
    }
  }

  async function save(event) {
    event.preventDefault();
    if (!token) {
      setError("Sign in to save a review.");
      return;
    }
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const record = await saveReview(studyId, review.status, review.assessment, review.notes, token);
      setSavedReview(record);
      setMessage("Review saved.");
      const history = await getAuditHistory(studyId, token);
      setAudit(history || []);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not save the review.");
    } finally {
      setBusy(false);
    }
  }

  function update(field, value) {
    setReview((current) => ({ ...current, [field]: value }));
    setMessage("");
  }

  return (
    <section id="doctor-review" className="panel review-panel" aria-labelledby="doctor-review-heading" aria-busy={busy}>
      <div className="section-heading">
        <div>
          <p className="eyebrow">Human review</p>
          <h2 id="doctor-review-heading">Doctor Review</h2>
        </div>
        <span className={`status-badge ${review.status === "Reviewed" ? "success" : "pending"}`}>
          {review.status}
        </span>
      </div>
      {!token && (
        <p className="error" role="alert">Please sign in to access doctor review.</p>
      )}
      <p className="muted">AI output and clinician-entered review are stored separately. AI analysis never marks a study reviewed or approved.</p>
      <div className="review-ai-summary">
        <h3>AI-generated study output</h3>
        <p>{savedReview?.ai_results ? JSON.stringify(savedReview.ai_results, null, 2) : aiSummary}</p>
        <p className="field-note">This snapshot is supplied by the UI and preserved after initial save; the review service cannot independently verify its model origin.</p>
      </div>
      {token && (
        <>
          <label htmlFor="review-status">Review status</label>
          <select id="review-status" value={review.status} onChange={(event) => update("status", event.target.value)} disabled={busy}>
            {VALID_STATUSES.map((status) => (
              <option key={status} value={status}>{status}</option>
            ))}
          </select>
          <label htmlFor="review-assessment">Clinical assessment</label>
          <textarea id="review-assessment" maxLength={10000} rows={3} value={review.assessment}
            onChange={(event) => update("assessment", event.target.value)} disabled={busy}
            placeholder="Enter your clinical assessment of this study." />
          <label htmlFor="review-notes">Review notes</label>
          <textarea id="review-notes" maxLength={5000} rows={3} value={review.notes}
            onChange={(event) => update("notes", event.target.value)} disabled={busy}
            placeholder="Enter additional notes." />
          <div className="review-actions">
            <button type="button" className="button-secondary" onClick={() => void retrieve()} disabled={busy}>
              Retrieve saved review
            </button>
          </div>
          <form onSubmit={(event) => void save(event)} className="review-form">
            <div className="review-actions">
              <button type="submit" disabled={busy}>
                {busy ? "Saving…" : savedReview ? "Update review" : "Save review"}
              </button>
            </div>
          </form>
          {savedReview && (
            <>
              <button type="button" className="button-secondary" onClick={() => setShowAudit(!showAudit)} disabled={busy}>
                {showAudit ? "Hide audit trail" : "Show audit trail"}
              </button>
              {showAudit && audit.length > 0 && (
                <div className="audit-trail">
                  <h4>Audit trail</h4>
                  <ul>
                    {audit.map((entry) => (
                      <li key={entry.id}>
                        <strong>{entry.event_type}</strong> by {entry.actor_username || entry.actor_id} at {new Date(entry.timestamp_utc).toLocaleString()}
                        {entry.before_json && (<> <br /> <span className="field-note">Before: {entry.before_json}</span> </>)}
                        {entry.after_json && (<> <br /> <span className="field-note">After: {entry.after_json}</span> </>)}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              <p className="field-note" role="status">
                {savedReview.updated_at ? `Updated ${new Date(savedReview.updated_at).toLocaleString()}` : ""}
              </p>
            </>
          )}
          {message && <p className="success-message" role="status">{message}</p>}
          {error && <p className="error" role="alert">{error}</p>}
        </>
      )}
      <p className="field-note">Local research storage only. Do not enter patient-identifying information. Reviewer identity and clinical authorization require a proper identity-management system before deployment.</p>
    </section>
  );
}
