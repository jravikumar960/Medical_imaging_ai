"""Local demonstration review log; it is not a clinical audit system."""
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import tempfile
import uuid

import pandas as pd


AUDIT_COLUMNS = [
    "timestamp_utc",
    "model",
    "prediction_session_id",
    "model_finding",
    "raw_model_score",
    "review_status",
    "review_decision",
    "review_notes",
]
REVIEW_STATUSES = ("Pending Review", "Reviewed", "Needs Further Review")
REVIEW_DECISIONS = ("Agree", "Disagree", "Inconclusive")
MAX_REVIEW_NOTES_CHARS = 500
_SESSION_FINGERPRINT_KEY = "_review_source_fingerprint"
_SESSION_ID_KEY = "_review_prediction_session_id"

_ALLOWED_TRANSITIONS = {
    "Pending Review": {"Pending Review", "Reviewed", "Needs Further Review"},
    "Needs Further Review": {"Needs Further Review", "Reviewed"},
    "Reviewed": {"Reviewed", "Needs Further Review"},
}
_LEGACY_DECISIONS = {
    "Agree for demo": "Agree",
    "Disagree for demo": "Disagree",
}


def allowed_review_statuses(current_status: str = "Pending Review") -> tuple[str, ...]:
    """Return allowed next statuses, including staying at the current status."""
    if current_status not in _ALLOWED_TRANSITIONS:
        raise ValueError("Invalid current review status.")
    return tuple(status for status in REVIEW_STATUSES if status in _ALLOWED_TRANSITIONS[current_status])


def get_prediction_session_id(session_state, source_bytes: bytes) -> str:
    """Use a per-session opaque UUID; retain only a digest in volatile session state."""
    if not isinstance(source_bytes, bytes) or not source_bytes:
        raise ValueError("Prediction session requires non-empty input bytes.")
    fingerprint = hashlib.sha256(source_bytes).hexdigest()
    existing_id = session_state.get(_SESSION_ID_KEY)
    if session_state.get(_SESSION_FINGERPRINT_KEY) != fingerprint or not isinstance(existing_id, str):
        session_state[_SESSION_FINGERPRINT_KEY] = fingerprint
        session_state[_SESSION_ID_KEY] = str(uuid.uuid4())
    return session_state[_SESSION_ID_KEY]


def _normalize_history(reviews: pd.DataFrame) -> pd.DataFrame:
    """Add the current schema while retaining events from the original CSV format."""
    frame = reviews.copy()
    legacy = "review_status" not in frame.columns
    for column in AUDIT_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    for column in ("prediction_session_id", "review_decision", "review_notes"):
        frame[column] = frame[column].astype("string").fillna("").astype(str)
    frame["review_decision"] = frame["review_decision"].replace(_LEGACY_DECISIONS)
    if legacy:
        # Old rows were written only after a reviewer selected agree/disagree.
        frame["review_status"] = "Reviewed"
    else:
        frame["review_status"] = frame["review_status"].astype("string").fillna("").astype(str)
        frame.loc[frame["review_status"].eq(""), "review_status"] = "Pending Review"
    return frame.reindex(columns=AUDIT_COLUMNS)


def read_reviews(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=AUDIT_COLUMNS)
    try:
        return _normalize_history(pd.read_csv(path))
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=AUDIT_COLUMNS)


def latest_review(path: Path, prediction_session_id: str, finding: str) -> dict | None:
    """Return the most recent persisted event for one demo prediction/finding."""
    reviews = read_reviews(path)
    selected = reviews[
        reviews["prediction_session_id"].eq(prediction_session_id)
        & reviews["model_finding"].astype(str).eq(finding)
    ]
    if selected.empty:
        return None
    return selected.iloc[-1].to_dict()


def append_review(
    path: Path,
    *,
    prediction_session_id: str,
    finding: str,
    score: float,
    status: str,
    decision: str = "",
    notes: str = "",
    model_name: str,
) -> None:
    """Append a validated review event while preserving existing local CSV rows."""
    if not isinstance(prediction_session_id, str):
        raise ValueError("prediction_session_id must be a UUID generated for this demo session.")
    try:
        canonical_id = str(uuid.UUID(prediction_session_id))
    except (ValueError, AttributeError, TypeError):
        raise ValueError("prediction_session_id must be a UUID generated for this demo session.") from None
    if canonical_id != prediction_session_id:
        raise ValueError("prediction_session_id must be a canonical UUID.")
    if not isinstance(finding, str) or not finding.strip():
        raise ValueError("finding must be a non-empty label.")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be non-empty.")
    if status not in REVIEW_STATUSES:
        raise ValueError("Invalid review status.")
    if not isinstance(decision, str):
        raise ValueError("decision must be Agree, Disagree, Inconclusive, or blank for pending review.")
    decision = decision.strip()
    if status == "Pending Review":
        if decision:
            raise ValueError("Pending Review must not have a final reviewer decision.")
    elif decision not in REVIEW_DECISIONS:
        raise ValueError("Reviewed items require an Agree, Disagree, or Inconclusive decision.")
    if notes is None:
        notes = ""
    if not isinstance(notes, str):
        raise ValueError("review notes must be text.")
    notes = notes.strip()
    if len(notes) > MAX_REVIEW_NOTES_CHARS:
        raise ValueError(f"review notes must be at most {MAX_REVIEW_NOTES_CHARS} characters.")
    try:
        score = float(score)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("raw model score must be finite.") from None
    if not math.isfinite(score):
        raise ValueError("raw model score must be finite.")

    path = Path(path)
    history = read_reviews(path)
    prior_events = history[
        history["prediction_session_id"].eq(prediction_session_id)
        & history["model_finding"].astype(str).eq(finding)
    ]
    prior_status = prior_events.iloc[-1]["review_status"] if not prior_events.empty else "Pending Review"
    if status not in _ALLOWED_TRANSITIONS.get(prior_status, set()):
        raise ValueError(f"Review status cannot transition from {prior_status!r} to {status!r}.")

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name.strip(),
        "prediction_session_id": prediction_session_id,
        "model_finding": finding.strip(),
        "raw_model_score": score,
        "review_status": status,
        "review_decision": decision,
        "review_notes": notes,
    }
    if path.is_file():
        try:
            existing_columns = list(pd.read_csv(path, nrows=0).columns)
        except pd.errors.EmptyDataError:
            existing_columns = []
        if existing_columns == AUDIT_COLUMNS:
            pd.DataFrame([row], columns=AUDIT_COLUMNS).to_csv(
                path, mode="a", header=False, index=False, encoding="utf-8"
            )
            return

    updated = pd.concat([history, pd.DataFrame([row], columns=AUDIT_COLUMNS)], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", suffix=".tmp", prefix=".review-",
            dir=path.parent, delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        updated.to_csv(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def reviews_csv_bytes(reviews: pd.DataFrame) -> bytes:
    return reviews.reindex(columns=AUDIT_COLUMNS).to_csv(index=False).encode("utf-8")
