import uuid

import pandas as pd
import pytest

from audit import (
    AUDIT_COLUMNS,
    allowed_review_statuses,
    append_review,
    get_prediction_session_id,
    latest_review,
    read_reviews,
    reviews_csv_bytes,
)


MODEL = "DenseNet121"
PREDICTION_ID = "1cae914b-95b2-46c1-9c78-2f9b4ac5d9e0"


def _append(path, *, status="Reviewed", decision="Agree", notes="", prediction_id=PREDICTION_ID, score=0.73):
    append_review(
        path,
        prediction_session_id=prediction_id,
        finding="Edema",
        score=score,
        status=status,
        decision=decision,
        notes=notes,
        model_name=MODEL,
    )


def test_opaque_prediction_session_id_is_stable_until_input_changes():
    state = {}
    first = get_prediction_session_id(state, b"synthetic input A")
    assert first == get_prediction_session_id(state, b"synthetic input A")
    second = get_prediction_session_id(state, b"synthetic input B")
    assert second != first
    assert str(uuid.UUID(second)) == second
    assert set(state) == {"_review_source_fingerprint", "_review_prediction_session_id"}
    assert "synthetic input" not in str(state)


def test_pending_review_empty_notes_and_persisted_transition_history(tmp_path):
    path = tmp_path / "audit.csv"
    _append(path, status="Pending Review", decision="", notes="   ")
    pending = read_reviews(path)
    assert len(pending) == 1
    assert pending.loc[0, "review_status"] == "Pending Review"
    assert pending.loc[0, "review_decision"] == ""
    assert pending.loc[0, "review_notes"] == ""

    _append(path, status="Reviewed", decision="Agree", notes="Reviewed for demo")
    _append(path, status="Needs Further Review", decision="Inconclusive", notes="Check again")
    _append(path, status="Reviewed", decision="Disagree", notes="Final demo decision")

    history = read_reviews(path)
    assert list(history.columns) == AUDIT_COLUMNS
    assert history["review_status"].tolist() == [
        "Pending Review", "Reviewed", "Needs Further Review", "Reviewed"
    ]
    latest = latest_review(path, PREDICTION_ID, "Edema")
    assert latest["review_status"] == "Reviewed"
    assert latest["review_decision"] == "Disagree"
    assert latest["review_notes"] == "Final demo decision"
    assert latest["raw_model_score"] == 0.73
    assert latest["prediction_session_id"] == PREDICTION_ID

    exported = pd.read_csv(pd.io.common.BytesIO(reviews_csv_bytes(history)), keep_default_na=False)
    assert exported.equals(history)
    assert "filename" not in history.columns
    assert "patient_id" not in history.columns


def test_status_transition_options_and_invalid_transition(tmp_path):
    path = tmp_path / "audit.csv"
    assert allowed_review_statuses("Pending Review") == (
        "Pending Review", "Reviewed", "Needs Further Review"
    )
    assert allowed_review_statuses("Reviewed") == ("Reviewed", "Needs Further Review")

    _append(path, status="Reviewed", decision="Agree")
    with pytest.raises(ValueError, match="cannot transition"):
        _append(path, status="Pending Review", decision="")
    assert len(read_reviews(path)) == 1


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"status": "Approved", "decision": "Agree"}, "Invalid review status"),
        ({"status": "Reviewed", "decision": "Maybe"}, "Agree, Disagree, or Inconclusive"),
        ({"status": "Reviewed", "decision": ""}, "require an Agree"),
        ({"status": "Pending Review", "decision": "Agree"}, "must not have a final"),
        ({"status": "Reviewed", "decision": "Agree", "score": float("nan")}, "must be finite"),
        ({"status": "Reviewed", "decision": "Agree", "prediction_id": "patient-123"}, "must be a UUID"),
        ({"status": "Reviewed", "decision": "Agree", "notes": "x" * 501}, "at most 500"),
        ({"status": "Reviewed", "decision": "Agree", "notes": 123}, "must be text"),
    ],
)
def test_invalid_review_input_is_rejected_without_persisting(tmp_path, kwargs, message):
    path = tmp_path / "audit.csv"
    with pytest.raises(ValueError, match=message):
        _append(path, **kwargs)
    assert not path.exists()


def test_invalid_prediction_identity_and_finding_are_rejected(tmp_path):
    path = tmp_path / "audit.csv"
    with pytest.raises(ValueError, match="non-empty input bytes"):
        get_prediction_session_id({}, b"")
    with pytest.raises(ValueError, match="non-empty label"):
        append_review(
            path,
            prediction_session_id=PREDICTION_ID,
            finding=" ",
            score=0.2,
            status="Reviewed",
            decision="Agree",
            model_name=MODEL,
        )
    assert not path.exists()


def test_legacy_review_csv_is_preserved_and_normalized_on_next_save(tmp_path):
    path = tmp_path / "legacy.csv"
    pd.DataFrame([{
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "model": MODEL,
        "model_finding": "Edema",
        "raw_model_score": 0.4,
        "review_decision": "Agree for demo",
    }]).to_csv(path, index=False)

    legacy = read_reviews(path)
    assert legacy.loc[0, "review_status"] == "Reviewed"
    assert legacy.loc[0, "review_decision"] == "Agree"
    assert legacy.loc[0, "prediction_session_id"] == ""
    _append(path, status="Reviewed", decision="Inconclusive", prediction_id="f3186fa0-0f7f-4839-9ca3-cdebf6d9350b")
    upgraded = read_reviews(path)
    assert len(upgraded) == 2
    assert list(upgraded.columns) == AUDIT_COLUMNS
    assert upgraded.loc[0, "model_finding"] == "Edema"
    assert upgraded.loc[1, "review_decision"] == "Inconclusive"


def test_empty_review_history_exports_csv_header(tmp_path):
    history = read_reviews(tmp_path / "not-created.csv")
    assert history.empty
    assert reviews_csv_bytes(history).decode("utf-8").startswith(
        "timestamp_utc,model,prediction_session_id"
    )
