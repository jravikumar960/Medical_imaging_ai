import json

from fastapi.testclient import TestClient

import doctor_reviews
from inference_api import api

TOKEN = "local-doctor-review-token-with-32-bytes"
STUDY_ID = "a" * 64


def _record(**overrides):
    record = {
        "status": "Pending",
        "ai_summary": "MONAI spleen mask generated; segmentation is not a diagnosis.",
        "doctor_comments": "",
        "agreement": "undetermined",
        "corrected_findings": "",
        "follow_up_recommendations": "",
        "reviewer_id": "reviewer-01",
    }
    record.update(overrides)
    return record


def test_review_persists_after_client_restart_and_connection_reopen(tmp_path, monkeypatch):
    db_path = tmp_path / "reviews.sqlite3"
    monkeypatch.setenv("MEDICAL_REVIEW_DB_PATH", str(db_path))
    monkeypatch.setenv("DOCTOR_REVIEW_API_TOKEN", TOKEN)
    headers = {"Authorization": f"Bearer {TOKEN}"}

    first_client = TestClient(api)
    created = first_client.put(f"/reviews/{STUDY_ID}", json=_record(), headers=headers)
    assert created.status_code == 200
    created_body = created.json()
    assert created_body["reviewer_id"] == "reviewer-01"
    assert created_body["reviewed_at"] is None

    # A new client represents an application restart; only the SQLite file carries state.
    restarted_client = TestClient(api)
    loaded = restarted_client.get(f"/reviews/{STUDY_ID}", headers=headers)
    assert loaded.status_code == 200
    loaded_body = loaded.json()
    assert loaded_body["ai_summary"] == _record()["ai_summary"]
    assert loaded_body["created_at"] == created_body["created_at"]
    assert loaded_body["updated_at"] == created_body["updated_at"]

    # Reopen the database through an independent connection.
    connection = doctor_reviews._connect()
    try:
        row = connection.execute(
            "SELECT record_json, created_at, updated_at FROM doctor_reviews WHERE study_id = ?",
            (STUDY_ID,),
        ).fetchone()
        assert row is not None
        persisted = json.loads(row["record_json"])
        assert persisted["status"] == "Pending"
        assert persisted["reviewer_id"] == "reviewer-01"
        assert row["created_at"] == created_body["created_at"]
    finally:
        connection.close()

    # Duplicate update preserves created_at while changing mutable review fields.
    updated = restarted_client.put(
        f"/reviews/{STUDY_ID}",
        json=_record(status="Reviewed", doctor_comments="Reviewed after restart."),
        headers=headers,
    )
    assert updated.status_code == 200
    assert updated.json()["created_at"] == created_body["created_at"]
    assert updated.json()["reviewed_at"] is not None

    final = restarted_client.get(f"/reviews/{STUDY_ID}", headers=headers)
    assert final.status_code == 200
    assert final.json()["doctor_comments"] == "Reviewed after restart."
    assert final.json()["created_at"] == created_body["created_at"]
