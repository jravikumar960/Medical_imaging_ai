from fastapi.testclient import TestClient

import inference_api
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


def test_review_requires_configured_access_token_and_keeps_records_local(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MEDICAL_REVIEW_DB_PATH", str(tmp_path / "reviews.sqlite3"))
    monkeypatch.delenv("DOCTOR_REVIEW_API_TOKEN", raising=False)
    client = TestClient(api)
    response = client.put(f"/reviews/{STUDY_ID}", json=_record())
    assert response.status_code == 503
    assert not (tmp_path / "reviews.sqlite3").exists()

    monkeypatch.setenv("DOCTOR_REVIEW_API_TOKEN", TOKEN)
    response = client.put(f"/reviews/{STUDY_ID}", json=_record())
    assert response.status_code == 401
    assert not (tmp_path / "reviews.sqlite3").exists()


def test_review_create_read_and_update_are_study_bound_and_not_auto_approved(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MEDICAL_REVIEW_DB_PATH", str(tmp_path / "reviews.sqlite3"))
    monkeypatch.setenv("DOCTOR_REVIEW_API_TOKEN", TOKEN)
    client = TestClient(api)
    headers = {"Authorization": f"Bearer {TOKEN}"}

    created = client.put(f"/reviews/{STUDY_ID}", json=_record(), headers=headers)
    assert created.status_code == 200
    assert created.json()["status"] == "Pending"
    assert created.json()["reviewed_at"] is None

    loaded = client.get(f"/reviews/{STUDY_ID}", headers=headers)
    assert loaded.status_code == 200
    assert loaded.json()["ai_summary"] == _record()["ai_summary"]

    updated = client.put(
        f"/reviews/{STUDY_ID}",
        json=_record(
            status="Reviewed",
            doctor_comments="Reviewed the segmentation output.",
            agreement="partially_agree",
            reviewer_id="reviewer-02",
        ),
        headers=headers,
    )
    assert updated.status_code == 200
    assert updated.json()["status"] == "Reviewed"
    assert updated.json()["reviewed_at"]
    assert updated.json()["reviewer_id"] == "reviewer-02"

    immutable = client.put(
        f"/reviews/{STUDY_ID}",
        json=_record(ai_summary="A client-supplied replacement summary."),
        headers=headers,
    )
    assert immutable.status_code == 409

    assert client.get(f"/reviews/{'b' * 64}", headers=headers).status_code == 404


def test_review_rejects_invalid_study_id_and_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDICAL_REVIEW_DB_PATH", str(tmp_path / "reviews.sqlite3"))
    monkeypatch.setenv("DOCTOR_REVIEW_API_TOKEN", TOKEN)
    client = TestClient(api)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    assert client.get("/reviews/not-a-hash", headers=headers).status_code == 422
    assert client.put(
        f"/reviews/{STUDY_ID}",
        json=_record(status="AI approved"),
        headers=headers,
    ).status_code == 422
    assert client.put(
        f"/reviews/{STUDY_ID}",
        json=_record(extra_field="unexpected"),
        headers=headers,
    ).status_code == 422


def test_review_route_cors_allows_bearer_preflight_from_local_vite():
    response = TestClient(api).options(
        f"/reviews/{STUDY_ID}",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "PUT",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://127.0.0.1:5173"
    assert response.headers.get("access-control-allow-headers")
