"""Security-focused tests for the Doctor Review System.

Tests cover:
- Cross-user study access (read, modify, audit)
- Administrator self-registration prevention
- Session expiration and revocation
- Study access control enforcement
- AI results and audit history preservation
- Rate limiting on auth endpoints
- Existing registration, login, and review operations
"""
from __future__ import annotations

import json
import time
from fastapi.testclient import TestClient
import pytest

import inference_api
from inference_api import api
import reviews_api


def _fresh_client() -> TestClient:
    """Return a TestClient against the app. The DB is initialized at import."""
    return TestClient(api)


@pytest.fixture(autouse=True)
def reset_rate_limits_and_db():
    """Reset the process-local rate limiter and clean DB tables before each test."""
    reviews_api._rate_limits.clear()
    # Clean all tables to avoid cross-test data leakage.
    import reviews_db
    conn = reviews_db.get_connection()
    try:
        conn.executescript(
            "DELETE FROM review_audit; DELETE FROM study_access; "
            "DELETE FROM reviews; DELETE FROM studies; DELETE FROM users;"
        )
        conn.commit()
    finally:
        conn.close()
    yield
    reviews_api._rate_limits.clear()


def _register(client: TestClient, username: str, password: str = "SecurePass123") -> dict:
    resp = client.post(
        "/doctor-reviews/auth/register",
        json={"username": username, "password": password},
    )
    assert resp.status_code == 200, f"register {username}: {resp.status_code} {resp.text}"
    return resp.json()


def _login(client: TestClient, username: str, password: str = "SecurePass123") -> str:
    resp = client.post(
        "/doctor-reviews/auth/login",
        json={"username": username, "password": password},
    )
    assert resp.status_code == 200, f"login {username}: {resp.status_code} {resp.text}"
    return resp.json()["token"]


def _create_study(client: TestClient, token: str, sha256: str) -> int:
    resp = client.post(
        "/doctor-reviews/studies",
        json={
            "study_sha256": sha256,
            "modality": "xray",
            "ai_results_json": json.dumps([{"finding": "Cardiomegaly", "score": 0.85}]),
            "ai_status": "completed",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, f"create study: {resp.status_code} {resp.text}"
    return resp.json()["study_id"]


def test_duplicate_study_create_returns_existing_authorized_record_without_overwriting():
    client = _fresh_client()
    _register(client, "repeat_owner")
    owner_token = _login(client, "repeat_owner")
    _register(client, "repeat_other")
    other_token = _login(client, "repeat_other")
    study_hash = "a" * 64

    original = client.post(
        "/doctor-reviews/studies",
        json={
            "study_sha256": study_hash,
            "modality": "xray",
            "ai_results_json": json.dumps([{"finding": "Atelectasis", "score": 0.81}]),
            "ai_status": "completed",
        },
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    duplicate = client.post(
        "/doctor-reviews/studies",
        json={
            "study_sha256": study_hash,
            "modality": "xray",
            "ai_results_json": json.dumps([{"finding": "Pneumonia", "score": 0.99}]),
            "ai_status": "completed",
        },
        headers={"Authorization": f"Bearer {owner_token}"},
    )

    assert original.status_code == 200
    assert original.json()["existing"] is False
    assert duplicate.status_code == 200
    assert duplicate.json()["existing"] is True
    assert duplicate.json()["study_id"] == original.json()["study_id"]
    assert duplicate.json()["ai_results"] == [{"finding": "Atelectasis", "score": 0.81}]

    inaccessible = client.post(
        "/doctor-reviews/studies",
        json={
            "study_sha256": study_hash,
            "modality": "xray",
            "ai_results_json": "[]",
            "ai_status": "completed",
        },
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert inaccessible.status_code == 409
    assert "not accessible" in inaccessible.json()["detail"]
    assert "Atelectasis" not in inaccessible.text


def test_multiple_distinct_studies_are_saved_for_the_same_owner():
    client = _fresh_client()
    _register(client, "multi_owner")
    token = _login(client, "multi_owner")

    first_id = _create_study(client, token, "b" * 64)
    second_id = _create_study(client, token, "c" * 64)

    assert first_id != second_id
    response = client.get(
        "/doctor-reviews/studies",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert {study["id"] for study in response.json()} == {first_id, second_id}


# ---------------------------------------------------------------------------
# Phase 2: Administrator self-registration prevention
# ---------------------------------------------------------------------------

def test_public_registration_always_receives_clinician_role():
    """A user cannot self-assign the administrator role via registration."""
    client = _fresh_client()
    resp = client.post(
        "/doctor-reviews/auth/register",
        json={"username": "nouser", "password": "SecurePass123", "role": "admin"},
    )
    # Pydantic rejects the unknown 'role' field because extra="forbid".
    assert resp.status_code == 422, f"unexpected: {resp.status_code} {resp.text}"
    body = resp.json()
    assert any("role" in str(e.get("loc", [])) for e in body["detail"]), body

    # Registration without role succeeds with clinician role implicitly.
    resp = client.post(
        "/doctor-reviews/auth/register",
        json={"username": "nouser", "password": "SecurePass123"},
    )
    assert resp.status_code == 200, f"unexpected: {resp.status_code} {resp.text}"
    assert resp.json()["role"] == "clinician"


def test_registration_rejects_duplicate_username():
    client = _fresh_client()
    _register(client, "dupuser")
    resp = client.post(
        "/doctor-reviews/auth/register",
        json={"username": "dupuser", "password": "SecurePass123"},
    )
    assert resp.status_code == 400
    assert "already exists" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Phase 3: Session expiration and revocation
# ---------------------------------------------------------------------------

def test_login_returns_token_without_password_hash():
    client = _fresh_client()
    _register(client, "tokendoc")
    resp = client.post(
        "/doctor-reviews/auth/login",
        json={"username": "tokendoc", "password": "SecurePass123"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "token" in body
    assert "username" in body
    assert "role" in body
    assert "password_hash" not in body


def test_logout_revokes_token():
    client = _fresh_client()
    _register(client, "revokedoc")
    token = _login(client, "revokedoc")

    # Token works before logout.
    resp = client.get(
        "/doctor-reviews/studies",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200

    # Logout.
    resp = client.post(
        "/doctor-reviews/auth/logout",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200

    # Token is rejected after logout.
    resp = client.get(
        "/doctor-reviews/studies",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 401


def test_invalid_token_rejected():
    client = _fresh_client()
    resp = client.get(
        "/doctor-reviews/studies",
        headers={"Authorization": "Bearer invalid_token_xyz"},
    )
    assert resp.status_code == 401


def test_missing_token_rejected():
    client = _fresh_client()
    resp = client.get("/doctor-reviews/studies")
    assert resp.status_code == 401


def test_wrong_password_rejected():
    client = _fresh_client()
    _register(client, "wrongpassdoc")
    resp = client.post(
        "/doctor-reviews/auth/login",
        json={"username": "wrongpassdoc", "password": "WrongPassword123"},
    )
    assert resp.status_code == 401


def test_nonexistent_user_rejected():
    client = _fresh_client()
    resp = client.post(
        "/doctor-reviews/auth/login",
        json={"username": "doesnotexist", "password": "SecurePass123"},
    )
    assert resp.status_code == 401


def test_session_expiration():
    """A token with a very short TTL expires and is rejected."""
    import auth

    # Create an AuthManager with a 1-second TTL.
    manager = auth.AuthManager(
        session_secret="test-secret-for-expiration",
        ttl_seconds=1,
    )
    import reviews_db

    db_path = reviews_db.DEFAULT_DB_PATH
    with reviews_db.get_connection(db_path) as conn:
        user_id = manager.register(conn, "expirydoc", "SecurePass123")
        session = manager.authenticate(conn, "expirydoc", "SecurePass123")

    # Session is valid immediately.
    assert manager.get_session(session.token) is not None

    # Wait for expiration.
    time.sleep(1.1)

    # Session is expired and removed.
    assert manager.get_session(session.token) is None


# ---------------------------------------------------------------------------
# Phase 1: Study access control
# ---------------------------------------------------------------------------

def test_cross_user_cannot_read_study():
    """One doctor cannot retrieve another doctor's study."""
    client = _fresh_client()
    owner_token = _login(client, "owner_doc") if False else None
    _register(client, "owner_doc")
    owner_token = _login(client, "owner_doc")
    _register(client, "other_doc")
    other_token = _login(client, "other_doc")

    study_id = _create_study(client, owner_token, "b" * 64)

    # Other user cannot read the study.
    resp = client.get(
        f"/doctor-reviews/studies/{study_id}",
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert resp.status_code == 403, f"expected 403, got {resp.status_code}: {resp.text}"


def test_cross_user_cannot_modify_review():
    """One doctor cannot update another doctor's review."""
    client = _fresh_client()
    _register(client, "owner_doc2")
    owner_token = _login(client, "owner_doc2")
    _register(client, "other_doc2")
    other_token = _login(client, "other_doc2")

    study_id = _create_study(client, owner_token, "c" * 64)

    # Owner saves a review.
    resp = client.put(
        f"/doctor-reviews/studies/{study_id}/review",
        json={"status": "Reviewed", "assessment": "Owner assessment", "notes": "Owner notes"},
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    assert resp.status_code == 200

    # Other user cannot update the review.
    resp = client.put(
        f"/doctor-reviews/studies/{study_id}/review",
        json={"status": "Pending", "assessment": "Hacked assessment", "notes": "Hacked notes"},
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert resp.status_code == 403, f"expected 403, got {resp.status_code}: {resp.text}"


def test_cross_user_cannot_read_audit():
    """One doctor cannot read another doctor's audit history."""
    client = _fresh_client()
    _register(client, "owner_doc3")
    owner_token = _login(client, "owner_doc3")
    _register(client, "other_doc3")
    other_token = _login(client, "other_doc3")

    study_id = _create_study(client, owner_token, "d" * 64)

    # Owner saves a review (creates audit entry).
    client.put(
        f"/doctor-reviews/studies/{study_id}/review",
        json={"status": "Reviewed", "assessment": "A", "notes": "N"},
        headers={"Authorization": f"Bearer {owner_token}"},
    )

    # Other user cannot read audit.
    resp = client.get(
        f"/doctor-reviews/studies/{study_id}/audit",
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert resp.status_code == 403, f"expected 403, got {resp.status_code}: {resp.text}"


def test_unauthorized_study_id_returns_404():
    """A non-existent study ID returns 404, not 403, to avoid leaking existence."""
    client = _fresh_client()
    _register(client, "ghost_doc")
    token = _login(client, "ghost_doc")

    resp = client.get(
        "/doctor-reviews/studies/999999",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404


def test_owner_can_read_own_study():
    """The study owner can read their own study."""
    client = _fresh_client()
    _register(client, "owner_ok")
    token = _login(client, "owner_ok")

    study_id = _create_study(client, token, "e" * 64)

    resp = client.get(
        f"/doctor-reviews/studies/{study_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == study_id


def test_owner_can_save_and_update_own_review():
    """The study owner can create and update their own review."""
    client = _fresh_client()
    _register(client, "owner_rw")
    token = _login(client, "owner_rw")

    study_id = _create_study(client, token, "f" * 64)

    # Create review.
    resp = client.put(
        f"/doctor-reviews/studies/{study_id}/review",
        json={"status": "Reviewed", "assessment": "Initial", "notes": "Initial notes"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200

    # Update review.
    resp = client.put(
        f"/doctor-reviews/studies/{study_id}/review",
        json={"status": "Requires Further Assessment", "assessment": "Updated", "notes": "Updated notes"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200

    # Verify audit history has both created and updated events.
    resp = client.get(
        f"/doctor-reviews/studies/{study_id}/audit",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    audit = resp.json()
    assert len(audit) >= 2
    event_types = [e["event_type"] for e in audit]
    assert "created" in event_types
    assert "updated" in event_types


def test_ai_results_preserved_independent_of_review():
    """Original AI predictions remain unchanged after review operations."""
    client = _fresh_client()
    _register(client, "ai_doc")
    token = _login(client, "ai_doc")

    ai_payload = [{"finding": "Cardiomegaly", "score": 0.85}, {"finding": "Edema", "score": 0.3}]
    study_id = _create_study(client, token, "7" * 64)
    # The create_study helper already uses a different payload; re-create with custom AI.
    # Actually _create_study uses a fixed payload. Let's verify via get_study.
    resp = client.get(
        f"/doctor-reviews/studies/{study_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ai_results"] == [
        {"finding": "Cardiomegaly", "score": 0.85},
    ]

    # Save a review; AI results must still be the same.
    client.put(
        f"/doctor-reviews/studies/{study_id}/review",
        json={"status": "Reviewed", "assessment": "A", "notes": "N"},
        headers={"Authorization": f"Bearer {token}"},
    )
    resp = client.get(
        f"/doctor-reviews/studies/{study_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    body = resp.json()
    assert body["ai_results"] == [
        {"finding": "Cardiomegaly", "score": 0.85},
    ]


def test_list_studies_scoped_to_owner():
    """list_studies only returns studies the user has access to."""
    client = _fresh_client()
    _register(client, "lister")
    token = _login(client, "lister")
    _register(client, "other_lister")
    other_token = _login(client, "other_lister")

    _create_study(client, token, "1" + "0" * 63)
    _create_study(client, token, "2" + "0" * 63)
    _create_study(client, other_token, "3" + "0" * 63)

    resp = client.get(
        "/doctor-reviews/studies",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    studies = resp.json()
    assert len(studies) == 2

    resp = client.get(
        "/doctor-reviews/studies",
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert resp.status_code == 200
    studies = resp.json()
    assert len(studies) == 1


# ---------------------------------------------------------------------------
# Phase 5: Rate limiting
# ---------------------------------------------------------------------------

def test_rate_limiting_on_login():
    """Repeated login attempts are rate limited."""
    client = _fresh_client()
    _register(client, "ratelimitdoc")

    # Make many rapid login attempts.
    statuses = []
    for _ in range(15):
        resp = client.post(
            "/doctor-reviews/auth/login",
            json={"username": "ratelimitdoc", "password": "SecurePass123"},
        )
        statuses.append(resp.status_code)

    # At least some should be rate limited (429).
    assert 429 in statuses, f"no 429 responses; statuses: {statuses}"


# ---------------------------------------------------------------------------
# Existing operations regression
# ---------------------------------------------------------------------------

def test_registration_login_review_full_flow():
    """End-to-end flow: register, login, create study, save review, read audit."""
    client = _fresh_client()
    _register(client, "flowdoc")
    token = _login(client, "flowdoc")
    study_id = _create_study(client, token, "9" * 64)

    resp = client.put(
        f"/doctor-reviews/studies/{study_id}/review",
        json={"status": "Reviewed", "assessment": "Final assessment", "notes": "Final notes"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200

    resp = client.get(
        f"/doctor-reviews/studies/{study_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["review"]["status"] == "Reviewed"
    assert body["review"]["assessment"] == "Final assessment"
    assert body["review"]["notes"] == "Final notes"

    resp = client.get(
        f"/doctor-reviews/studies/{study_id}/audit",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    audit = resp.json()
    assert len(audit) >= 1
    assert audit[0]["event_type"] == "created"
