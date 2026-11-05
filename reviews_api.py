"""Doctor Review API — user-authenticated review system for imaging studies.

Endpoints (all under /doctor-reviews):

Auth:
  POST /doctor-reviews/auth/register   — create a clinician account (role always "clinician")
  POST /doctor-reviews/auth/login      — verify credentials, return session token
  POST /doctor-reviews/auth/logout     — invalidate session token

Studies and reviews:
  POST /doctor-reviews/studies                          — create/initialize a study from an inference result
  GET  /doctor-reviews/studies                          — list studies the user has access to
  GET  /doctor-reviews/studies/{study_id}               — retrieve a study and its review
  PUT  /doctor-reviews/studies/{study_id}/review        — save or update a clinician review
  GET  /doctor-reviews/studies/{study_id}/audit         — retrieve immutable audit history

Security:
  - Passwords are hashed with PBKDF2-HMAC-SHA256 (auth.py).
  - Session tokens are bearer tokens in the Authorization header.
  - Session tokens expire after REVIEW_SESSION_TTL_SECONDS (default 24 hours).
  - Study access is enforced server-side: only the owner, an assigned reviewer,
    or an administrator may read or modify a study.
  - Registration always assigns the "clinician" role; client-supplied roles are ignored.
  - The original AI result is stored as an immutable snapshot and is never
    overwritten when a review is saved.
  - Review audit trail records creation, status changes, assessment changes,
    and notes changes with actor, timestamp, and before/after values.
  - Review updates and audit entries are committed atomically in one transaction.

Development prototype notice:
  - Sessions are stored in memory and do not survive server restarts.
  - The session secret must be set via REVIEW_SESSION_SECRET in production;
    a missing secret causes the module to fail to import.
  - Rate limiting is process-local and does not span multiple workers.
  - This is loopback research software; do not expose to a network.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

import auth
import reviews_db

# Ensure SQLite schema exists before any request hits the DB.
reviews_db.init_db()

router = APIRouter(prefix="/doctor-reviews", tags=["doctor reviews"])

VALID_REVIEW_STATUSES = ("Pending", "In Review", "Reviewed", "Requires Further Assessment")

# Session TTL in seconds (default 24 hours). Override via env var.
SESSION_TTL_SECONDS = int(os.environ.get("REVIEW_SESSION_TTL_SECONDS", "86400"))

# In-memory session store (development prototype; does not survive restart).
auth_manager = auth.AuthManager(
    session_secret=os.environ.get("REVIEW_SESSION_SECRET"),
    ttl_seconds=SESSION_TTL_SECONDS,
)

# Process-local rate-limit counters: key -> list of request timestamps.
_rate_limits: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 10


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bearer_token(authorization: Optional[str]) -> Optional[str]:
    if authorization and authorization.startswith("Bearer "):
        return authorization[7:]
    return None


def _require_session(authorization: Optional[str]) -> auth.Session:
    token = _bearer_token(authorization)
    session = auth_manager.get_session(token)
    if session is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return session


def _require_study_access(
    conn,
    study_id: int,
    session: auth.Session,
    require_write: bool = False,
) -> None:
    """Verify the session user has access to the study.

    Raises 403 if the user is authenticated but lacks access.
    Raises 404 if the study does not exist (to avoid leaking study existence).
    """
    study = reviews_db.get_study(conn, study_id)
    if study is None:
        raise HTTPException(status_code=404, detail="Study not found.")
    access = reviews_db.check_study_access(conn, study_id, session.user_id)
    if access is None:
        raise HTTPException(status_code=403, detail="You do not have access to this study.")
    if require_write and access["access_role"] not in ("owner", "admin"):
        raise HTTPException(
            status_code=403, detail="Insufficient permission to modify this study."
        )


def _rate_limit(request: Request, limit_key: str) -> None:
    """Process-local rate limiter. Raises 429 when the limit is exceeded."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    timestamps = _rate_limits[limit_key]
    # Prune old entries.
    _rate_limits[limit_key] = [t for t in timestamps if t > window_start]
    timestamps = _rate_limits[limit_key]
    if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later.",
        )
    timestamps.append(now)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class AuthResponse(BaseModel):
    token: str
    username: str
    role: str


class StudyCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    study_sha256: str = Field(min_length=64, max_length=64)
    modality: str = Field(default="xray")
    ai_results_json: str = Field(min_length=1)
    ai_status: str = Field(default="completed")


class ReviewSaveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    status: str
    assessment: str = Field(default="", max_length=10000)
    notes: str = Field(default="", max_length=5000)


class StudySummary(BaseModel):
    id: int
    study_sha256: str
    modality: str
    ai_status: str
    created_at: str
    review_status: Optional[str] = None
    review_id: Optional[int] = None
    reviewer_username: Optional[str] = None


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@router.post("/auth/register", response_model=dict)
def register(request: Request, req: RegisterRequest):
    _rate_limit(request, "register")
    # Client-supplied role is ignored; public registrations are always clinician.
    db_path = reviews_db.DEFAULT_DB_PATH
    with reviews_db.get_connection(db_path) as conn:
        try:
            user_id = auth_manager.register(
                conn, req.username, req.password, role="clinician"
            )
        except auth.AuthError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    return {"user_id": user_id, "username": req.username, "role": "clinician"}


@router.post("/auth/login", response_model=AuthResponse)
def login(request: Request, req: LoginRequest):
    _rate_limit(request, "login")
    db_path = reviews_db.DEFAULT_DB_PATH
    with reviews_db.get_connection(db_path) as conn:
        try:
            session = auth_manager.authenticate(conn, req.username, req.password)
        except auth.AuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc))
    return AuthResponse(token=session.token, username=session.username, role=session.role)


@router.post("/auth/logout")
def logout(authorization: Optional[str] = Header(default=None)):
    token = _bearer_token(authorization)
    auth_manager.logout(token)
    return {"status": "logged_out"}


# ---------------------------------------------------------------------------
# Study endpoints
# ---------------------------------------------------------------------------

@router.post("/studies", response_model=dict)
def create_study(
    req: StudyCreateRequest,
    authorization: Optional[str] = Header(default=None),
):
    session = _require_session(authorization)
    # Validate SHA-256 format
    if not all(c in "0123456789abcdef" for c in req.study_sha256):
        raise HTTPException(status_code=400, detail="study_sha256 must be a lowercase hex SHA-256.")
    # Validate AI results JSON
    try:
        json.loads(req.ai_results_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="ai_results_json must be valid JSON.")
    db_path = reviews_db.DEFAULT_DB_PATH
    with reviews_db.get_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = reviews_db.get_study_by_sha256(conn, req.study_sha256)
        if existing is not None:
            if reviews_db.check_study_access(conn, existing["id"], session.user_id) is None:
                raise HTTPException(
                    status_code=409,
                    detail="A matching study record exists but is not accessible to this account.",
                )
            study = existing
            was_existing = True
        else:
            study_id = reviews_db.create_study(
                conn,
                req.study_sha256,
                req.modality,
                req.ai_results_json,
                req.ai_status,
                created_by=session.user_id,
            )
            study = reviews_db.get_study(conn, study_id)
            was_existing = False
        if study is None:
            raise HTTPException(status_code=500, detail="Study record could not be read after persistence.")
        return {
            "study_id": study["id"],
            "study_sha256": study["study_sha256"],
            "modality": study["modality"],
            "ai_results": json.loads(study["ai_results_json"]),
            "ai_status": study["ai_status"],
            "created_at": study["created_at"],
            "existing": was_existing,
        }


@router.get("/studies", response_model=list[StudySummary])
def list_studies(
    authorization: Optional[str] = Header(default=None),
    status: Optional[str] = Query(default=None),
    modality: Optional[str] = Query(default=None),
):
    session = _require_session(authorization)
    if status is not None and status not in VALID_REVIEW_STATUSES:
        raise HTTPException(status_code=400, detail=f"Invalid status filter: {status!r}")
    db_path = reviews_db.DEFAULT_DB_PATH
    with reviews_db.get_connection(db_path) as conn:
        rows = reviews_db.list_studies(
            conn,
            status_filter=status,
            modality_filter=modality,
            user_id=session.user_id,
        )
    return [StudySummary(**reviews_db._row_to_dict(r)) for r in rows]


@router.get("/studies/{study_id}", response_model=dict)
def get_study(study_id: str, authorization: Optional[str] = Header(default=None)):
    session = _require_session(authorization)
    try:
        sid = int(study_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Study not found.")
    db_path = reviews_db.DEFAULT_DB_PATH
    with reviews_db.get_connection(db_path) as conn:
        _require_study_access(conn, sid, session, require_write=False)
        study = reviews_db.get_study(conn, sid)
        review = reviews_db.get_review_by_study(conn, sid)
    result = {
        "id": study["id"],
        "study_sha256": study["study_sha256"],
        "modality": study["modality"],
        "ai_results": json.loads(study["ai_results_json"]),
        "ai_status": study["ai_status"],
        "created_at": study["created_at"],
        "review": None,
    }
    if review is not None:
        result["review"] = {
            "id": review["id"],
            "status": review["status"],
            "assessment": review["assessment"],
            "notes": review["notes"],
            "reviewer_id": review["reviewer_id"],
            "reviewer_username": review["reviewer_username"],
            "created_at": review["created_at"],
            "updated_at": review["updated_at"],
        }
    return result


@router.put("/studies/{study_id}/review", response_model=dict)
def save_review(
    study_id: str,
    req: ReviewSaveRequest,
    authorization: Optional[str] = Header(default=None),
):
    session = _require_session(authorization)
    try:
        sid = int(study_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Study not found.")
    if req.status not in VALID_REVIEW_STATUSES:
        raise HTTPException(status_code=400, detail=f"Invalid review status: {req.status!r}")
    db_path = reviews_db.DEFAULT_DB_PATH
    with reviews_db.get_connection(db_path) as conn:
        _require_study_access(conn, sid, session, require_write=True)
        existing = reviews_db.get_review_by_study(conn, sid)
        if existing is None:
            review_id = reviews_db.create_review(
                conn, sid, session.user_id,
                status=req.status, assessment=req.assessment, notes=req.notes,
            )
        else:
            # Only the original reviewer or an admin may update.
            if existing["reviewer_id"] != session.user_id:
                access = reviews_db.check_study_access(conn, sid, session)
                if access is None or access["access_role"] != "admin":
                    raise HTTPException(
                        status_code=403,
                        detail="Only the original reviewer or an administrator may update this review.",
                    )
            reviews_db.update_review(
                conn, int(existing["id"]), session.user_id,
                status=req.status, assessment=req.assessment, notes=req.notes,
            )
            review_id = int(existing["id"])
    return {"review_id": review_id, "study_id": sid, "status": req.status}


@router.get("/studies/{study_id}/audit", response_model=list[dict])
def get_audit(study_id: str, authorization: Optional[str] = Header(default=None)):
    session = _require_session(authorization)
    try:
        sid = int(study_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Study not found.")
    db_path = reviews_db.DEFAULT_DB_PATH
    with reviews_db.get_connection(db_path) as conn:
        _require_study_access(conn, sid, session, require_write=False)
        review = reviews_db.get_review_by_study(conn, sid)
        if review is None:
            raise HTTPException(status_code=404, detail="No review exists for this study.")
        rows = reviews_db.get_audit_history(conn, int(review["id"]))
    return [reviews_db._row_to_dict(r) for r in rows]
