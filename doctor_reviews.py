"""Local, token-protected persistence for clinician-entered study reviews."""
from __future__ import annotations

import hmac
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field


router = APIRouter(prefix="/reviews", tags=["doctor reviews"])
STUDY_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ReviewRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: Literal["Pending", "In Review", "Reviewed", "Requires Further Investigation"]
    ai_summary: str = Field(min_length=1, max_length=8000)
    doctor_comments: str = Field(default="", max_length=10000)
    agreement: Literal["undetermined", "agree", "partially_agree", "disagree"] = "undetermined"
    corrected_findings: str = Field(default="", max_length=5000)
    follow_up_recommendations: str = Field(default="", max_length=5000)
    reviewer_id: str = Field(min_length=1, max_length=128)


def _database_path() -> Path:
    configured = os.environ.get("MEDICAL_REVIEW_DB_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        base = Path(local_app_data)
    else:
        base = Path.home() / ".local" / "share"
    return base / "medical-imaging-ai" / "doctor-reviews.sqlite3"


def _connect() -> sqlite3.Connection:
    path = _database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS doctor_reviews (
            study_id TEXT PRIMARY KEY,
            record_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            reviewed_at TEXT
        )
        """
    )
    return connection


@contextmanager
def _connection():
    connection = _connect()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _authorize(authorization: str | None) -> None:
    configured = os.environ.get("DOCTOR_REVIEW_API_TOKEN", "")
    if len(configured.encode("utf-8")) < 32:
        raise HTTPException(
            status_code=503,
            detail="Doctor review storage is disabled until DOCTOR_REVIEW_API_TOKEN is configured with at least 32 bytes.",
        )
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="A doctor review access token is required.")
    supplied = authorization[7:]
    if not hmac.compare_digest(supplied.encode("utf-8"), configured.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Doctor review access token is invalid.")


def _validate_study_id(study_id: str) -> None:
    if not STUDY_ID_PATTERN.fullmatch(study_id):
        raise HTTPException(status_code=422, detail="Study ID must be the lowercase SHA-256 of the analyzed input.")


@router.get("/{study_id}")
def get_review(study_id: str, authorization: str | None = Header(default=None)):
    _authorize(authorization)
    _validate_study_id(study_id)
    with _connection() as connection:
        row = connection.execute(
            "SELECT record_json, created_at, updated_at, reviewed_at FROM doctor_reviews WHERE study_id = ?",
            (study_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="No doctor review exists for this study.")
    return {
        "study_id": study_id,
        **json.loads(row["record_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "reviewed_at": row["reviewed_at"],
    }


@router.put("/{study_id}")
def save_review(
    study_id: str,
    record: ReviewRecord,
    authorization: str | None = Header(default=None),
):
    _authorize(authorization)
    _validate_study_id(study_id)
    now = datetime.now(timezone.utc).isoformat()
    reviewed_at = now if record.status in {"Reviewed", "Requires Further Investigation"} else None
    serialized = json.dumps(record.model_dump(), ensure_ascii=False, sort_keys=True)

    with _connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT record_json, created_at FROM doctor_reviews WHERE study_id = ?",
            (study_id,),
        ).fetchone()
        created_at = existing["created_at"] if existing else now
        if existing:
            previous = json.loads(existing["record_json"])
            if previous["ai_summary"] != record.ai_summary:
                raise HTTPException(
                    status_code=409,
                    detail="The AI-generated study summary is immutable after a review record is created.",
                )
        connection.execute(
            """
            INSERT INTO doctor_reviews (study_id, record_json, created_at, updated_at, reviewed_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(study_id) DO UPDATE SET
                record_json = excluded.record_json,
                updated_at = excluded.updated_at,
                reviewed_at = excluded.reviewed_at
            """,
            (study_id, serialized, created_at, now, reviewed_at),
        )
    return {
        "study_id": study_id,
        **record.model_dump(),
        "created_at": created_at,
        "updated_at": now,
        "reviewed_at": reviewed_at,
    }
