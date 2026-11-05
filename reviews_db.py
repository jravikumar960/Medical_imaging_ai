"""SQLite-backed persistence for doctor reviews and users.

This module replaces the CSV-only audit log with a relational store for the
Doctor Review feature. It uses Python's standard-library sqlite3 so no
third-party database driver is required.

Schema:
  users           - clinician accounts (username, password hash, role)
  studies         - imaging studies linked to inference results
  reviews         - clinician review records (one per study)
  review_audit    - immutable audit trail of review changes

Security notes:
  - Passwords are hashed with PBKDF2-HMAC-SHA256 (stdlib hashlib), never
    stored in plaintext.
  - Session tokens are generated with secrets.token_urlsafe.
  - No secrets are hardcoded; the session secret comes from an environment
    variable with a development-only fallback.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(os.environ.get("REVIEW_DB_PATH", "data/reviews.db"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Create tables if they do not exist. Idempotent."""
    conn = get_connection(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'clinician',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS studies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_sha256 TEXT UNIQUE NOT NULL,
                modality TEXT NOT NULL DEFAULT 'xray',
                ai_results_json TEXT NOT NULL,
                ai_status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_id INTEGER NOT NULL REFERENCES studies(id) ON DELETE CASCADE,
                reviewer_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                status TEXT NOT NULL DEFAULT 'Pending',
                assessment TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS study_access (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_id INTEGER NOT NULL REFERENCES studies(id) ON DELETE CASCADE,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                access_role TEXT NOT NULL DEFAULT 'owner',
                granted_at TEXT NOT NULL,
                granted_by INTEGER REFERENCES users(id),
                UNIQUE(study_id, user_id)
            );

            CREATE TABLE IF NOT EXISTS review_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id INTEGER NOT NULL REFERENCES reviews(id) ON DELETE CASCADE,
                actor_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                event_type TEXT NOT NULL,
                before_json TEXT,
                after_json TEXT,
                timestamp_utc TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_studies_sha256 ON studies(study_sha256);
            CREATE INDEX IF NOT EXISTS idx_reviews_study ON reviews(study_id);
            CREATE INDEX IF NOT EXISTS idx_review_audit_review ON review_audit(review_id);
CREATE INDEX IF NOT EXISTS idx_study_access_study ON study_access(study_id);
CREATE INDEX IF NOT EXISTS idx_study_access_user ON study_access(user_id);
            """
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def create_user(
    conn: sqlite3.Connection,
    username: str,
    password_hash: str,
    role: str = "clinician",
) -> int:
    cur = conn.execute(
        "INSERT INTO users (username, password_hash, role, created_at) "
        "VALUES (?, ?, ?, ?)",
        (username, password_hash, role, _utc_now()),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_user_by_username(conn: sqlite3.Connection, username: str) -> sqlite3.Row | None:
    cur = conn.execute(
        "SELECT id, username, password_hash, role, created_at FROM users WHERE username = ?",
        (username,),
    )
    return cur.fetchone()


def get_user(conn: sqlite3.Connection, user_id: int) -> sqlite3.Row | None:
    cur = conn.execute(
        "SELECT id, username, role, created_at FROM users WHERE id = ?",
        (user_id,),
    )
    return cur.fetchone()


# ---------------------------------------------------------------------------
# Studies
# ---------------------------------------------------------------------------

def create_study(
    conn: sqlite3.Connection,
    study_sha256: str,
    modality: str,
    ai_results_json: str,
    ai_status: str = "pending",
    created_by: int | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO studies (study_sha256, modality, ai_results_json, ai_status, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (study_sha256, modality, ai_results_json, ai_status, _utc_now()),
    )
    study_id = int(cur.lastrowid)
    if created_by is not None:
        grant_study_access(conn, study_id, created_by, access_role="owner", granted_by=created_by)
    conn.commit()
    return study_id


def get_study_by_sha256(conn: sqlite3.Connection, study_sha256: str) -> sqlite3.Row | None:
    cur = conn.execute(
        "SELECT id, study_sha256, modality, ai_results_json, ai_status, created_at "
        "FROM studies WHERE study_sha256 = ?",
        (study_sha256,),
    )
    return cur.fetchone()


def get_study(conn: sqlite3.Connection, study_id: int) -> sqlite3.Row | None:
    cur = conn.execute(
        "SELECT id, study_sha256, modality, ai_results_json, ai_status, created_at "
        "FROM studies WHERE id = ?",
        (study_id,),
    )
    return cur.fetchone()


def list_studies(
    conn: sqlite3.Connection,
    status_filter: str | None = None,
    modality_filter: str | None = None,
    user_id: int | None = None,
) -> list[sqlite3.Row]:
    query = (
        "SELECT s.id, s.study_sha256, s.modality, s.ai_results_json, s.ai_status, "
        "s.created_at, r.status AS review_status, r.id AS review_id, r.reviewer_id "
        "FROM studies s "
        "LEFT JOIN reviews r ON r.study_id = s.id"
    )
    params: list[Any] = []
    conditions: list[str] = []
    if status_filter:
        conditions.append("r.status = ?")
        params.append(status_filter)
    if modality_filter:
        conditions.append("s.modality = ?")
        params.append(modality_filter)
    if user_id is not None:
        conditions.append("s.id IN (SELECT study_id FROM study_access WHERE user_id = ?)")
        params.append(user_id)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY s.created_at DESC"
    cur = conn.execute(query, params)
    return cur.fetchall()


# ---------------------------------------------------------------------------
# Study access control
# ---------------------------------------------------------------------------

STUDY_ACCESS_ROLES = ("owner", "reviewer", "admin")


def _validate_access_role(role: str) -> None:
    if role not in STUDY_ACCESS_ROLES:
        raise ValueError(f"Invalid study access role: {role!r}")


def grant_study_access(
    conn: sqlite3.Connection,
    study_id: int,
    user_id: int,
    access_role: str = "owner",
    granted_by: int | None = None,
) -> int:
    """Grant a user access to a study. Idempotent for owner role."""
    _validate_access_role(access_role)
    now = _utc_now()
    # If an owner record already exists, update its role; otherwise insert.
    existing = conn.execute(
        "SELECT id, access_role FROM study_access WHERE study_id = ? AND user_id = ?",
        (study_id, user_id),
    ).fetchone()
    if existing is not None:
        conn.execute(
            "UPDATE study_access SET access_role = ?, granted_at = ?, granted_by = ? "
            "WHERE id = ?",
            (access_role, now, granted_by, existing["id"]),
        )
        conn.commit()
        return int(existing["id"])
    cur = conn.execute(
        "INSERT INTO study_access (study_id, user_id, access_role, granted_at, granted_by) "
        "VALUES (?, ?, ?, ?, ?)",
        (study_id, user_id, access_role, now, granted_by),
    )
    conn.commit()
    return int(cur.lastrowid)


def revoke_study_access(
    conn: sqlite3.Connection,
    study_id: int,
    user_id: int,
) -> bool:
    """Revoke a user's access to a study. Returns True if a row was removed."""
    cur = conn.execute(
        "DELETE FROM study_access WHERE study_id = ? AND user_id = ?",
        (study_id, user_id),
    )
    conn.commit()
    return cur.rowcount > 0


def check_study_access(
    conn: sqlite3.Connection,
    study_id: int,
    user_id: int,
) -> sqlite3.Row | None:
    """Return the access row if the user has access to the study, else None."""
    cur = conn.execute(
        "SELECT id, study_id, user_id, access_role, granted_at, granted_by "
        "FROM study_access WHERE study_id = ? AND user_id = ?",
        (study_id, user_id),
    )
    return cur.fetchone()


def list_study_users(
    conn: sqlite3.Connection,
    study_id: int,
) -> list[sqlite3.Row]:
    """List all users with access to a study."""
    cur = conn.execute(
        "SELECT sa.id, sa.study_id, sa.user_id, sa.access_role, sa.granted_at, "
        "sa.granted_by, u.username "
        "FROM study_access sa JOIN users u ON u.id = sa.user_id "
        "WHERE sa.study_id = ?",
        (study_id,),
    )
    return cur.fetchall()


# ---------------------------------------------------------------------------
# Reviews
# ---------------------------------------------------------------------------

VALID_STATUSES = ("Pending", "In Review", "Reviewed", "Requires Further Assessment")

def _validate_status(status: str) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid review status: {status!r}")


def create_review(
    conn: sqlite3.Connection,
    study_id: int,
    reviewer_id: int,
    status: str = "Pending",
    assessment: str = "",
    notes: str = "",
) -> int:
    _validate_status(status)
    now = _utc_now()
    cur = conn.execute(
        "INSERT INTO reviews (study_id, reviewer_id, status, assessment, notes, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (study_id, reviewer_id, status, assessment, notes, now, now),
    )
    conn.execute(
        "INSERT INTO review_audit (review_id, actor_id, event_type, before_json, after_json, timestamp_utc) "
        "VALUES (?, ?, 'created', NULL, ?, ?)",
        (int(cur.lastrowid), reviewer_id, _review_snapshot(status, assessment, notes), now),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_review_by_study(conn: sqlite3.Connection, study_id: int) -> sqlite3.Row | None:
    cur = conn.execute(
        "SELECT r.id, r.study_id, r.reviewer_id, r.status, r.assessment, r.notes, "
        "r.created_at, r.updated_at, u.username AS reviewer_username "
        "FROM reviews r JOIN users u ON u.id = r.reviewer_id "
        "WHERE r.study_id = ?",
        (study_id,),
    )
    return cur.fetchone()


def update_review(
    conn: sqlite3.Connection,
    review_id: int,
    reviewer_id: int,
    status: str | None = None,
    assessment: str | None = None,
    notes: str | None = None,
) -> None:
    cur = conn.execute(
        "SELECT status, assessment, notes FROM reviews WHERE id = ?",
        (review_id,),
    )
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"Review {review_id} not found.")

    before = _review_snapshot(row["status"], row["assessment"], row["notes"])

    updates: list[str] = []
    params: list[Any] = []
    if status is not None:
        _validate_status(status)
        updates.append("status = ?")
        params.append(status)
    if assessment is not None:
        updates.append("assessment = ?")
        params.append(assessment)
    if notes is not None:
        updates.append("notes = ?")
        params.append(notes)
    if not updates:
        return
    updates.append("updated_at = ?")
    params.append(_utc_now())
    params.append(review_id)

    conn.execute(
        f"UPDATE reviews SET {', '.join(updates)} WHERE id = ?",
        params,
    )

    after = _review_snapshot(
        status if status is not None else row["status"],
        assessment if assessment is not None else row["assessment"],
        notes if notes is not None else row["notes"],
    )
    conn.execute(
        "INSERT INTO review_audit (review_id, actor_id, event_type, before_json, after_json, timestamp_utc) "
        "VALUES (?, ?, 'updated', ?, ?, ?)",
        (review_id, reviewer_id, before, after, _utc_now()),
    )
    conn.commit()


def _review_snapshot(status: str, assessment: str, notes: str) -> str:
    import json
    return json.dumps({"status": status, "assessment": assessment, "notes": notes})


def get_audit_history(conn: sqlite3.Connection, review_id: int) -> list[sqlite3.Row]:
    cur = conn.execute(
        "SELECT a.id, a.review_id, a.actor_id, a.event_type, a.before_json, a.after_json, "
        "a.timestamp_utc, u.username AS actor_username "
        "FROM review_audit a JOIN users u ON u.id = a.actor_id "
        "WHERE a.review_id = ? ORDER BY a.timestamp_utc ASC",
        (review_id,),
    )
    return cur.fetchall()


# ---------------------------------------------------------------------------
# Review record helpers (dict conversion for API responses)
# ---------------------------------------------------------------------------

def _row_to_dict(row: sqlite3.Row) -> dict:
    return {key: row[key] for key in row.keys()}
