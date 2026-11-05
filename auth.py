"""Clinician authentication using Python standard library only.

No third-party auth packages (bcrypt, passlib, PyJWT) are required. Passwords
are hashed with PBKDF2-HMAC-SHA256 (NIST SP 800-132 approved). Session tokens
are generated with secrets.token_urlsafe and expire after a configurable TTL.

This is a development-prototype authentication layer for loopback use. It is
not a replacement for a production identity-management system (OIDC, LDAP,
etc.). The session secret must be set via REVIEW_SESSION_SECRET; a missing
secret causes the module to fail to import. Sessions are stored in memory and
do not survive server restarts, so all sessions are lost on restart and in
multi-worker deployments each worker maintains its own session store.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Optional

import reviews_db


# PBKDF2 parameters (NIST SP 800-132): at least 100,000 iterations for
# PBKDF2-HMAC-SHA256 as of current guidance.
PBKDF2_ITERATIONS = 200_000
SALT_BYTES = 32
TOKEN_BYTES = 32


@dataclass
class Session:
    user_id: int
    username: str
    role: str
    token: str
    created_at: float = field(default_factory=time.time)
    ttl_seconds: int = 86400  # 24 hours

    @property
    def expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds


def hash_password(password: str) -> str:
    """Return a PBKDF2-HMAC-SHA256 hash of ``password`` as ``salt$hash``."""
    salt = secrets.token_bytes(SALT_BYTES)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"{salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Verify ``password`` against a ``salt$hash`` string. Constant-time compare."""
    try:
        salt_hex, hash_hex = stored.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, AttributeError):
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return secrets.compare_digest(dk, expected)


def generate_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


class AuthError(Exception):
    """Raised for authentication or authorization failures."""


class AuthManager:
    def __init__(
        self,
        session_secret: Optional[str] = None,
        ttl_seconds: int = 86400,
    ):
        # The session secret is mandatory. Fail safely if it is absent.
        if session_secret is None:
            session_secret = os.environ.get("REVIEW_SESSION_SECRET")
        if not session_secret:
            raise RuntimeError(
                "REVIEW_SESSION_SECRET environment variable must be set. "
                "Set it to a strong, randomly generated value before starting the server."
            )
        self.session_secret = session_secret
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, Session] = {}

    def register(
        self,
        conn,
        username: str,
        password: str,
        role: str = "clinician",
    ) -> int:
        if not username or not username.strip():
            raise AuthError("Username is required.")
        if not password or len(password) < 8:
            raise AuthError("Password must be at least 8 characters.")
        if reviews_db.get_user_by_username(conn, username.strip()) is not None:
            raise AuthError("Username already exists.")
        return reviews_db.create_user(
            conn, username.strip(), hash_password(password), role
        )

    def authenticate(self, conn, username: str, password: str) -> Session:
        user = reviews_db.get_user_by_username(conn, username.strip())
        if user is None:
            raise AuthError("Invalid username or password.")
        if not verify_password(password, user["password_hash"]):
            raise AuthError("Invalid username or password.")
        token = generate_token()
        session = Session(
            user_id=int(user["id"]),
            username=user["username"],
            role=user["role"],
            token=token,
            ttl_seconds=self.ttl_seconds,
        )
        self._sessions[token] = session
        return session

    def get_session(self, token: Optional[str]) -> Optional[Session]:
        if not token:
            return None
        session = self._sessions.get(token)
        if session is None:
            return None
        if session.expired:
            self._sessions.pop(token, None)
            return None
        return session

    def logout(self, token: Optional[str]) -> None:
        if token:
            self._sessions.pop(token, None)


def require_auth(auth_manager: AuthManager, token: Optional[str]) -> Session:
    session = auth_manager.get_session(token)
    if session is None:
        raise AuthError("Authentication required.")
    return session


def require_role(session: Session, allowed_roles: tuple[str, ...]) -> None:
    if session.role not in allowed_roles:
        raise AuthError("Insufficient permissions.")
