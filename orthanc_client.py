"""Small authenticated Orthanc REST client; never requests patient tags or logs payloads."""
from __future__ import annotations

import base64
import json
import math
import os
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


_RESOURCE_ID = re.compile(r"^[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{8}){4}$")
_MAX_RESPONSE_BYTES = 25 * 1024 * 1024


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, _request, _file, _code, _message, _headers, _new_url):
        return None


def _open_url(request, timeout):
    return build_opener(_RejectRedirects).open(request, timeout=timeout)


class OrthancError(RuntimeError):
    """Raised for configuration, network, or response errors without PHI details."""


class OrthancClient:
    def __init__(self, base_url: str, username: str, password: str, timeout: float = 10.0):
        if not isinstance(base_url, str) or not base_url or base_url != base_url.strip() or any(
            ord(character) < 32 for character in base_url
        ):
            raise ValueError("ORTHANC_URL must be a clean http(s) origin.")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.path not in {"", "/"}:
            raise ValueError("ORTHANC_URL must be an http(s) origin without a path.")
        try:
            parsed.port
        except ValueError:
            raise ValueError("ORTHANC_URL must contain a valid port.") from None
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Put Orthanc credentials in separate secret settings, not the URL.")
        if parsed.scheme == "http" and parsed.hostname.lower() not in {
            "localhost", "127.0.0.1", "::1", "orthanc"
        }:
            raise ValueError("Unencrypted Orthanc HTTP is allowed only for loopback or the private Docker service.")
        if not username or not password:
            raise ValueError("Orthanc username and password are required.")
        try:
            timeout = float(timeout)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Orthanc timeout must be a finite positive number of seconds.") from None
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Orthanc timeout must be a finite positive number of seconds.")
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.timeout = timeout

    @classmethod
    def from_environment(cls):
        return cls(
            os.environ.get("ORTHANC_URL", "http://127.0.0.1:8042"),
            os.environ.get("ORTHANC_USERNAME", ""),
            os.environ.get("ORTHANC_PASSWORD", ""),
        )

    def _request(self, path: str, expect_json: bool = True):
        token = base64.b64encode(f"{self.username}:{self.password}".encode()).decode("ascii")
        headers = {"Authorization": f"Basic {token}", "Accept": "application/json"}
        request = Request(f"{self.base_url}{path}", headers=headers)
        try:
            with _open_url(request, timeout=self.timeout) as response:
                payload = response.read(_MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            status_code = exc.code
            exc.close()
            raise OrthancError(f"Orthanc returned HTTP {status_code}.") from None
        except (URLError, TimeoutError, OSError):
            raise OrthancError("Could not connect to Orthanc.") from None
        if len(payload) > _MAX_RESPONSE_BYTES:
            raise OrthancError("Orthanc response exceeds the configured size limit.")
        if not expect_json:
            return payload
        try:
            return json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise OrthancError("Orthanc returned an invalid response.") from None

    def system_version(self) -> str:
        response = self._request("/system")
        if not isinstance(response, dict) or not isinstance(response.get("Version"), str):
            raise OrthancError("Orthanc system response did not include a version.")
        return response["Version"]

    def list_studies(self) -> list[str]:
        response = self._request("/studies")
        if not isinstance(response, list) or any(not isinstance(item, str) or not _RESOURCE_ID.fullmatch(item) for item in response):
            raise OrthancError("Orthanc returned invalid study identifiers.")
        return response

    def list_instances(self, study_id: str) -> list[str]:
        self._validate_id(study_id)
        response = self._request(f"/studies/{study_id}/instances")
        if not isinstance(response, list):
            raise OrthancError("Orthanc returned an invalid instance list.")
        ids = [item.get("ID") for item in response if isinstance(item, dict)]
        if len(ids) != len(response) or any(not isinstance(item, str) or not _RESOURCE_ID.fullmatch(item) for item in ids):
            raise OrthancError("Orthanc returned invalid instance identifiers.")
        return ids

    def download_instance(self, instance_id: str) -> bytes:
        self._validate_id(instance_id)
        return self._request(f"/instances/{instance_id}/file", expect_json=False)

    @staticmethod
    def _validate_id(resource_id: str):
        if not isinstance(resource_id, str) or not _RESOURCE_ID.fullmatch(resource_id):
            raise ValueError("Invalid Orthanc resource identifier.")
