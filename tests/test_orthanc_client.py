import base64
from io import BytesIO
import json
from urllib.error import HTTPError

import pytest

import orthanc_client
from orthanc_client import OrthancClient, OrthancError


STUDY = "a1b2c3d4-01234567-89abcdef-76543210-fedcba98"
INSTANCE = "fedcba98-76543210-89abcdef-01234567-a1b2c3d4"


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, limit):
        return self.payload[:limit]


def test_client_requires_auth_and_restricts_unencrypted_remote_http():
    with pytest.raises(ValueError, match="required"):
        OrthancClient("http://localhost:8042", "", "")
    with pytest.raises(ValueError, match="Unencrypted"):
        OrthancClient("http://example.org:8042", "orthanc", "secret")
    assert OrthancClient("http://orthanc:8042", "orthanc", "secret")
    assert OrthancClient("https://orthanc.example.org", "orthanc", "secret")


@pytest.mark.parametrize("url", [
    "ftp://localhost:8042",
    "http://example.org:8042",
    "http://user:password@localhost:8042",
    "http://localhost:8042/orthanc",
    "http://localhost:8042?token=hidden",
    "http://localhost:8042/#fragment",
    "http://localhost:bad-port",
    " http://localhost:8042",
])
def test_client_rejects_unsafe_or_non_origin_urls(url):
    with pytest.raises(ValueError):
        OrthancClient(url, "reviewer", "password")


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), "not-a-number"])
def test_client_rejects_invalid_timeouts(timeout):
    with pytest.raises(ValueError, match="finite positive"):
        OrthancClient("http://localhost:8042", "reviewer", "password", timeout=timeout)


def test_client_lists_only_opaque_ids_and_downloads_instance_without_logging(monkeypatch):
    replies = {
        "/system": {"Version": "1.12.8", "Name": "private server"},
        "/studies": [STUDY],
        f"/studies/{STUDY}/instances": [{"ID": INSTANCE, "PatientName": "not returned"}],
        f"/instances/{INSTANCE}/file": b"DICM payload",
    }
    seen = []

    def fake_urlopen(request, timeout):
        seen.append((request.full_url, request.get_header("Authorization"), timeout))
        path = request.full_url.removeprefix("http://orthanc:8042")
        payload = replies[path]
        if isinstance(payload, bytes):
            return _Response(payload)
        return _Response(json.dumps(payload).encode())

    monkeypatch.setattr(orthanc_client, "_open_url", fake_urlopen)
    client = OrthancClient("http://orthanc:8042", "reviewer", "password", timeout=2)
    assert client.system_version() == "1.12.8"
    assert client.list_studies() == [STUDY]
    assert client.list_instances(STUDY) == [INSTANCE]
    assert client.download_instance(INSTANCE) == b"DICM payload"
    assert len(seen) == 4
    assert all(item[1] == "Basic " + base64.b64encode(b"reviewer:password").decode() for item in seen)
    assert all(item[2] == 2.0 for item in seen)
    assert all("PatientName" not in url for url, *_ in seen)


def test_client_rejects_malformed_ids_before_request(monkeypatch):
    monkeypatch.setattr(orthanc_client, "_open_url", lambda *_a, **_k: pytest.fail("network should not be called"))
    client = OrthancClient("http://localhost:8042", "reviewer", "password")
    with pytest.raises(ValueError, match="identifier"):
        client.download_instance("../system")
    with pytest.raises(ValueError, match="identifier"):
        client.list_instances("a" * 24)


def test_client_sanitizes_http_error_body(monkeypatch):
    def fail(*_args, **_kwargs):
        raise HTTPError("http://localhost", 401, "unauthorized", {}, BytesIO(b"PatientName leaked"))

    monkeypatch.setattr(orthanc_client, "_open_url", fail)
    client = OrthancClient("http://localhost:8042", "reviewer", "password")
    with pytest.raises(OrthancError, match="HTTP 401") as error:
        client.system_version()
    assert "PatientName" not in str(error.value)


@pytest.mark.parametrize("status", [401, 500])
def test_client_sanitizes_server_errors(monkeypatch, status):
    def fail(*_args, **_kwargs):
        raise HTTPError("http://localhost", status, "server failure", {}, BytesIO(b"private response"))

    monkeypatch.setattr(orthanc_client, "_open_url", fail)
    client = OrthancClient("http://localhost:8042", "reviewer", "password")
    with pytest.raises(OrthancError, match=f"HTTP {status}") as error:
        client.list_studies()
    assert "private response" not in str(error.value)


def test_client_maps_timeout_and_network_errors_to_sanitized_error(monkeypatch):
    def timeout(*_args, **_kwargs):
        raise TimeoutError("private host details")

    monkeypatch.setattr(orthanc_client, "_open_url", timeout)
    client = OrthancClient("http://localhost:8042", "reviewer", "password")
    with pytest.raises(OrthancError, match="Could not connect") as error:
        client.system_version()
    assert "private host details" not in str(error.value)


@pytest.mark.parametrize("payload", [b"not json", b"{}", b'["not-an-orthanc-id"]'])
def test_client_rejects_malformed_study_responses(monkeypatch, payload):
    monkeypatch.setattr(orthanc_client, "_open_url", lambda *_args, **_kwargs: _Response(payload))
    client = OrthancClient("http://localhost:8042", "reviewer", "password")
    with pytest.raises(OrthancError):
        client.list_studies()


def test_client_rejects_malformed_instance_list(monkeypatch):
    payload = json.dumps([{"PatientName": "must not escape"}]).encode()
    monkeypatch.setattr(orthanc_client, "_open_url", lambda *_args, **_kwargs: _Response(payload))
    client = OrthancClient("http://localhost:8042", "reviewer", "password")
    with pytest.raises(OrthancError, match="instance identifiers"):
        client.list_instances(STUDY)


def test_client_bounds_instance_response_size(monkeypatch):
    monkeypatch.setattr(orthanc_client, "_MAX_RESPONSE_BYTES", 4)
    monkeypatch.setattr(orthanc_client, "_open_url", lambda *_args, **_kwargs: _Response(b"12345"))
    client = OrthancClient("http://localhost:8042", "reviewer", "password")
    with pytest.raises(OrthancError, match="size limit"):
        client.download_instance(INSTANCE)


def test_redirects_are_not_followed():
    from orthanc_client import _RejectRedirects

    handler = _RejectRedirects()
    assert handler.redirect_request(None, None, 302, "Found", {}, "https://elsewhere.example") is None
