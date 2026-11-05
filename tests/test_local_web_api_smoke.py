from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "local_web_api_smoke", ROOT / "tests" / "local_web_api_smoke.py"
)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke
SPEC.loader.exec_module(smoke)


def test_node_timeout_keeps_captured_diagnostics_and_failure(monkeypatch, capsys):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["node", "live_api_smoke.mjs"],
            timeout=30,
            output="request started",
            stderr="server returned an error",
        )

    monkeypatch.setattr(smoke.subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        smoke._run_node_smoke(["node", "live_api_smoke.mjs"], cwd=ROOT)

    captured = capsys.readouterr()
    assert "Node smoke stdout before timeout:\nrequest started" in captured.err
    assert "Node smoke stderr before timeout:\nserver returned an error" in captured.err


def test_smoke_runner_uses_bounded_capture_and_timeout(monkeypatch):
    observed = {}

    def completed(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(args[0], 0, "ok\n", "")

    monkeypatch.setattr(smoke.subprocess, "run", completed)
    result = smoke._run_node_smoke(["node", "live_api_smoke.mjs"], cwd=ROOT)

    assert result.returncode == 0
    assert observed["kwargs"]["capture_output"] is True
    assert observed["kwargs"]["text"] is True
    assert observed["kwargs"]["timeout"] == 30
    assert observed["kwargs"]["cwd"] == ROOT
