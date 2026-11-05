from fnmatch import fnmatchcase
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _patterns(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _dockerignore_matches(relative_path: str, pattern: str) -> bool:
    """Approximate the positive glob forms used here; Docker itself is not installed."""
    pattern = pattern.strip().lstrip("/")
    variants = [pattern]
    if pattern.startswith("**/"):
        variants.append(pattern[3:])  # Docker's ** also matches zero directories.
    path_parts = relative_path.replace("\\", "/").split("/")
    parent_paths = ["/".join(path_parts[:i]) for i in range(1, len(path_parts))]
    for variant in variants:
        clean = variant.rstrip("/")
        if fnmatchcase(relative_path, variant) or fnmatchcase(relative_path, clean):
            return True
        if any(fnmatchcase(parent, clean) for parent in parent_paths):
            return True
        if clean.endswith("/**") and relative_path.startswith(clean[:-3].rstrip("/") + "/"):
            return True
    return False


def _requirements(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, version = line.split("==", maxsplit=1)
            result[name.lower().replace("_", "-")] = version
    return result


def test_docker_image_runs_streamlit_classifier_as_non_root_with_healthcheck():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.13-slim" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert 'CMD ["streamlit", "run", "app.py"' in dockerfile
    assert '"--server.address=0.0.0.0"' in dockerfile
    assert '"--server.port=8501"' in dockerfile
    assert "EXPOSE 8501" in dockerfile
    assert "_stcore/health" in dockerfile
    assert "python -m pip install --no-cache-dir -r requirements-docker.txt" in dockerfile
    assert "pip install --upgrade" not in dockerfile
    assert "HOME=/home/appuser" in dockerfile
    assert "/app/data /home/appuser/.torchxrayvision" in dockerfile
    assert "chown -R appuser:appuser /app/data /home/appuser/.torchxrayvision" in dockerfile
    assert "uvicorn" not in dockerfile.lower()
    assert "requirements-segmentation.txt" not in dockerfile


def test_docker_runtime_dependencies_match_project_pins_and_exclude_dev_workflows():
    project = _requirements(ROOT / "requirements.txt")
    runtime = _requirements(ROOT / "requirements-docker.txt")
    assert runtime
    assert set(runtime) <= set(project)
    assert all(runtime[name] == project[name] for name in runtime)
    assert {
        "streamlit", "torch", "torchvision", "torchxrayvision", "numpy",
        "pillow", "matplotlib", "pandas", "pydicom",
    } <= set(runtime)
    assert not ({"fastapi", "httpx", "uvicorn", "pytest"} & set(runtime))
    assert "monai" not in runtime


def test_dockerignore_excludes_sensitive_data_artifacts_secrets_and_local_state():
    patterns = _patterns(ROOT / ".dockerignore")
    ignored_paths = (
        ".venv/Scripts/python.exe",
        ".venv-seg/Lib/site-packages/monai/__init__.py",
        ".python310/python.exe",
        ".dvc/config.local",
        "dataset.dvc",
        "data/audit_log.csv",
        "datasets/study/image.png",
        "authorized_rsna/patient.csv",
        "evaluation/manifest.csv",
        "nested/patient_manifest.csv",
        "nested/patient_list.xlsx",
        "artifacts/model.pt",
        "patient.dcm",
        "nested/patient.dicom",
        "nested/scan.nii.gz",
        "nested/scan.nrrd",
        "nested/image.jpeg",
        "nested/model.safetensors",
        "nested/model.bin",
        "nested/labels.json",
        "nested/cohort.parquet",
        "nested/archive.tar.gz",
        ".env.local",
        "nested/.env.production",
        ".streamlit/secrets.toml",
        "nested/secrets.toml",
        ".aws/credentials",
        "keys/private.pem",
        "nested/cache.tmp",
        "nested/.tmp-run/output.txt",
        ".tmp-browser/output.txt",
        "nested/__pycache__/module.pyc",
        "nested/debug.log",
    )
    for relative_path in ignored_paths:
        assert any(_dockerignore_matches(relative_path, pattern) for pattern in patterns), relative_path


def test_dockerignore_keeps_application_and_runtime_source_files():
    patterns = _patterns(ROOT / ".dockerignore")
    required_paths = (
        "Dockerfile",
        "app.py",
        "audit.py",
        "imaging.py",
        "dicom_io.py",
        "orthanc_client.py",
        "requirements.txt",
        "requirements-docker.txt",
        "inference_api.py",
        "segmentation.py",
    )
    for relative_path in required_paths:
        assert not any(_dockerignore_matches(relative_path, pattern) for pattern in patterns), relative_path


def test_readme_uses_loopback_host_binding_and_describes_model_cache():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "--publish 127.0.0.1:8501:8501" in readme
    assert "/home/appuser/.torchxrayvision" in readme
    assert "Streamlit CXR classifier" in readme
