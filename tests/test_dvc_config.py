from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_dvc_configuration_disables_telemetry_and_ignores_local_sensitive_artifacts():
    config = (ROOT / ".dvc" / "config").read_text(encoding="utf-8")
    ignored = (ROOT / ".dvcignore").read_text(encoding="utf-8")
    internal_ignored = (ROOT / ".dvc" / ".gitignore").read_text(encoding="utf-8")
    assert "analytics = false" in config
    for pattern in (
        ".venv/", ".venv-seg/", ".python310/", ".dvc/cache/",
        "web/node_modules/", "web/dist/", "data/audit_log.csv",
    ):
        assert pattern in ignored
    for trackable_pattern in ("*.dcm", "*.pt", "*.nii.gz"):
        assert trackable_pattern not in ignored
    assert "/cache" in internal_ignored
    assert "/config.local" in internal_ignored


def test_no_restricted_dataset_or_weight_pointers_are_checked_in_by_default():
    pointer_files = [path for path in ROOT.rglob("*.dvc") if path.is_file()]
    assert pointer_files == []


def test_dvc_config_is_not_ignored_by_git_but_is_excluded_from_docker_context():
    gitignore_lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    dockerignore_lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert ".dvc" not in gitignore_lines
    assert ".dvc/" not in gitignore_lines
    assert ".dvc" in dockerignore_lines


def test_dvc_config_has_no_remote_or_embedded_credentials():
    """This checkout intentionally has no remote; never echo config contents."""
    config = (ROOT / ".dvc" / "config").read_text(encoding="utf-8")
    sections = re.findall(r"(?m)^\s*\[([^]]+)\]\s*$", config)
    keys = re.findall(r"(?m)^\s*([A-Za-z0-9_.-]+)\s*=", config)
    lower_config = config.lower()
    credential_markers = (
        "password", "passwd", "token", "secret", "access_key", "secret_key",
        "credential", "authorization", "bearer ",
    )
    assert not any(section.strip().lower().startswith("remote ") for section in sections)
    assert not any(key.lower().startswith("remote.") for key in keys)
    assert not any(marker in lower_config for marker in credential_markers)
    assert not re.search(r"(?i)https?://[^\s/@]+:[^\s/@]+@", config)


def test_local_checkpoint_is_ignored_by_git_and_has_no_dvc_pointer():
    gitignore_lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    checkpoint = "bundles/spleen_ct_segmentation/models/model.pt"
    assert checkpoint in gitignore_lines
    assert (ROOT / checkpoint).is_file()
    assert not (ROOT / f"{checkpoint}.dvc").exists()
