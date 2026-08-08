"""Reproducibility metadata for experiment runs."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import subprocess

from src.state import ExperimentState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKED_PACKAGES = (
    "pandas",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "joblib",
    "streamlit",
    "optuna",
)


def dataset_sha256(dataset_path: str | Path) -> str:
    """Return a stable SHA-256 fingerprint without loading the file at once."""

    digest = sha256()
    with Path(dataset_path).open("rb") as dataset_file:
        for chunk in iter(lambda: dataset_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    """Return versions for the direct runtime dependencies that are installed."""

    installed: dict[str, str] = {}
    for package in TRACKED_PACKAGES:
        try:
            installed[package] = version(package)
        except PackageNotFoundError:
            installed[package] = "not-installed"
    return installed


def git_revision(project_root: str | Path = PROJECT_ROOT) -> str | None:
    """Return the current Git commit when Git metadata is available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(project_root),
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def attach_provenance(
    state: ExperimentState,
    *,
    dataset_path: str | Path,
    created_at_utc: str | None = None,
    resumed_from_run_id: str | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> ExperimentState:
    """Attach JSON-safe provenance for the current run to an experiment state."""

    state.provenance = {
        "schema_version": 1,
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": dataset_sha256(dataset_path),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "git_revision": git_revision(project_root),
        "run_id": state.run_id,
        "resumed_from_run_id": resumed_from_run_id,
    }
    return state
