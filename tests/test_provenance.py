"""Tests for experiment provenance and reproducibility metadata."""

from hashlib import sha256
from pathlib import Path

from src.provenance import attach_provenance, dataset_sha256
from src.reporting import create_final_report_markdown, create_final_report_summary
from src.state import ExperimentState


def test_dataset_sha256_matches_file_contents(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    contents = b"Feature,Target\n1,Yes\n2,No\n"
    dataset.write_bytes(contents)

    assert dataset_sha256(dataset) == sha256(contents).hexdigest()


def test_attach_provenance_records_reproducible_run_context(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.csv"
    contents = b"Feature,Target\n1,Yes\n2,No\n"
    dataset.write_bytes(contents)

    state = ExperimentState(
        dataset_path=str(dataset),
        target_column="Target",
        run_id="run-002",
    )

    attach_provenance(
        state,
        dataset_path=dataset,
        created_at_utc="2026-08-06T12:00:00+00:00",
        resumed_from_run_id="run-001",
        project_root=tmp_path,
    )

    provenance = state.provenance
    assert provenance["schema_version"] == 1
    assert provenance["created_at_utc"] == "2026-08-06T12:00:00+00:00"
    assert provenance["dataset_sha256"] == sha256(contents).hexdigest()
    assert provenance["python_version"]
    assert provenance["python_implementation"]
    assert provenance["platform"]
    assert provenance["package_versions"]["scikit-learn"]
    assert provenance["run_id"] == "run-002"
    assert provenance["resumed_from_run_id"] == "run-001"
    assert len(provenance["git_revision"]) == 40
    assert all(character in "0123456789abcdef" for character in provenance["git_revision"])


def test_report_exposes_provenance_and_summary_identifiers() -> None:
    state = ExperimentState(
        dataset_path="data/sample.csv",
        target_column="Target",
    )
    state.provenance = {
        "run_id": "run-002",
        "created_at_utc": "2026-08-06T12:00:00+00:00",
        "resumed_from_run_id": "run-001",
        "dataset_sha256": "a" * 64,
        "git_revision": "b" * 40,
        "python_implementation": "CPython",
        "python_version": "3.13.3",
        "platform": "Windows-test",
        "package_versions": {
            "pandas": "3.0.3",
            "scikit-learn": "1.9.0",
        },
    }

    markdown = create_final_report_markdown(state)
    summary = create_final_report_summary(state)

    assert "# 13. Reproducibility and Provenance" in markdown
    assert "# 14. Final Conclusion" in markdown
    assert "- **Run ID:** run-002" in markdown
    assert "- **Resumed from run:** run-001" in markdown
    assert f"- **Dataset SHA-256:** `{'a' * 64}`" in markdown
    assert f"- **Git revision:** `{'b' * 40}`" in markdown
    assert "| pandas | 3.0.3 |" in markdown
    assert "| scikit-learn | 1.9.0 |" in markdown
    assert summary["run_id"] == "run-002"
    assert summary["dataset_sha256"] == "a" * 64
    assert summary["git_revision"] == "b" * 40


def test_report_handles_legacy_state_without_provenance() -> None:
    state = ExperimentState(
        dataset_path="data/legacy.csv",
        target_column="Target",
    )

    markdown = create_final_report_markdown(state)

    assert "# 13. Reproducibility and Provenance" in markdown
    assert "No reproducibility metadata was recorded for this run." in markdown
