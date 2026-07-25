from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.run_context import (
    RunPaths,
    create_run_paths,
)


def test_create_run_paths_builds_expected_layout(
    tmp_path: Path,
) -> None:
    """
    One run must receive one predictable isolated directory containing
    report, state and model locations.
    """

    created_at = datetime(
        2026,
        7,
        26,
        7,
        15,
        30,
        123456,
        tzinfo=timezone.utc,
    )

    paths = create_run_paths(
        output_root=tmp_path / "outputs" / "runs",
        created_at=created_at,
        unique_suffix="ABC-123_45",
    )

    expected_run_id = (
        "20260726T071530_123456Z_abc12345"
    )

    expected_run_directory = (
        tmp_path
        / "outputs"
        / "runs"
        / expected_run_id
    )

    assert isinstance(paths, RunPaths)

    assert paths.run_id == expected_run_id

    assert (
        paths.run_directory
        == expected_run_directory
    )

    assert (
        paths.state_path
        == (
            expected_run_directory
            / "experiment_state.json"
        )
    )

    assert (
        paths.report_path
        == (
            expected_run_directory
            / "final_report.md"
        )
    )

    assert (
        paths.model_directory
        == expected_run_directory / "models"
    )

    assert paths.run_directory.is_dir()
    assert paths.model_directory.is_dir()

    # Files are created later by their responsible tools.
    assert paths.state_path.exists() is False
    assert paths.report_path.exists() is False


def test_separate_runs_receive_different_directories(
    tmp_path: Path,
) -> None:
    """
    Two runs created at the same timestamp must still remain isolated.
    """

    created_at = datetime(
        2026,
        7,
        26,
        8,
        0,
        0,
        tzinfo=timezone.utc,
    )

    first_paths = create_run_paths(
        output_root=tmp_path,
        created_at=created_at,
        unique_suffix="first-run",
    )

    second_paths = create_run_paths(
        output_root=tmp_path,
        created_at=created_at,
        unique_suffix="second-run",
    )

    assert (
        first_paths.run_id
        != second_paths.run_id
    )

    assert (
        first_paths.run_directory
        != second_paths.run_directory
    )

    assert (
        first_paths.state_path
        != second_paths.state_path
    )

    assert (
        first_paths.report_path
        != second_paths.report_path
    )

    assert (
        first_paths.model_directory
        != second_paths.model_directory
    )

    assert first_paths.run_directory.exists()
    assert second_paths.run_directory.exists()


def test_existing_run_directory_is_never_overwritten(
    tmp_path: Path,
) -> None:
    """
    Reusing an existing run identifier must fail rather than overwrite
    artifacts from the earlier experiment.
    """

    created_at = datetime(
        2026,
        7,
        26,
        9,
        30,
        0,
        tzinfo=timezone.utc,
    )

    first_paths = create_run_paths(
        output_root=tmp_path,
        created_at=created_at,
        unique_suffix="collision-test",
    )

    marker_path = (
        first_paths.run_directory
        / "existing_artifact.txt"
    )

    marker_path.write_text(
        "This belongs to the first run.",
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError,
    ):
        create_run_paths(
            output_root=tmp_path,
            created_at=created_at,
            unique_suffix="collision-test",
        )

    assert (
        marker_path.read_text(
            encoding="utf-8",
        )
        == "This belongs to the first run."
    )


def test_run_paths_convert_to_json_safe_dictionary(
    tmp_path: Path,
) -> None:
    """
    Run path metadata must be serializable into ExperimentState JSON.
    """

    created_at = datetime(
        2026,
        7,
        26,
        10,
        45,
        0,
        tzinfo=timezone.utc,
    )

    paths = create_run_paths(
        output_root=tmp_path,
        created_at=created_at,
        unique_suffix="json-test",
    )

    converted = paths.to_dict()

    assert converted == {
        "run_id": paths.run_id,
        "run_directory": str(
            paths.run_directory
        ),
        "state_path": str(
            paths.state_path
        ),
        "report_path": str(
            paths.report_path
        ),
        "model_directory": str(
            paths.model_directory
        ),
    }

    assert all(
        isinstance(value, str)
        for value in converted.values()
    )


def test_run_suffix_requires_letter_or_number(
    tmp_path: Path,
) -> None:
    """
    A suffix containing only unsafe punctuation must be rejected clearly.
    """

    with pytest.raises(
        ValueError,
        match=(
            "Run suffix must contain at least "
            "one letter or number"
        ),
    ):
        create_run_paths(
            output_root=tmp_path,
            unique_suffix="---___...",
        )