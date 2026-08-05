from pathlib import Path

from src.state import (
    ExperimentState,
    state_to_dict,
)
from src.ui.runtime import (
    _apply_payload_run_paths,
)


def test_payload_run_paths_are_attached_to_ui_state(
    tmp_path: Path,
) -> None:
    run_directory = (
        tmp_path
        / "sessions"
        / "session-001"
        / "runs"
        / "run-001"
    )

    payload = {
        "run_id": "run-001",
        "run_dir": str(run_directory),
        "state_path": str(
            run_directory / "experiment_state.json"
        ),
        "report_path": str(
            run_directory / "final_report.md"
        ),
        "model_directory": str(
            run_directory / "models"
        ),
    }

    state = ExperimentState(
        dataset_path="input/dataset.csv",
        target_column="Target",
        user_objective="Test UI run isolation",
    )

    returned_state = _apply_payload_run_paths(
        state=state,
        payload=payload,
    )

    assert returned_state is state
    assert state.run_id == "run-001"
    assert state.run_directory == str(
        run_directory.resolve()
    )
    assert state.state_path == str(
        (
            run_directory
            / "experiment_state.json"
        ).resolve()
    )
    assert state.report_path == str(
        (
            run_directory
            / "final_report.md"
        ).resolve()
    )
    assert state.model_directory == str(
        (
            run_directory
            / "models"
        ).resolve()
    )

    serialised_state = state_to_dict(state)

    assert all(
        isinstance(
            serialised_state[field_name],
            str,
        )
        for field_name in [
            "run_id",
            "run_directory",
            "state_path",
            "report_path",
            "model_directory",
        ]
    )

def test_resumed_ui_run_replaces_original_run_paths(
    tmp_path: Path,
) -> None:
    state = ExperimentState(
        dataset_path="input/dataset.csv",
        target_column="Target",
    )

    state.run_id = "original-run"
    state.run_directory = "old/run"
    state.state_path = "old/run/experiment_state.json"
    state.report_path = "old/run/final_report.md"
    state.model_directory = "old/run/models"

    resumed_directory = tmp_path / "resumed-run"

    payload = {
        "run_id": "resumed-run",
        "run_dir": str(resumed_directory),
        "state_path": str(
            resumed_directory / "experiment_state.json"
        ),
        "report_path": str(
            resumed_directory / "final_report.md"
        ),
        "model_directory": str(
            resumed_directory / "models"
        ),
    }

    _apply_payload_run_paths(
        state=state,
        payload=payload,
    )

    assert state.run_id == "resumed-run"
    assert state.run_directory == str(
        resumed_directory.resolve()
    )
    assert state.state_path == str(
        (
            resumed_directory
            / "experiment_state.json"
        ).resolve()
    )
    assert state.report_path == str(
        (
            resumed_directory
            / "final_report.md"
        ).resolve()
    )
    assert state.model_directory == str(
        (
            resumed_directory
            / "models"
        ).resolve()
    )