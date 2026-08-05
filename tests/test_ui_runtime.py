"""Regression tests for Streamlit run-context integration."""

from pathlib import Path
import pickle
from typing import Any

from src.state import ExperimentState, state_to_dict
from src.ui import runtime as runtime_module
from src.ui.runtime import _apply_payload_run_paths


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


def test_start_background_run_creates_official_worker_payload(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """
    Starting a UI experiment must send one internally consistent RunPaths
    context to the worker without actually launching model training.
    """

    captured: dict[str, Any] = {}
    real_create_run_paths = runtime_module.create_run_paths

    def fake_create_run_paths(
        output_root: str | Path = "outputs/runs",
    ):
        captured["output_root"] = Path(output_root)
        run_paths = real_create_run_paths(
            output_root=output_root,
            unique_suffix="ui-runtime-test",
        )
        captured["run_paths"] = run_paths
        return run_paths

    class FakeProcess:
        pid = 4242

        def __init__(
            self,
            *,
            target: Any,
            args: tuple[str, ...],
            name: str,
            daemon: bool,
        ) -> None:
            captured["process_target"] = target
            captured["process_args"] = args
            captured["process_name"] = name
            captured["process_daemon"] = daemon

        def start(self) -> None:
            captured["process_started"] = True

    class FakeContext:
        def Process(self, **kwargs: Any) -> FakeProcess:
            return FakeProcess(**kwargs)

    def fake_get_context(method: str) -> FakeContext:
        captured["process_method"] = method
        return FakeContext()

    session_state: dict[str, Any] = {}

    monkeypatch.setattr(
        runtime_module,
        "create_run_paths",
        fake_create_run_paths,
    )
    monkeypatch.setattr(
        runtime_module.multiprocessing,
        "get_context",
        fake_get_context,
    )
    monkeypatch.setattr(
        runtime_module.st,
        "session_state",
        session_state,
    )
    monkeypatch.setattr(
        runtime_module,
        "_PROCESS_REGISTRY",
        {},
    )

    root = tmp_path / "session-root"
    dataset_bytes = b"Feature,Target\n1,Yes\n2,No\n"

    handle = runtime_module.start_background_run(
        root=root,
        dataset_bytes=dataset_bytes,
        dataset_name="sample.csv",
        target_column="Target",
        user_objective="Predict the target",
        task_override=None,
        approve_drop_id_columns=True,
        max_runtime_seconds=300,
    )

    run_paths = captured["run_paths"]
    payload_path = run_paths.run_directory / "payload.pkl"

    with payload_path.open("rb") as handle_file:
        payload = pickle.load(handle_file)

    expected_dataset_path = (
        run_paths.run_directory
        / "input"
        / "sample.csv"
    ).resolve()

    assert captured["output_root"] == root / "runs"
    assert captured["process_method"] == "spawn"
    assert captured["process_target"] is runtime_module._worker
    assert captured["process_args"] == (str(payload_path),)
    assert captured["process_name"] == f"automl-{run_paths.run_id}"
    assert captured["process_daemon"] is False
    assert captured["process_started"] is True

    assert payload["run_id"] == run_paths.run_id
    assert payload["run_dir"] == str(
        run_paths.run_directory.resolve()
    )
    assert payload["state_path"] == str(
        run_paths.state_path.resolve()
    )
    assert payload["report_path"] == str(
        run_paths.report_path.resolve()
    )
    assert payload["model_directory"] == str(
        run_paths.model_directory.resolve()
    )
    assert Path(payload["dataset_path"]) == expected_dataset_path
    assert expected_dataset_path.read_bytes() == dataset_bytes
    assert payload["target_column"] == "Target"
    assert payload["user_objective"] == "Predict the target"
    assert payload["approve_drop_id_columns"] is True
    assert payload["initial_state"] is None
    assert payload["approval_decisions"] == []
    assert payload["resumed_from"] is None

    assert Path(payload["state_path"]).parent == Path(payload["run_dir"])
    assert Path(payload["report_path"]).parent == Path(payload["run_dir"])
    assert Path(payload["model_directory"]).parent == Path(payload["run_dir"])
    assert "reports" not in Path(payload["state_path"]).parts

    assert handle == {
        "run_id": run_paths.run_id,
        "run_dir": str(run_paths.run_directory),
        "started_at": handle["started_at"],
        "pid": 4242,
        "max_runtime_seconds": 300,
    }
    assert session_state["active_run"] == handle
    assert runtime_module._PROCESS_REGISTRY[run_paths.run_id].pid == 4242