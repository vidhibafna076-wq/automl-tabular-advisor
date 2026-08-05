"""Isolated, cancellable experiment execution and session-scoped storage."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import multiprocessing
import os
import pickle
import re
import secrets
import sys
import time
import traceback

import streamlit as st

from .helpers import PROJECT_ROOT, safe_filename

from src.run_context import (
    RunPaths,
    apply_run_paths_to_state,
    create_run_paths,
)


SESSION_PATTERN = re.compile(r"^[a-f0-9]{24}$")
_PROCESS_REGISTRY: dict[str, multiprocessing.Process] = {}


def get_session_id() -> str:
    """Keep a non-identifying session token in the URL for run isolation."""

    try:
        candidate = st.query_params.get("session")
        if isinstance(candidate, list):
            candidate = candidate[0] if candidate else None
    except Exception:
        candidate = None

    if not isinstance(candidate, str) or not SESSION_PATTERN.fullmatch(candidate):
        candidate = secrets.token_hex(12)
        try:
            st.query_params["session"] = candidate
        except Exception:
            pass
    return candidate


def session_root(session_id: str) -> Path:
    root = PROJECT_ROOT / "outputs" / "sessions" / session_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def _apply_payload_run_paths(
    state: Any,
    payload: dict[str, Any],
) -> Any:
    """
    Attach the current UI run's artifact paths to an experiment state.

    This deliberately overwrites older run metadata when a paused experiment
    is resumed in a new run directory.
    """

    run_paths = RunPaths(
        run_id=str(payload["run_id"]),
        run_directory=Path(payload["run_dir"]).resolve(),
        state_path=Path(payload["state_path"]).resolve(),
        report_path=Path(payload["report_path"]).resolve(),
        model_directory=Path(
            payload["model_directory"]
        ).resolve(),
    )

    return apply_run_paths_to_state(
        state=state,
        run_paths=run_paths,
    )


def _atomic_pickle(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle)
    temporary.replace(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_event(path: Path, event: dict[str, Any]) -> None:
    event_with_time = {
        **event,
        "ui_recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event_with_time, default=str) + "\n")


def _approval_columns(approval: Any) -> list[str]:
    if isinstance(approval, str):
        return []
    if not isinstance(approval, dict):
        return []

    columns: list[str] = []
    for key in ("column", "column_name", "feature"):
        value = approval.get(key)
        if isinstance(value, str) and value:
            columns.append(value)
    for key in ("columns", "column_names", "features"):
        value = approval.get(key)
        if isinstance(value, list):
            columns.extend(str(item) for item in value if item)
    return list(dict.fromkeys(columns))


def _apply_approval_decisions(
    state: Any,
    decisions: list[dict[str, Any]],
) -> Any:
    """Apply explicit human decisions before asking the orchestrator to resume."""

    config = dict(getattr(state, "preprocessing_config", {}) or {})
    pending = list(config.get("pending_approvals", []) or [])
    columns_to_drop = list(config.get("columns_to_drop", []) or [])
    profile = dict(getattr(state, "profile", {}) or {})
    likely_identifiers = list(profile.get("possible_id_columns", []) or [])

    audit_rows: list[dict[str, Any]] = []
    for index, approval in enumerate(pending):
        decision = decisions[index] if index < len(decisions) else {}
        action = decision.get("action")
        columns = _approval_columns(approval)
        if not columns and isinstance(approval, str):
            columns = likely_identifiers

        if action == "approve":
            for column in columns:
                if column not in columns_to_drop:
                    columns_to_drop.append(column)
        elif action == "reject":
            columns_to_drop = [
                column for column in columns_to_drop if column not in set(columns)
            ]
        else:
            raise ValueError("Every pending preprocessing decision needs a response.")

        audit_rows.append(
            {
                "approval_index": index,
                "action": action,
                "columns": columns,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    config["columns_to_drop"] = columns_to_drop
    config["pending_approvals"] = []
    config["human_approval_decisions"] = audit_rows
    state.preprocessing_config = config
    state.status = "preprocessing_config_created"

    completed = list(getattr(state, "completed_steps", []) or [])
    if not any(
        marker in completed
        for marker in ("created_preprocessing_config", "preprocessing_config_created")
    ):
        completed.append("created_preprocessing_config")
    state.completed_steps = completed

    history = list(getattr(state, "tool_history", []) or [])
    history.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_name": "human_approval",
            "status": "completed",
            "message": f"Recorded {len(audit_rows)} preprocessing decision(s).",
        }
    )
    state.tool_history = history
    return state


def _worker(payload_path: str) -> None:
    """Child-process entry point. Raw exceptions stay in the run log."""

    payload_file = Path(payload_path).resolve()
    with payload_file.open("rb") as handle:
        payload = pickle.load(handle)

    run_dir = Path(payload["run_dir"]).resolve()
    result_path = run_dir / "result.pkl"
    error_path = run_dir / "error.json"
    log_path = run_dir / "errors" / "worker.log"
    events_path = run_dir / "events.jsonl"
    run_dir.joinpath("errors").mkdir(parents=True, exist_ok=True)

    try:
        project_root = str(payload["project_root"])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from src.agents.orchestrator_agent import run_orchestrator
        from src.state import ExperimentState, save_state

        os.chdir(run_dir)
        started = time.monotonic()

        def progress_callback(event: dict[str, Any]) -> None:
            _append_event(
                events_path,
                {
                    **event,
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                },
            )

        initial_state = payload.get("initial_state")
        if initial_state is None:
            state = ExperimentState(
                dataset_path=str(Path(payload["dataset_path"]).resolve()),
                target_column=payload["target_column"],
                user_objective=payload["user_objective"],
            )
            task_override = payload.get("task_override")
            if task_override:
                try:
                    state.task_type = task_override
                except Exception:
                    progress_callback(
                        {
                            "action": "task_type_override",
                            "phase": "warning",
                            "message": "The backend state did not accept a task override.",
                        }
                    )
            if "created_experiment_state" not in state.completed_steps:
                state.completed_steps.append("created_experiment_state")
        else:
            state = _apply_approval_decisions(
                initial_state,
                payload.get("approval_decisions", []),
            )

        _apply_payload_run_paths(
            state=state,
            payload=payload,
        )

        orchestrator_result = run_orchestrator(
            state=state,
            approve_drop_id_columns=payload["approve_drop_id_columns"],
            progress_callback=progress_callback,
        )

        state = orchestrator_result["state"]

        if not state.state_path:
            raise RuntimeError(
                "The experiment state path was not configured."
            )

        save_state(
            state=state,
            output_path=state.state_path,
        )

        elapsed = round(time.monotonic() - started, 2)
        orchestrator_result["ui_run_id"] = payload["run_id"]
        orchestrator_result["ui_run_dir"] = str(run_dir)
        orchestrator_result["ui_elapsed_seconds"] = elapsed
        orchestrator_result["ui_state_path"] = state.state_path
        orchestrator_result["ui_resumed_from"] = payload.get("resumed_from")
        _atomic_pickle(result_path, orchestrator_result)

        critic = getattr(state, "critic_report", {}) or {}
        comparison = getattr(state, "comparison_summary", {}) or {}
        _atomic_json(
            run_dir / "summary.json",
            {
                "run_id": payload["run_id"],
                "created_at": payload["created_at"],
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": elapsed,
                "status": getattr(state, "status", "unknown"),
                "target_column": getattr(state, "target_column", None),
                "task_type": getattr(state, "task_type", None),
                "reliability": critic.get("overall_reliability"),
                "decision": critic.get("recommendation_decision"),
                "primary_metric": comparison.get("primary_metric"),
                "best_model": comparison.get("best_useful_display_name"),
                "resumed_from": payload.get("resumed_from"),
            },
        )
    except BaseException as error:
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        _atomic_json(
            error_path,
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "reference": payload.get("run_id"),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )


def start_background_run(
    *,
    root: Path,
    dataset_bytes: bytes | None,
    dataset_name: str | None,
    target_column: str | None,
    user_objective: str | None,
    task_override: str | None,
    approve_drop_id_columns: bool,
    initial_state: Any | None = None,
    approval_decisions: list[dict[str, Any]] | None = None,
    resumed_from: str | None = None,
    max_runtime_seconds: int | None = None,
) -> dict[str, Any]:
    """Create an isolated run and start the orchestrator in a child process."""

    run_paths = create_run_paths(
        output_root=root / "runs",
    )

    run_id = run_paths.run_id
    run_dir = run_paths.run_directory

    input_dir = run_dir / "input"
    input_dir.mkdir(exist_ok=False)

    run_dir.joinpath("errors").mkdir(
        exist_ok=False,
    )

    if initial_state is None:
        if dataset_bytes is None or not dataset_name:
            raise ValueError("Dataset bytes are required for a new run.")
        dataset_path = input_dir / safe_filename(dataset_name)
        dataset_path.write_bytes(dataset_bytes)
    else:
        source = Path(str(getattr(initial_state, "dataset_path", ""))).resolve()
        if not source.exists():
            raise FileNotFoundError("The source dataset for this paused run is unavailable.")
        dataset_path = input_dir / safe_filename(source.name)
        dataset_path.write_bytes(source.read_bytes())
        initial_state = pickle.loads(pickle.dumps(initial_state))
        initial_state.dataset_path = str(dataset_path.resolve())

    payload = {
        "run_id": run_id,
        "run_dir": str(
            run_paths.run_directory.resolve()
        ),
        "state_path": str(
            run_paths.state_path.resolve()
        ),
        "report_path": str(
            run_paths.report_path.resolve()
        ),
        "model_directory": str(
            run_paths.model_directory.resolve()
        ),
        "project_root": str(PROJECT_ROOT),
        "dataset_path": str(dataset_path.resolve()),
        "target_column": target_column,
        "user_objective": user_objective,
        "task_override": task_override,
        "approve_drop_id_columns": approve_drop_id_columns,
        "initial_state": initial_state,
        "approval_decisions": approval_decisions or [],
        "resumed_from": resumed_from,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload_path = run_dir / "payload.pkl"
    _atomic_pickle(payload_path, payload)

    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_worker,
        args=(str(payload_path),),
        name=f"automl-{run_id}",
        daemon=False,
    )
    process.start()
    _PROCESS_REGISTRY[run_id] = process

    handle = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "started_at": time.time(),
        "pid": process.pid,
        "max_runtime_seconds": max_runtime_seconds,
    }
    st.session_state["active_run"] = handle
    return handle


def read_events(run_dir: Path) -> list[dict[str, Any]]:
    events_file = run_dir / "events.jsonl"
    if not events_file.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        for line in events_file.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if isinstance(value, dict):
                events.append(value)
    except (OSError, ValueError):
        return events
    return events


def load_run_result(run_dir: Path, allowed_root: Path) -> dict[str, Any] | None:
    """Load only result files created beneath the current session root."""

    resolved_run = run_dir.resolve()
    resolved_root = allowed_root.resolve()
    if resolved_root not in resolved_run.parents:
        return None
    result_path = resolved_run / "result.pkl"
    if not result_path.exists():
        return None
    try:
        with result_path.open("rb") as handle:
            result = pickle.load(handle)
        return result if isinstance(result, dict) else None
    except (OSError, pickle.PickleError, EOFError):
        return None


def poll_active_run(handle: dict[str, Any], root: Path) -> dict[str, Any]:
    run_dir = Path(handle["run_dir"])
    result = load_run_result(run_dir, root)
    if result is not None:
        _PROCESS_REGISTRY.pop(str(handle.get("run_id")), None)
        return {"status": "completed", "result": result}

    error_file = run_dir / "error.json"
    if error_file.exists():
        _PROCESS_REGISTRY.pop(str(handle.get("run_id")), None)
        try:
            error = json.loads(error_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            error = {}
        return {"status": "failed", "error": error}

    cancelled_file = run_dir / "cancelled.json"
    if cancelled_file.exists():
        _PROCESS_REGISTRY.pop(str(handle.get("run_id")), None)
        return {"status": "cancelled"}

    process = _PROCESS_REGISTRY.get(str(handle.get("run_id")))
    if process is not None and not process.is_alive():
        _PROCESS_REGISTRY.pop(str(handle.get("run_id")), None)
        return {
            "status": "failed",
            "error": {"reference": handle.get("run_id")},
        }

    return {
        "status": "running",
        "events": read_events(run_dir),
        "elapsed_seconds": round(time.time() - handle["started_at"], 1),
    }


def cancel_active_run(handle: dict[str, Any]) -> bool:
    """Terminate the exact Process object associated with this UI session."""

    run_id = str(handle.get("run_id"))
    process = _PROCESS_REGISTRY.get(run_id)
    if process is None or not process.is_alive():
        return False

    process.terminate()
    process.join(timeout=5)
    _PROCESS_REGISTRY.pop(run_id, None)
    run_dir = Path(handle["run_dir"])
    _atomic_json(
        run_dir / "cancelled.json",
        {
            "status": "cancelled",
            "cancelled_at": datetime.now(timezone.utc).isoformat(),
            "run_id": handle.get("run_id"),
        },
    )
    return True


def completed_runs(root: Path) -> list[Path]:
    runs_dir = root / "runs"
    if not runs_dir.exists():
        return []
    return sorted(
        (
            path
            for path in runs_dir.iterdir()
            if path.is_dir() and (path / "result.pkl").exists()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )