"""Top-level Streamlit application and experiment setup workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .components import (
    inject_css,
    render_dataset_strip,
    render_header,
    render_hero,
    render_section_title,
    render_workflow,
    safe_container_with_border,
)
from .constants import (
    ACTION_LABELS,
    ACTION_STAGE,
    APP_TITLE,
    MAX_UPLOAD_BYTES,
    PREVIEW_ROWS,
    WORKFLOW_STEPS,
)
from .helpers import (
    PROJECT_ROOT,
    friendly_failure,
    humanise,
    load_json,
    parse_csv_preview,
)
from .prediction import show_prediction_workbench
from .results import show_results_dashboard
from .runtime import (
    cancel_active_run,
    completed_runs,
    get_session_id,
    load_run_result,
    poll_active_run,
    session_root,
    start_background_run,
)


TRAIN_WORKSPACE = "Train and evaluate"
PREDICT_WORKSPACE = "Predict with a saved model"


def _readiness() -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not (PROJECT_ROOT / "src" / "agents" / "orchestrator_agent.py").exists():
        issues.append("Orchestrator module not found")
    if not (PROJECT_ROOT / "src" / "state.py").exists():
        issues.append("Experiment state module not found")
    if not (PROJECT_ROOT / "src" / "inference.py").exists():
        issues.append("Inference module not found")
    return not issues, issues


def _history_label(run_dir: Path) -> str:
    summary = load_json(run_dir / "summary.json")
    model = summary.get("best_model") or "No selected model"
    status = humanise(summary.get("status"), "Unknown")
    return f"{run_dir.name} · {model} · {status}"


def _open_result(
    result: dict[str, Any],
    *,
    switch_to_training: bool = False,
) -> None:
    state = result.get("state")
    if state is None:
        st.session_state["ui_notice"] = (
            "warning",
            "That run does not contain a readable experiment state.",
        )
        return
    st.session_state["last_state"] = state
    st.session_state["last_orchestrator_result"] = result
    st.session_state["current_run_dir"] = result.get("ui_run_dir")
    if switch_to_training:
        st.session_state["workspace_mode"] = TRAIN_WORKSPACE


def show_sidebar_help(root: Path, readiness_issues: list[str]) -> None:
    with st.sidebar:
        st.markdown(f"## {APP_TITLE}")
        st.caption(
            "A controlled tabular modelling workflow with planning, evaluation, "
            "human approval, reliability review, and saved-model inference."
        )
        st.markdown("### What this evaluates")
        st.markdown(
            "- Classification and regression\n"
            "- Dummy-baseline usefulness\n"
            "- Cross-validation stability\n"
            "- Tuning evidence\n"
            "- Final holdout evidence\n"
            "- Critic limitations"
        )
        st.warning(
            "Local demo privacy: do not upload secrets, health records, identity "
            "documents, or production customer data. Run artifacts remain on the "
            "machine until you remove them."
        )
        if readiness_issues:
            st.error("Setup check: " + "; ".join(readiness_issues))

        history = completed_runs(root)
        st.markdown("### This session's runs")
        if not history:
            st.caption("No completed runs in this session.")
        else:
            selected = st.selectbox(
                "Run history",
                options=history[:20],
                format_func=_history_label,
                key="history_selector",
                label_visibility="collapsed",
            )
            if st.button(
                "Open selected run",
                width="stretch",
                key="open_history_run",
            ):
                result = load_run_result(selected, root)
                if result:
                    _open_result(result, switch_to_training=True)
                    st.rerun()
                else:
                    st.error("That run could not be reopened.")


def _task_override_options(
    dataframe: pd.DataFrame,
    target_column: str | None,
) -> list[str]:
    options = [
        "Auto-detect",
        "Binary classification",
        "Multiclass classification",
        "Regression",
    ]
    if not target_column or target_column not in dataframe:
        return options
    unique = dataframe[target_column].nunique(dropna=True)
    if unique == 2:
        return options
    if unique <= 20:
        return [
            "Auto-detect",
            "Multiclass classification",
            "Binary classification",
            "Regression",
        ]
    return [
        "Auto-detect",
        "Regression",
        "Multiclass classification",
        "Binary classification",
    ]


def _task_override_value(label: str) -> str | None:
    return {
        "Auto-detect": None,
        "Binary classification": "binary_classification",
        "Multiclass classification": "multiclass_classification",
        "Regression": "regression",
    }[label]


def _show_target_distribution(
    dataframe: pd.DataFrame,
    target_column: str,
    truncated: bool,
) -> None:
    target = dataframe[target_column].dropna()
    if target.empty:
        st.warning("The target is empty in the preview.")
        return

    unique = target.nunique()
    st.markdown("### Target distribution")
    if unique <= 20:
        counts = target.astype(str).value_counts(dropna=False).rename("Rows")
        st.bar_chart(counts)
    elif pd.api.types.is_numeric_dtype(target):
        bins = min(20, max(5, int(unique**0.5)))
        distribution = (
            pd.cut(target, bins=bins, duplicates="drop")
            .value_counts()
            .sort_index()
            .rename("Rows")
        )
        distribution.index = distribution.index.astype(str)
        st.bar_chart(distribution)
    else:
        st.info(
            f"The target has {unique:,} distinct values in the preview. "
            "This may be an identifier or an unsuitable prediction target."
        )
    if truncated:
        st.caption(
            f"Distribution uses the first {PREVIEW_ROWS:,} rows, not the full dataset."
        )


def _workload_message(
    dataframe: pd.DataFrame,
    file_size: int,
    truncated: bool,
) -> str:
    preview_cells = dataframe.shape[0] * dataframe.shape[1]
    if file_size > 75 * 1024 * 1024 or preview_cells > 2_000_000:
        size = "large"
    elif file_size > 20 * 1024 * 1024 or preview_cells > 300_000:
        size = "medium"
    else:
        size = "small"
    scope = "at least " if truncated else ""
    return (
        f"Estimated workload: **{size}** ({scope}{len(dataframe):,} rows visible, "
        f"{len(dataframe.columns):,} columns). Runtime depends on model families and "
        "cross-validation; the run can be terminated from the progress panel."
    )


def show_setup_workbench(root: Path) -> None:
    render_hero(
        "New experiment",
        "Build and evaluate a candidate model.",
        "12-stage evaluation",
        "Planner decisions, holdout evidence, and critic review",
    )
    st.info(
        "Upload structured CSV data only. This local application saves a run-specific "
        "copy so the experiment can be reproduced. Avoid sensitive or regulated data."
    )

    mode = st.radio(
        "Dataset source",
        ("Upload CSV", "Use sample dataset"),
        horizontal=True,
        key="dataset_source",
    )
    file_bytes: bytes | None = None
    filename: str | None = None
    preview: pd.DataFrame | None = None
    truncated = False
    preview_failed = False

    source_column, details_column = st.columns([0.55, 0.45], gap="large")
    with source_column:
        with safe_container_with_border():
            st.markdown("### Dataset")
            if mode == "Use sample dataset":
                sample_path = PROJECT_ROOT / "data" / "sample" / "loan_sample.csv"
                if not sample_path.exists():
                    st.error("The sample dataset is not available in data/sample.")
                else:
                    file_bytes = sample_path.read_bytes()
                    filename = sample_path.name
                    st.success("loan_sample.csv selected")
            else:
                upload = st.file_uploader(
                    "Upload CSV dataset",
                    type=["csv"],
                    key="dataset_upload",
                    help="Maximum file size for this application: 150 MB.",
                )
                if upload is not None:
                    file_bytes = upload.getvalue()
                    filename = upload.name

            if file_bytes is not None:
                if len(file_bytes) > MAX_UPLOAD_BYTES:
                    st.error("This file exceeds the 150 MB application limit.")
                    file_bytes = None
                else:
                    try:
                        preview, truncated = parse_csv_preview(file_bytes)
                        if preview.empty:
                            st.error("The CSV contains headers but no data rows.")
                            preview = None
                    except Exception:
                        preview_failed = True
                        st.error(friendly_failure("csv"))

            if preview is None and not preview_failed:
                st.markdown(
                    '<div class="empty-preview">Choose a CSV to inspect its structure.</div>',
                    unsafe_allow_html=True,
                )
            elif preview is not None:
                render_dataset_strip(preview, target_column=None, truncated=truncated)
                st.dataframe(preview.head(20), width="stretch", hide_index=True)
                if truncated:
                    st.caption(
                        f"Only the first {PREVIEW_ROWS:,} rows are parsed during setup. "
                        "The full file is read once by the experiment worker."
                    )

    target_column: str | None = None
    objective = ""
    task_override_label = "Auto-detect"
    approve_drop_identifiers = False
    runtime_limit_label = "20 minutes"
    run_clicked = False

    with details_column:
        with safe_container_with_border():
            st.markdown("### Experiment definition")
            if preview is not None:
                options = list(preview.columns)
                default_index = (
                    options.index("Loan_Status")
                    if mode == "Use sample dataset" and "Loan_Status" in options
                    else None
                )
                target_column = st.selectbox(
                    "Prediction target",
                    options=options,
                    index=default_index,
                    placeholder="Select the column to predict",
                )
            else:
                st.selectbox(
                    "Prediction target",
                    options=[],
                    placeholder="Choose a dataset first",
                    disabled=True,
                )

            objective_default = (
                "Predict whether a loan application will be approved"
                if mode == "Use sample dataset"
                else ""
            )
            objective = st.text_area(
                "Experiment objective",
                value=objective_default,
                placeholder="Describe the decision this model should support.",
                key=(
                    "sample_objective"
                    if mode == "Use sample dataset"
                    else "upload_objective"
                ),
                height=110,
                disabled=preview is None,
            )

            with st.expander("Modelling controls", expanded=True):
                task_options = (
                    _task_override_options(preview, target_column)
                    if preview is not None
                    else ["Auto-detect"]
                )
                task_override_label = st.selectbox(
                    "Task type",
                    options=task_options,
                    help=(
                        "Leave this on Auto-detect unless the preview makes the "
                        "intended task unambiguous."
                    ),
                )
                approve_drop_identifiers = st.checkbox(
                    "Pre-approve dropping likely identifier columns",
                    value=(mode == "Use sample dataset"),
                    key=(
                        "sample_drop_identifiers"
                        if mode == "Use sample dataset"
                        else "upload_drop_identifiers"
                    ),
                    help=(
                        "Off by default because removing a feature is consequential. "
                        "The built-in sample enables this because Loan_ID is a known "
                        "demonstration identifier. When off, the workflow may pause "
                        "for a human decision."
                    ),
                )
                runtime_limit_label = st.selectbox(
                    "Automatic run limit",
                    (
                        "1 minute",
                        "5 minutes",
                        "10 minutes",
                        "20 minutes",
                        "30 minutes",
                        "No automatic limit",
                    ),
                    index=3,
                    help=(
                        "The isolated worker is terminated if it exceeds this limit. "
                        "Completed stages remain diagnostic only."
                    ),
                )

            if preview is not None and file_bytes is not None:
                st.info(_workload_message(preview, len(file_bytes), truncated))

            run_clicked = st.button(
                "Start evaluation",
                type="primary",
                width="stretch",
                disabled=(
                    preview is None
                    or target_column is None
                    or not objective.strip()
                    or file_bytes is None
                ),
            )

    if preview is not None and target_column:
        _show_target_distribution(preview, target_column, truncated)

    render_workflow()

    if run_clicked and file_bytes is not None and filename and target_column:
        try:
            start_background_run(
                root=root,
                dataset_bytes=file_bytes,
                dataset_name=filename,
                target_column=target_column,
                user_objective=objective.strip(),
                task_override=_task_override_value(task_override_label),
                approve_drop_id_columns=approve_drop_identifiers,
                max_runtime_seconds=(
                    None
                    if runtime_limit_label == "No automatic limit"
                    else int(runtime_limit_label.split()[0]) * 60
                ),
            )
            st.session_state.pop("last_state", None)
            st.session_state.pop("last_orchestrator_result", None)
            st.rerun()
        except Exception:
            st.error(
                "The run could not be started. Check that the outputs directory is "
                "writable, then try again."
            )


def _latest_progress(events: list[dict[str, Any]]) -> tuple[int, str]:
    stage = 0
    label = "Preparing the isolated run"
    for event in events:
        action = str(event.get("action", ""))
        event_stage = ACTION_STAGE.get(action)
        if event_stage:
            stage = max(stage, event_stage)
            label = ACTION_LABELS.get(action, humanise(action))
    return stage, label


def _active_run_body(root: Path) -> None:
    handle = st.session_state.get("active_run")
    if not handle:
        return
    status = poll_active_run(handle, root)
    run_id = handle.get("run_id", "current run")

    if status["status"] == "completed":
        _open_result(status["result"])
        st.session_state.pop("active_run", None)
        st.session_state["ui_notice"] = (
            "success",
            "Evaluation completed. Review the evidence below.",
        )
        st.rerun()

    if status["status"] == "failed":
        st.session_state.pop("active_run", None)
        st.session_state["ui_notice"] = (
            "error",
            friendly_failure("experiment", run_id),
        )
        st.rerun()

    if status["status"] == "cancelled":
        st.session_state.pop("active_run", None)
        st.session_state["ui_notice"] = (
            "warning",
            f"Run {run_id} was terminated. Its incomplete result is not available for use.",
        )
        st.rerun()

    events = status.get("events", [])
    stage, label = _latest_progress(events)
    elapsed_seconds = float(status.get("elapsed_seconds", 0))
    runtime_limit = handle.get("max_runtime_seconds")
    if runtime_limit and elapsed_seconds >= float(runtime_limit):
        if cancel_active_run(handle):
            st.session_state.pop("active_run", None)
            st.session_state["ui_notice"] = (
                "warning",
                f"Run {run_id} reached its {int(runtime_limit) // 60}-minute limit and was terminated.",
            )
            st.rerun()

    render_hero(
        "Evaluation in progress",
        label,
        f"Run {run_id}",
        "The worker is isolated from other sessions",
    )
    progress = min(stage / len(WORKFLOW_STEPS), 0.96)
    st.progress(
        progress,
        text=(
            f"Stage {stage or 1} of {len(WORKFLOW_STEPS)} · "
            f"{elapsed_seconds:g} seconds elapsed"
        ),
    )
    render_section_title(
        "Live execution record",
        "The last events reported by the orchestrator. Per-model timing appears when emitted by the backend.",
    )
    if events:
        timeline = pd.DataFrame(
            [
                {
                    "Elapsed (s)": event.get("elapsed_seconds"),
                    "Action": ACTION_LABELS.get(
                        str(event.get("action", "")),
                        humanise(event.get("action")),
                    ),
                    "Phase": humanise(event.get("phase")),
                    "Message": event.get("message", ""),
                }
                for event in events[-8:]
            ]
        )
        st.dataframe(timeline, width="stretch", hide_index=True)
    else:
        st.info("The worker has started; waiting for the first progress event.")

    st.warning(
        "Terminating a run stops the child process. Partial run files remain for "
        "diagnosis but are never listed as completed models."
    )
    if st.button(
        "Terminate this run",
        type="secondary",
        key=f"cancel_{run_id}",
    ):
        if cancel_active_run(handle):
            st.session_state.pop("active_run", None)
            st.session_state["ui_notice"] = (
                "warning",
                f"Run {run_id} was terminated.",
            )
            st.rerun()
        else:
            st.warning(
                "The worker is no longer attached to this browser session. "
                "Refresh to check whether it has completed."
            )


def render_active_run(root: Path) -> None:
    fragment = getattr(st, "fragment", None)
    if fragment is not None:
        fragment(run_every=1.0)(_active_run_body)(root)
    else:
        _active_run_body(root)
        st.button("Refresh run status", key="refresh_run_status")


def _clear_current_result() -> None:
    for key in (
        "last_state",
        "last_orchestrator_result",
        "current_run_dir",
        "dataset_upload",
        "sample_objective",
        "upload_objective",
    ):
        st.session_state.pop(key, None)


def _display_notice() -> None:
    notice = st.session_state.pop("ui_notice", None)
    if not notice:
        return
    tone, message = notice
    if tone == "success":
        st.success(message)
    elif tone == "error":
        st.error(message)
    else:
        st.warning(message)


def main() -> None:
    inject_css()
    session_id = get_session_id()
    root = session_root(session_id)
    ready, readiness_issues = _readiness()
    show_sidebar_help(root, readiness_issues)
    render_header(ready)
    _display_notice()

    workspace = st.radio(
        "Workspace",
        (TRAIN_WORKSPACE, PREDICT_WORKSPACE),
        horizontal=True,
        key="workspace_mode",
        label_visibility="collapsed",
    )

    if st.session_state.get("active_run"):
        render_active_run(root)
        return

    if workspace == PREDICT_WORKSPACE:
        show_prediction_workbench(root)
        return

    state = st.session_state.get("last_state")
    orchestrator_result = st.session_state.get("last_orchestrator_result")
    if state is None or orchestrator_result is None:
        show_setup_workbench(root)
        return

    _, action_column = st.columns([0.76, 0.24])
    with action_column:
        if st.button(
            "Start a new experiment",
            width="stretch",
            key="new_experiment",
        ):
            _clear_current_result()
            st.rerun()

    def resume(decisions: list[dict[str, Any]]) -> None:
        try:
            start_background_run(
                root=root,
                dataset_bytes=None,
                dataset_name=None,
                target_column=None,
                user_objective=None,
                task_override=None,
                approve_drop_id_columns=True,
                initial_state=state,
                approval_decisions=decisions,
                resumed_from=str(orchestrator_result.get("ui_run_id", "")),
                max_runtime_seconds=20 * 60,
            )
            st.session_state.pop("last_state", None)
            st.session_state.pop("last_orchestrator_result", None)
            st.rerun()
        except Exception:
            st.error(
                "The paused run could not be resumed. Its source dataset may no "
                "longer be available; reopen the run or start a revised experiment."
            )

    def open_prediction() -> None:
        st.session_state["workspace_mode"] = PREDICT_WORKSPACE

    show_results_dashboard(
        state,
        orchestrator_result,
        on_resume=resume,
        on_predict=open_prediction,
    )
