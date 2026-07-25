"""Saved-model discovery, schema review, and stale-safe batch inference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import inspect
import shutil
import traceback

import pandas as pd
import streamlit as st

from .components import (
    display_tags,
    render_dataset_strip,
    render_hero,
    render_section_title,
    safe_container_with_border,
)
from .constants import MAX_UPLOAD_BYTES
from .helpers import (
    PROJECT_ROOT,
    display_value,
    expected_feature_columns,
    fingerprint_bytes,
    first_mapping,
    first_value,
    friendly_failure,
    holdout_evidence,
    humanise,
    parse_csv_preview,
    safe_filename,
    state_attr,
    tuning_evidence,
)
from .runtime import completed_runs, load_run_result


@dataclass(frozen=True)
class SavedModelRecord:
    path: Path
    run_id: str
    target_column: str
    task_type: str
    model_name: str
    decision: str
    reliability: str
    primary_metric: str
    primary_score: Any
    holdout_score: Any
    tuning_applied: bool | None
    expected_features: tuple[str, ...]
    saved_at: str
    legacy: bool = False


def _record_label(record: SavedModelRecord) -> str:
    review = {
        "recommend_candidate_model": "supported",
        "recommend_with_caution": "caution",
        "unverified_legacy": "unverified legacy",
    }.get(record.decision, humanise(record.decision).lower())
    return (
        f"{record.model_name} · target {record.target_column} · "
        f"{review} · {record.saved_at}"
    )


def discover_saved_models(
    root: Path,
    *,
    include_legacy: bool = False,
) -> list[SavedModelRecord]:
    records: list[SavedModelRecord] = []
    for run_dir in completed_runs(root):
        result = load_run_result(run_dir, root)
        if not result:
            continue
        state = result.get("state")
        if state is None:
            continue
        critic = first_mapping(state, "critic_report")
        decision = critic.get("recommendation_decision")
        if decision not in {"recommend_candidate_model", "recommend_with_caution"}:
            continue

        comparison = first_mapping(state, "comparison_summary")
        selected = critic.get("selected_candidate", {}) or {}
        model_name = first_value(
            selected,
            "display_name",
            default=first_value(
                comparison,
                "best_useful_display_name",
                "selected_model_name",
                default="Saved candidate",
            ),
        )
        metric = first_value(
            selected,
            "primary_metric",
            default=comparison.get("primary_metric", "N/A"),
        )
        score = first_value(
            selected,
            "primary_score",
            default=comparison.get("best_useful_score"),
        )
        holdout = holdout_evidence(state)
        tuning = tuning_evidence(state)
        persistence = first_mapping(
            state,
            "model_persistence",
            "persistence_summary",
            "model_artifact_summary",
        )
        persisted_value = first_value(
            persistence,
            "model_path",
            "pipeline_path",
            "artifact_path",
            "saved_model_path",
        )
        candidates: list[Path] = []
        if persisted_value:
            persisted_path = Path(str(persisted_value))
            if not persisted_path.is_absolute():
                persisted_path = run_dir / persisted_path
            if persisted_path.exists():
                candidates = [persisted_path]
        if not candidates:
            candidates = sorted(run_dir.glob("**/*_pipeline.joblib"))

        selected_id = str(
            first_value(
                selected,
                "model_id",
                default=first_value(
                    comparison,
                    "best_useful_model_id",
                    "selected_model_id",
                    default="",
                ),
            )
        )
        if len(candidates) > 1 and selected_id:
            matching = [
                path
                for path in candidates
                if selected_id.lower() in path.stem.lower()
            ]
            if matching:
                candidates = matching

        for path in candidates:
            records.append(
                SavedModelRecord(
                    path=path.resolve(),
                    run_id=str(result.get("ui_run_id", run_dir.name)),
                    target_column=str(state_attr(state, "target_column", default="N/A")),
                    task_type=str(state_attr(state, "task_type", default="N/A")),
                    model_name=str(model_name),
                    decision=str(decision),
                    reliability=str(critic.get("overall_reliability", "N/A")),
                    primary_metric=str(metric),
                    primary_score=score,
                    holdout_score=holdout["score"],
                    tuning_applied=tuning["applied"],
                    expected_features=tuple(expected_feature_columns(state)),
                    saved_at=datetime.fromtimestamp(path.stat().st_mtime).strftime(
                        "%d %b %Y %H:%M"
                    ),
                )
            )

    if include_legacy:
        legacy_dir = PROJECT_ROOT / "outputs" / "models"
        if legacy_dir.exists():
            known = {record.path for record in records}
            for path in sorted(
                legacy_dir.glob("*_pipeline.joblib"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            ):
                resolved = path.resolve()
                if resolved in known:
                    continue
                records.append(
                    SavedModelRecord(
                        path=resolved,
                        run_id="legacy",
                        target_column="Unknown",
                        task_type="Unknown",
                        model_name=path.name.replace("_pipeline.joblib", "").replace(
                            "_", " "
                        ),
                        decision="unverified_legacy",
                        reliability="Unverified",
                        primary_metric="Unknown",
                        primary_score=None,
                        holdout_score=None,
                        tuning_applied=None,
                        expected_features=(),
                        saved_at=datetime.fromtimestamp(path.stat().st_mtime).strftime(
                            "%d %b %Y %H:%M"
                        ),
                        legacy=True,
                    )
                )

    return sorted(records, key=lambda record: record.path.stat().st_mtime, reverse=True)


def _copy_prediction_output(
    result: dict[str, Any],
    prediction_dir: Path,
) -> Path:
    source_value = result.get("output_path")
    if not source_value:
        raise FileNotFoundError("Prediction output was not recorded.")
    source = Path(str(source_value))
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    if not source.exists():
        raise FileNotFoundError("Prediction output was not created.")

    destination = prediction_dir / safe_filename(
        source.name,
        fallback="predictions.csv",
    )
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination.resolve()


def _prediction_kwargs(
    model_path: Path,
    input_path: Path,
    requested_output: Path,
) -> dict[str, str]:
    """Use a run-specific output path when the backend supports that argument."""

    from src.inference import predict_csv

    kwargs = {
        "model_path": str(model_path),
        "input_path": str(input_path),
    }
    try:
        parameters = inspect.signature(predict_csv).parameters
    except (TypeError, ValueError):
        parameters = {}
    for parameter_name in ("output_path", "prediction_output_path"):
        if parameter_name in parameters:
            kwargs[parameter_name] = str(requested_output)
            break
    return kwargs


def invalidate_stale_prediction(
    current_context: str,
    session_state: Any | None = None,
) -> bool:
    """Clear a result whenever either the model or uploaded bytes change."""

    state = session_state if session_state is not None else st.session_state
    previous_context = state.get("prediction_context")
    if previous_context == current_context:
        return False
    state.pop("last_prediction_result", None)
    state["prediction_context"] = current_context
    return True


def _render_model_metadata(record: SavedModelRecord) -> None:
    columns = st.columns(4)
    columns[0].metric("Target", record.target_column)
    columns[1].metric("Task", humanise(record.task_type))
    columns[2].metric("Reliability", humanise(record.reliability))
    columns[3].metric(
        "Review",
        (
            "Supported"
            if record.decision == "recommend_candidate_model"
            else "Caution"
            if record.decision == "recommend_with_caution"
            else "Unverified"
        ),
    )

    evidence = st.columns(4)
    evidence[0].metric("Primary metric", record.primary_metric)
    evidence[1].metric("CV score", display_value(record.primary_score))
    evidence[2].metric("Holdout score", display_value(record.holdout_score))
    evidence[3].metric(
        "Tuned",
        (
            "Yes"
            if record.tuning_applied is True
            else "No"
            if record.tuning_applied is False
            else "Unknown"
        ),
    )
    st.caption(f"Run ID: {record.run_id} · saved {record.saved_at}")
    st.caption(f"Artifact file: {record.path.name}")
    st.markdown("**Expected raw features**")
    display_tags(
        list(record.expected_features),
        "No feature schema was recorded for this artifact. Backend validation will be required.",
    )


def show_prediction_workbench(root: Path) -> None:
    render_hero(
        "Saved-model inference",
        "Run predictions with a reviewed candidate.",
        "Saved pipeline selection",
        "Review metadata and schema before inference",
    )

    st.info(
        "A reliability review is evidence, not deployment approval. Verify the "
        "target, schema, holdout result, and limitations before using predictions."
    )
    include_legacy = st.checkbox(
        "Show legacy local pipelines without run metadata",
        value=False,
        help=(
            "Legacy files cannot be verified against a run-specific critic decision "
            "or expected feature schema."
        ),
    )
    models = discover_saved_models(root, include_legacy=include_legacy)
    if not models:
        st.warning(
            "No session-scoped, reviewed pipeline is available. Complete an experiment "
            "that persists a supported candidate first."
        )
        return

    selected = st.selectbox(
        "Saved model",
        options=models,
        format_func=_record_label,
        key="prediction_model_selector",
    )
    _render_model_metadata(selected)

    if selected.decision == "recommend_with_caution":
        st.warning(
            "This candidate was recommended with caution. Treat its predictions as "
            "provisional and follow the critic's next actions."
        )
    elif selected.legacy:
        st.error(
            "This legacy artifact has no verifiable run metadata. Use it only for "
            "local compatibility testing."
        )

    settings_column, preview_column = st.columns([0.4, 0.6], gap="large")
    upload = None
    delete_input = True
    with settings_column:
        with safe_container_with_border():
            st.markdown("### Prediction input")
            upload = st.file_uploader(
                "Upload prediction CSV",
                type=["csv"],
                key="prediction_csv_upload",
                help="The target column is not required. Maximum file size: 150 MB.",
            )
            delete_input = st.checkbox(
                "Delete the saved input copy after inference",
                value=True,
                key="delete_prediction_input",
            )
            st.caption(
                "The uploaded file is checked against the recorded raw feature schema "
                "before the pipeline is called."
            )

    preview: pd.DataFrame | None = None
    truncated = False
    upload_bytes: bytes | None = None
    preview_error = False
    input_fingerprint = "no-input"
    if upload is not None:
        upload_bytes = upload.getvalue()
        input_fingerprint = fingerprint_bytes(upload_bytes)
        if len(upload_bytes) > MAX_UPLOAD_BYTES:
            preview_error = True
            st.error("This file exceeds the 150 MB application limit.")
        else:
            try:
                preview, truncated = parse_csv_preview(upload_bytes)
            except Exception:
                preview_error = True

    model_stat = selected.path.stat()
    current_context = (
        f"{selected.path}:{model_stat.st_size}:{model_stat.st_mtime_ns}:"
        f"{input_fingerprint}"
    )
    invalidate_stale_prediction(current_context)

    missing: list[str] = []
    extra: list[str] = []
    if preview is not None and selected.expected_features:
        input_columns = set(map(str, preview.columns))
        expected = set(selected.expected_features)
        missing = sorted(expected - input_columns)
        extra = sorted(input_columns - expected - {selected.target_column})

    with preview_column:
        with safe_container_with_border():
            st.markdown("### Input preview")
            if preview_error:
                st.error(friendly_failure("csv"))
            elif preview is None:
                st.markdown(
                    '<div class="empty-preview">Upload a CSV to inspect its rows and schema.</div>',
                    unsafe_allow_html=True,
                )
            else:
                render_dataset_strip(preview, target_column=None, truncated=truncated)
                st.dataframe(preview.head(20), width="stretch", hide_index=True)
                if truncated:
                    st.caption(
                        "Setup statistics use the first 5,000 rows; the full file is "
                        "passed to inference after validation."
                    )

    if missing:
        st.error("Missing required features: " + ", ".join(missing))
    if extra:
        st.warning(
            "Extra columns will be retained in the output but ignored by the model: "
            + ", ".join(extra)
        )

    generate = st.button(
        "Generate predictions",
        type="primary",
        width="stretch",
        disabled=(
            upload_bytes is None
            or preview_error
            or bool(missing)
            or not selected.path.exists()
        ),
        key="generate_saved_model_predictions",
    )

    if generate and upload_bytes is not None:
        prediction_id = (
            datetime.now().strftime("prediction_%Y%m%d_%H%M%S_")
            + input_fingerprint[:8]
        )
        prediction_dir = root / "predictions" / prediction_id
        prediction_dir.mkdir(parents=True, exist_ok=False)
        input_path = prediction_dir / safe_filename(upload.name)
        input_path.write_bytes(upload_bytes)
        requested_output = prediction_dir / "predictions.csv"

        status = st.status(
            "Validating the schema and loading the saved pipeline...",
            expanded=True,
        )
        try:
            from src.inference import predict_csv

            result = predict_csv(
                **_prediction_kwargs(selected.path, input_path, requested_output)
            )
            output_path = _copy_prediction_output(result, prediction_dir)
            result = {
                **result,
                "_ui_context": current_context,
                "_ui_output_path": str(output_path),
                "_ui_input_fingerprint": input_fingerprint,
                "_ui_run_id": selected.run_id,
                "_ui_review_decision": selected.decision,
                "_ui_expected_features": list(selected.expected_features),
            }
            st.session_state["last_prediction_result"] = result
            status.update(
                label="Predictions generated",
                state="complete",
                expanded=False,
            )
        except Exception:
            (prediction_dir / "prediction_error.log").write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )
            st.session_state.pop("last_prediction_result", None)
            status.update(
                label="Prediction could not be completed",
                state="error",
                expanded=True,
            )
            st.error(friendly_failure("prediction", prediction_id))
        finally:
            if delete_input and input_path.exists():
                input_path.unlink()

    result = st.session_state.get("last_prediction_result")
    if not result or result.get("_ui_context") != current_context:
        return

    output_path = Path(str(result.get("_ui_output_path", "")))
    if not output_path.exists():
        st.error("The session-specific prediction output is no longer available.")
        return

    try:
        output_dataframe = pd.read_csv(output_path)
    except Exception:
        st.error("The prediction output was created but could not be previewed.")
        return

    render_section_title(
        "Prediction result",
        "Confirm the model and row count before downloading this session-specific output.",
    )
    summary = st.columns(4)
    summary[0].metric("Rows predicted", result.get("rows_predicted", len(output_dataframe)))
    summary[1].metric("Model", result.get("display_name", selected.model_name))
    summary[2].metric("Task", humanise(result.get("task_type", selected.task_type)))
    summary[3].metric(
        "Tuned model",
        "Yes" if result.get("tuning_applied") else "No",
    )

    ignored = result.get("ignored_input_columns", []) or []
    if ignored:
        st.warning("Ignored input columns: " + ", ".join(map(str, ignored)))
    if result.get("probability_columns"):
        st.info("Class probabilities and prediction confidence are included.")

    st.dataframe(output_dataframe, width="stretch", hide_index=True)
    st.download_button(
        "Download prediction CSV",
        data=output_path.read_bytes(),
        file_name=output_path.name,
        mime="text/csv",
        type="primary",
        key=f"download_predictions_{input_fingerprint[:12]}_{selected.run_id}",
    )

    with st.expander("Prediction execution record", expanded=False):
        safe_result = {
            key: value
            for key, value in result.items()
            if key not in {"_ui_context", "_ui_input_fingerprint"}
        }
        st.json(safe_result)
