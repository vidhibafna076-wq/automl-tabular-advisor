"""Evidence-first experiment results and human approval workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd
import streamlit as st

from .components import (
    display_tags,
    render_banner,
    render_finding,
    render_section_title,
    render_workflow,
    safe_container_with_border,
)
from .constants import (
    PAUSED_APPROVAL_STATUS,
    RECOMMENDATION_LABELS,
)
from .helpers import (
    best_leaderboard_item,
    display_value,
    expected_feature_columns,
    first_mapping,
    first_value,
    format_metric,
    holdout_evidence,
    humanise,
    resolve_artifact_path,
    state_attr,
    tuning_evidence,
)


def _run_id(orchestrator_result: dict[str, Any]) -> str:
    return str(orchestrator_result.get("ui_run_id", "current_run"))


def _run_dir(orchestrator_result: dict[str, Any]) -> Path | None:
    value = orchestrator_result.get("ui_run_dir")
    return Path(value) if value else None


def _decision_content(
    state: Any,
    orchestrator_result: dict[str, Any],
) -> tuple[str, str, str]:
    if state_attr(state, "status") == PAUSED_APPROVAL_STATUS:
        return (
            "warning",
            "Human decision required",
            "Review every proposed preprocessing change below, then resume the same evaluation.",
        )

    if not orchestrator_result.get("success", False):
        return (
            "danger",
            "The run did not complete",
            "No model decision should be used from this incomplete run.",
        )

    critic = first_mapping(state, "critic_report")
    decision = critic.get("recommendation_decision")
    reliability = humanise(critic.get("overall_reliability"), "unknown")

    if decision == "do_not_recommend_model":
        return (
            "danger",
            "Model recommendation withheld",
            f"The critic rated the evidence {reliability.lower()} and did not support persisting a candidate.",
        )
    if decision == "recommend_with_caution":
        return (
            "warning",
            "Provisional candidate only",
            f"The critic rated the evidence {reliability.lower()}. Use the model for further validation, not as deployment approval.",
        )
    if decision == "recommend_candidate_model":
        selected = critic.get("selected_candidate", {}) or {}
        model = selected.get("display_name", "a candidate model")
        return (
            "success",
            "Candidate supported by current evidence",
            f"The critic selected {model} with {reliability.lower()} reliability. Review holdout and limitation evidence before use.",
        )
    return (
        "warning",
        "No final recommendation recorded",
        "The workflow finished without a recognised critic decision. Inspect the execution record before continuing.",
    )


def _approval_title(approval: Any, index: int) -> str:
    if isinstance(approval, str):
        return approval
    if isinstance(approval, Mapping):
        return str(
            first_value(
                approval,
                "title",
                "decision",
                "issue",
                "action",
                "message",
                default=f"Preprocessing decision {index + 1}",
            )
        )
    return f"Preprocessing decision {index + 1}"


def _approval_detail(approval: Any) -> str:
    if not isinstance(approval, Mapping):
        return "The preprocessing planner requested a human decision."
    return str(
        first_value(
            approval,
            "reason",
            "finding",
            "evidence",
            "description",
            default="The preprocessing planner requested a human decision.",
        )
    )


def render_approval_panel(
    state: Any,
    orchestrator_result: dict[str, Any],
    on_resume: Callable[[list[dict[str, Any]]], None],
) -> None:
    config = first_mapping(state, "preprocessing_config")
    approvals = list(config.get("pending_approvals", []) or [])
    run_id = _run_id(orchestrator_result)

    render_section_title(
        "Review preprocessing decisions",
        "Nothing is approved by default. Choose an outcome for every requested change.",
    )
    if not approvals:
        st.warning(
            "The backend reported an approval pause but did not include individual decisions. "
            "Start a revised experiment after checking the preprocessing artifact."
        )
        return

    choices: list[str] = []
    with st.form(f"approval_form_{run_id}"):
        for index, approval in enumerate(approvals):
            with safe_container_with_border():
                st.markdown(f"#### {index + 1}. {_approval_title(approval, index)}")
                st.write(_approval_detail(approval))
                if isinstance(approval, Mapping):
                    columns = (
                        approval.get("columns")
                        or approval.get("column_names")
                        or [approval.get("column")]
                    )
                    columns = [value for value in columns if value]
                    if columns:
                        st.caption("Affected columns: " + ", ".join(map(str, columns)))
                choice = st.radio(
                    "Your decision",
                    (
                        "Choose an outcome",
                        "Approve the proposed change",
                        "Reject it and keep the affected columns",
                    ),
                    key=f"approval_{run_id}_{index}",
                )
                choices.append(choice)

        confirmed = st.checkbox(
            "I have reviewed every decision and understand that preprocessing changes affect model results.",
            key=f"approval_confirm_{run_id}",
        )
        submitted = st.form_submit_button(
            "Apply decisions and resume",
            type="primary",
        )

    if submitted:
        if any(choice == "Choose an outcome" for choice in choices):
            st.error("Choose an outcome for every preprocessing decision.")
            return
        if not confirmed:
            st.error(
                "Confirm that you reviewed every preprocessing decision before resuming."
            )
            return
        decisions = [
            {
                "action": (
                    "approve"
                    if choice == "Approve the proposed change"
                    else "reject"
                )
            }
            for choice in choices
        ]
        on_resume(decisions)


def make_leaderboard_dataframe(leaderboard: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in leaderboard:
        improvement = item.get("improvement_over_dummy", {}) or {}
        rows.append(
            {
                "Rank": item.get("rank"),
                "Model": item.get("display_name"),
                "Family": item.get("family"),
                "Metric": item.get("primary_metric"),
                "CV score": format_metric(item.get("primary_score")),
                "CV std": format_metric(item.get("primary_cv_std")),
                "Train score": format_metric(item.get("primary_train_score")),
                "Dummy score": format_metric(item.get("dummy_baseline_score")),
                "Improvement": format_metric(improvement.get("raw_improvement")),
                "Beats dummy": improvement.get("beats_dummy"),
                "Reliability": item.get("reliability_level"),
            }
        )
    return pd.DataFrame(rows)


def make_training_metrics_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        for metric_name, values in (result.get("metrics", {}) or {}).items():
            if not isinstance(values, Mapping):
                continue
            rows.append(
                {
                    "Model": result.get("display_name"),
                    "Metric": metric_name,
                    "CV mean": format_metric(values.get("cv_mean")),
                    "CV std": format_metric(values.get("cv_std")),
                    "Train mean": format_metric(values.get("train_mean")),
                }
            )
    return pd.DataFrame(rows)


def show_leaderboard_chart(state: Any) -> None:
    rows: list[dict[str, Any]] = []
    leaderboard = state_attr(state, "leaderboard", default=[]) or []
    for item in leaderboard:
        score = format_metric(item.get("primary_score"))
        dummy = format_metric(item.get("dummy_baseline_score"))
        if score is not None:
            rows.append(
                {
                    "Model": item.get("display_name"),
                    "Candidate CV score": score,
                    "Dummy baseline": dummy,
                }
            )

    if not rows:
        st.info("No model scores are available to chart.")
        return
    chart = pd.DataFrame(rows).set_index("Model")
    try:
        st.bar_chart(
            chart,
            horizontal=True,
            width="stretch",
            color=["#49d7e8", "#7f8b9b"],
        )
    except TypeError:
        st.bar_chart(chart, horizontal=True, width="stretch")
    st.caption(
        "Cyan is the candidate cross-validation score; grey is the same dummy "
        "baseline repeated for a direct comparison."
    )


def show_status_metrics(state: Any) -> None:
    critic = first_mapping(state, "critic_report")
    training = first_mapping(state, "training_summary")
    comparison = first_mapping(state, "comparison_summary")
    columns = st.columns(4)
    columns[0].metric("Task", humanise(state_attr(state, "task_type")))
    columns[1].metric(
        "Reliability",
        humanise(critic.get("overall_reliability")),
    )
    columns[2].metric(
        "Models completed",
        first_value(training, "models_completed", "completed_models", default=0),
    )
    columns[3].metric(
        "Selected candidate",
        first_value(
            comparison,
            "best_useful_display_name",
            "selected_model_name",
            default="None",
        ),
    )


def _model_artifacts(orchestrator_result: dict[str, Any]) -> list[Path]:
    run_dir = _run_dir(orchestrator_result)
    if not run_dir or not run_dir.exists():
        return []
    return sorted(run_dir.glob("**/*_pipeline.joblib"))


def show_overview_tab(
    state: Any,
    orchestrator_result: dict[str, Any],
    on_predict: Callable[[], None],
) -> None:
    render_section_title(
        "Decision summary",
        "The minimum evidence needed to judge whether this candidate is useful.",
    )
    best = best_leaderboard_item(state)
    comparison = first_mapping(state, "comparison_summary")
    holdout = holdout_evidence(state)
    tuning = tuning_evidence(state)

    cv_score = format_metric(best.get("primary_score")) if best else None
    dummy_score = format_metric(
        first_value(
            comparison,
            "dummy_baseline_score",
            default=best.get("dummy_baseline_score") if best else None,
        )
    )
    improvement = None
    if best:
        improvement = format_metric(
            (best.get("improvement_over_dummy", {}) or {}).get("raw_improvement")
        )
    if improvement is None and cv_score is not None and dummy_score is not None:
        improvement = round(cv_score - dummy_score, 4)

    metrics = st.columns(4)
    metrics[0].metric(
        "Primary metric",
        first_value(comparison, "primary_metric", default="N/A"),
    )
    metrics[1].metric("Cross-validation", display_value(cv_score))
    metrics[2].metric(
        "Versus dummy",
        display_value(improvement),
        help="Absolute difference between the candidate and dummy-baseline score.",
    )
    metrics[3].metric(
        "Final holdout",
        display_value(holdout["score"], "Not evaluated"),
    )

    if best:
        cv_std = format_metric(best.get("primary_cv_std"))
        if cv_std is not None:
            st.caption(
                f"Cross-validation variation is {cv_std:g}. Lower variation usually "
                "suggests more stable results across folds, but it does not prove future performance."
            )

    evidence_columns = st.columns(2, gap="large")
    with evidence_columns[0]:
        status = (
            "Applied"
            if tuning["applied"] is True
            else "Not applied"
            if tuning["applied"] is False
            else "Not recorded"
        )
        st.markdown(
            f"""
            <div class="evidence-card">
                <h3>Tuning evidence</h3>
                <p><strong>Status:</strong> {status}</p>
                <p><strong>Before:</strong> {display_value(tuning["baseline_score"])}</p>
                <p><strong>After:</strong> {display_value(tuning["tuned_score"])}</p>
                <p><strong>Change:</strong> {display_value(tuning["improvement"])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with evidence_columns[1]:
        holdout_status = (
            "Evaluated"
            if holdout["evaluated"] is True
            else "Skipped"
            if holdout["evaluated"] is False
            else "Not recorded"
        )
        gap = None
        if cv_score is not None and holdout["score"] is not None:
            gap = round(holdout["score"] - cv_score, 4)
        st.markdown(
            f"""
            <div class="evidence-card">
                <h3>Untouched holdout evidence</h3>
                <p><strong>Status:</strong> {holdout_status}</p>
                <p><strong>Score:</strong> {display_value(holdout["score"])}</p>
                <p><strong>CV-to-holdout change:</strong> {display_value(gap)}</p>
                <p><strong>Rows:</strong> {display_value(holdout["rows"])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if holdout["evaluated"] is not True:
        st.warning(
            holdout["reason"]
            or "No untouched holdout result was recorded. Treat cross-validation as provisional evidence."
        )

    chart_column, action_column = st.columns([0.62, 0.38], gap="large")
    with chart_column:
        st.markdown("### Candidate comparison")
        show_leaderboard_chart(state)
    with action_column:
        st.markdown("### What to do next")
        actions = first_mapping(state, "critic_report").get("next_actions", []) or []
        if actions:
            for index, action in enumerate(actions, start=1):
                st.markdown(f"**{index}.** {action}")
        else:
            st.info("No critic next actions were recorded.")

        artifacts = _model_artifacts(orchestrator_result)
        decision = first_mapping(state, "critic_report").get("recommendation_decision")
        can_predict = bool(artifacts) and decision in {
            "recommend_candidate_model",
            "recommend_with_caution",
        }
        st.button(
            "Use the saved candidate for prediction",
            type="primary",
            width="stretch",
            disabled=not can_predict,
            on_click=on_predict,
            key=f"use_model_{_run_id(orchestrator_result)}",
        )
        if not can_predict:
            st.caption(
                "Prediction becomes available only when the critic records a usable "
                "decision and a pipeline artifact exists."
            )


def show_data_tab(state: Any) -> None:
    profile = first_mapping(state, "profile")
    config = first_mapping(state, "preprocessing_config")
    render_section_title(
        "Data and preprocessing",
        "Detected structure, consequential transformations, and quality risks.",
    )

    columns = st.columns(4)
    columns[0].metric("Rows", profile.get("rows", "N/A"))
    columns[1].metric("Columns", profile.get("columns", "N/A"))
    columns[2].metric("Features", profile.get("feature_count", "N/A"))
    columns[3].metric("Duplicate rows", profile.get("duplicate_rows", 0))

    feature_column, preprocessing_column = st.columns(2, gap="large")
    with feature_column:
        with safe_container_with_border():
            st.markdown("### Detected columns")
            st.markdown("**Numerical**")
            display_tags(profile.get("numerical_columns", []))
            st.markdown("**Categorical**")
            display_tags(profile.get("categorical_columns", []))
            st.markdown("**Possible identifiers**")
            display_tags(profile.get("possible_id_columns", []))
            st.markdown("**Possible dates**")
            display_tags(profile.get("possible_date_columns", []))
    with preprocessing_column:
        with safe_container_with_border():
            st.markdown("### Preprocessing decisions")
            st.markdown("**Dropped columns**")
            display_tags(config.get("columns_to_drop", []), "No columns were dropped.")
            missing = config.get("missing_value_strategy", {}) or {}
            numerical = (missing.get("numerical", {}) or {}).get("strategy", "N/A")
            categorical = (missing.get("categorical", {}) or {}).get("strategy", "N/A")
            validation = config.get("validation_strategy", {}) or {}
            metric_strategy = config.get("metric_strategy", {}) or {}
            st.write(f"**Numerical missing values:** {numerical}")
            st.write(f"**Categorical missing values:** {categorical}")
            st.write(f"**Validation:** {validation.get('method', 'N/A')}")
            st.write(f"**Primary metric:** {metric_strategy.get('primary_metric', 'N/A')}")
            decisions = config.get("human_approval_decisions", []) or []
            if decisions:
                st.success(f"{len(decisions)} human preprocessing decision(s) recorded.")

    st.markdown("### Expected raw prediction schema")
    expected = expected_feature_columns(state)
    display_tags(
        expected,
        "The experiment artifact did not record enough information to reconstruct the raw input schema.",
    )

    quality_issues = state_attr(state, "quality_issues", default=[]) or []
    st.markdown("### Data quality findings")
    if not quality_issues:
        st.success("No data quality findings were recorded.")
    else:
        for issue in quality_issues:
            render_finding(issue, "finding")

    leakage = state_attr(state, "leakage_warnings", default=[]) or []
    if leakage:
        st.markdown("### Leakage warnings")
        for warning in leakage:
            render_finding(warning, "finding")

    with st.expander("Technical preprocessing artifact", expanded=False):
        st.json(config)


def show_models_tab(state: Any) -> None:
    render_section_title(
        "Model evaluation",
        "Cross-validation, dummy-baseline comparison, and class-level diagnostics.",
    )
    leaderboard = state_attr(state, "leaderboard", default=[]) or []
    dataframe = make_leaderboard_dataframe(leaderboard)
    if dataframe.empty:
        st.info("No leaderboard is available.")
        return

    show_leaderboard_chart(state)
    st.markdown("### Leaderboard")
    st.dataframe(dataframe, width="stretch", hide_index=True)

    results = state_attr(state, "model_results", default=[]) or []
    with st.expander("All training metrics", expanded=False):
        metrics = make_training_metrics_dataframe(results)
        if metrics.empty:
            st.info("No training metrics are available.")
        else:
            st.dataframe(metrics, width="stretch", hide_index=True)

    st.markdown("### Model diagnostics")
    for result in results:
        name = result.get("display_name", "Model")
        score = display_value(result.get("primary_score"))
        with st.expander(f"{name} · primary score {score}", expanded=False):
            summary = st.columns(4)
            summary[0].metric("Status", humanise(result.get("status")))
            summary[1].metric("Family", result.get("family", "N/A"))
            summary[2].metric("Primary metric", result.get("primary_metric", "N/A"))
            summary[3].metric("Primary score", score)

            if result.get("error"):
                st.warning("This candidate did not complete. See the local run log for technical details.")

            diagnostics = result.get("classification_diagnostics", {}) or {}
            if not diagnostics:
                st.caption("No class-level diagnostics were recorded for this model.")
                continue

            diagnostic_columns = st.columns(3)
            diagnostic_columns[0].metric(
                "Positive label",
                diagnostics.get("positive_label", "N/A"),
            )
            diagnostic_columns[1].metric(
                "OOF ROC-AUC",
                display_value(diagnostics.get("out_of_fold_roc_auc")),
            )
            diagnostic_columns[2].metric(
                "OOF PR-AUC",
                display_value(diagnostics.get("out_of_fold_pr_auc")),
            )

            matrix = diagnostics.get("confusion_matrix", [])
            labels = diagnostics.get("labels", [])
            if matrix and labels:
                st.markdown("#### Confusion matrix")
                matrix_df = pd.DataFrame(
                    matrix,
                    index=[f"Actual {label}" for label in labels],
                    columns=[f"Predicted {label}" for label in labels],
                )
                st.dataframe(matrix_df, width="stretch")

            class_metrics = diagnostics.get("class_level_metrics", {})
            if class_metrics:
                st.markdown("#### Class-level metrics")
                st.dataframe(
                    pd.DataFrame.from_dict(class_metrics, orient="index"),
                    width="stretch",
                )


def show_tuning_holdout_tab(state: Any) -> None:
    render_section_title(
        "Tuning and final holdout",
        "Whether optimisation helped and how the final candidate performed on untouched data.",
    )
    tuning = tuning_evidence(state)
    holdout = holdout_evidence(state)

    tuning_columns = st.columns(4)
    tuning_columns[0].metric(
        "Tuning applied",
        "Yes" if tuning["applied"] is True else "No" if tuning["applied"] is False else "Unknown",
    )
    tuning_columns[1].metric("Baseline CV", display_value(tuning["baseline_score"]))
    tuning_columns[2].metric("Tuned CV", display_value(tuning["tuned_score"]))
    tuning_columns[3].metric("Tuning change", display_value(tuning["improvement"]))

    if tuning["parameters"]:
        st.markdown("### Selected hyperparameters")
        st.json(tuning["parameters"])
    elif tuning["reason"]:
        st.info(str(tuning["reason"]))
    else:
        st.info(
            "No selected tuning parameters were recorded. This may mean tuning was skipped "
            "or the backend artifact needs the `tuning_applied` and `tuned_parameters` fields."
        )

    st.markdown("### Untouched holdout")
    holdout_columns = st.columns(4)
    holdout_columns[0].metric(
        "Evaluated",
        "Yes" if holdout["evaluated"] is True else "No" if holdout["evaluated"] is False else "Unknown",
    )
    holdout_columns[1].metric("Metric", holdout["metric"] or "N/A")
    holdout_columns[2].metric("Score", display_value(holdout["score"]))
    holdout_columns[3].metric("Rows", display_value(holdout["rows"]))

    if holdout["evaluated"] is not True:
        st.warning(
            holdout["reason"]
            or "The experiment does not contain a final holdout score. Do not describe this candidate as independently validated."
        )

    with st.expander("Technical tuning artifact", expanded=False):
        st.json(tuning["raw"])
    with st.expander("Technical holdout artifact", expanded=False):
        st.json(holdout["raw"])


def show_reliability_tab(
    state: Any,
    orchestrator_result: dict[str, Any],
) -> None:
    render_section_title(
        "Reliability review",
        "Why the critic supported, qualified, or withheld the candidate recommendation.",
    )
    critic = first_mapping(state, "critic_report")
    if not critic:
        st.info("The workflow did not produce a critic report.")
    else:
        columns = st.columns(3)
        columns[0].metric(
            "Overall reliability",
            humanise(critic.get("overall_reliability")),
        )
        columns[1].metric(
            "Decision",
            RECOMMENDATION_LABELS.get(
                critic.get("recommendation_decision"),
                humanise(critic.get("recommendation_decision")),
            ),
        )
        columns[2].metric(
            "Tuning allowed",
            humanise(critic.get("can_proceed_to_tuning")),
        )

        findings = critic.get("findings", []) or []
        st.markdown("### Critic findings")
        if findings:
            for finding in findings:
                render_finding(finding, "evidence")
        else:
            st.info("No critic findings were recorded.")

    with st.expander("Execution timeline", expanded=False):
        st.write(f"**Run ID:** `{_run_id(orchestrator_result)}`")
        st.write(f"**Final status:** `{state_attr(state, 'status', default='unknown')}`")
        st.write(
            f"**Stop reason:** `{orchestrator_result.get('stop_reason', 'not recorded')}`"
        )
        elapsed = orchestrator_result.get("ui_elapsed_seconds")
        if elapsed is not None:
            st.write(f"**Elapsed time:** {elapsed} seconds")

        actions = orchestrator_result.get("actions_taken", []) or []
        if actions:
            st.markdown("#### Orchestrator actions")
            st.dataframe(pd.DataFrame(actions), width="stretch", hide_index=True)

        history = state_attr(state, "tool_history", default=[]) or []
        if history:
            timeline = pd.DataFrame(
                [
                    {
                        "Timestamp": event.get("timestamp"),
                        "Actor / tool": event.get("tool_name"),
                        "Status": event.get("status"),
                        "Message": event.get("message"),
                    }
                    for event in history
                ]
            )
            st.markdown("#### Tool events")
            st.dataframe(timeline, width="stretch", hide_index=True)


def show_report_tab(state: Any, orchestrator_result: dict[str, Any]) -> None:
    render_section_title(
        "Final report",
        "Download the run-specific experiment record or inspect it in place.",
    )
    run_dir = _run_dir(orchestrator_result)
    path = resolve_artifact_path(state_attr(state, "final_report_path"), run_dir)
    if not path or not path.exists():
        st.info("No readable final report was created for this run.")
        return

    report_text = path.read_text(encoding="utf-8")
    run_id = _run_id(orchestrator_result)
    st.download_button(
        "Download Markdown report",
        data=report_text,
        file_name=f"{run_id}_automl_report.md",
        mime="text/markdown",
        type="primary",
        key=f"download_report_{run_id}",
    )
    with st.expander("Preview report", expanded=False):
        st.markdown(report_text)


def show_results_dashboard(
    state: Any,
    orchestrator_result: dict[str, Any],
    *,
    on_resume: Callable[[list[dict[str, Any]]], None],
    on_predict: Callable[[], None],
) -> None:
    render_workflow(state)
    tone, title, message = _decision_content(state, orchestrator_result)
    render_banner(tone, title, message)

    if state_attr(state, "status") == PAUSED_APPROVAL_STATUS:
        render_approval_panel(state, orchestrator_result, on_resume)
        with st.expander("Inspect paused preprocessing artifact", expanded=False):
            st.json(first_mapping(state, "preprocessing_config"))
        return

    show_status_metrics(state)
    tabs = st.tabs(
        [
            "Decision",
            "Data",
            "Models",
            "Tuning & holdout",
            "Reliability",
            "Report",
        ]
    )
    with tabs[0]:
        show_overview_tab(state, orchestrator_result, on_predict)
    with tabs[1]:
        show_data_tab(state)
    with tabs[2]:
        show_models_tab(state)
    with tabs[3]:
        show_tuning_holdout_tab(state)
    with tabs[4]:
        show_reliability_tab(state, orchestrator_result)
    with tabs[5]:
        show_report_tab(state, orchestrator_result)
