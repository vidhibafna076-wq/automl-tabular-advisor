from datetime import datetime
from pathlib import Path
from typing import Any

from src.state import ExperimentState


def _value(value: Any, default: str = "Not available") -> str:
    """
    Convert values into report-safe text.
    """

    if value is None:
        return default

    return str(value)


def _format_list(items: list[Any]) -> str:
    """
    Format a list as Markdown bullet points.
    """

    if not items:
        return "- None"

    return "\n".join(f"- {item}" for item in items)


def _format_key_value_dict(data: dict[str, Any]) -> str:
    """
    Format a dictionary as Markdown bullet points.
    """

    if not data:
        return "- None"

    lines = []

    for key, value in data.items():
        lines.append(f"- **{key}:** {value}")

    return "\n".join(lines)


def _format_provenance(provenance: dict[str, Any]) -> str:
    """Format reproducibility metadata without assuming a current schema."""

    if not provenance:
        return "No reproducibility metadata was recorded for this run."

    lines = [
        f"- **Run ID:** {_value(provenance.get('run_id'))}",
        f"- **Run created at (UTC):** {_value(provenance.get('created_at_utc'))}",
        f"- **Resumed from run:** {_value(provenance.get('resumed_from_run_id'))}",
        f"- **Dataset SHA-256:** `{_value(provenance.get('dataset_sha256'))}`",
        f"- **Git revision:** `{_value(provenance.get('git_revision'))}`",
        f"- **Python:** {_value(provenance.get('python_implementation'))} "
        f"{_value(provenance.get('python_version'))}",
        f"- **Platform:** {_value(provenance.get('platform'))}",
    ]

    package_data = provenance.get("package_versions") or {}
    lines.extend(["", "## Runtime Package Versions", ""])
    if package_data:
        lines.extend(["| Package | Version |", "|---|---|"])
        for package, package_version in sorted(package_data.items()):
            lines.append(f"| {package} | {package_version} |")
    else:
        lines.append("No runtime package versions were recorded.")

    return "\n".join(lines)


def _format_quality_issues(issues: list[dict[str, Any]]) -> str:
    """
    Format data quality issues for the report.
    """

    if not issues:
        return "No data quality issues were recorded."

    lines = []

    for index, issue in enumerate(issues, start=1):
        lines.append(f"### {index}. {issue['issue']} [{issue['severity'].upper()}]")
        lines.append(f"**Finding:** {issue['finding']}")
        lines.append("")
        lines.append(f"**Why it matters:** {issue['why_it_matters']}")
        lines.append("")
        lines.append(f"**Recommendation:** {issue['recommendation']}")
        lines.append("")

    return "\n".join(lines)


def _format_experiment_plan(plan: list[dict[str, Any]]) -> str:
    """
    Format experiment plan steps.
    """

    if not plan:
        return "No experiment plan was created."

    lines = []

    for index, step in enumerate(plan, start=1):
        approval = "Yes" if step["requires_user_approval"] else "No"

        lines.append(f"### {index}. {step['title']} [{step['priority'].upper()}]")
        lines.append(f"**Action:** {step['action']}")
        lines.append("")
        lines.append(f"**Reason:** {step['reason']}")
        lines.append("")
        lines.append(f"**Requires user approval:** {approval}")
        lines.append("")
        lines.append(f"**Status:** {step['status']}")
        lines.append("")

    return "\n".join(lines)


def _format_model_results(model_results: list[dict[str, Any]]) -> str:
    """
    Format model training results.
    """

    if not model_results:
        return "No model results were created."

    lines = []

    for result in model_results:
        lines.append(f"### {result['display_name']}")
        lines.append(f"- **Model ID:** {result['model_id']}")
        lines.append(f"- **Family:** {result['family']}")
        lines.append(f"- **Status:** {result['status']}")
        lines.append(f"- **Primary metric:** {result['primary_metric']}")
        lines.append(f"- **Primary score:** {result['primary_score']}")

        if result.get("error"):
            lines.append(f"- **Error:** {result['error']}")
            lines.append("")
            continue

        lines.append("")
        lines.append("| Metric | CV Mean | CV Std | Train Mean |")
        lines.append("|---|---:|---:|---:|")

        for metric_name, values in result.get("metrics", {}).items():
            lines.append(
                f"| {metric_name} | "
                f"{values.get('cv_mean')} | "
                f"{values.get('cv_std')} | "
                f"{values.get('train_mean')} |"
            )

        diagnostics = result.get("classification_diagnostics", {})

        if diagnostics:
            lines.append("")
            lines.append("#### Classification Diagnostics")
            lines.append("")

            if diagnostics.get("positive_label") is not None:
                lines.append(f"- **Positive label:** {diagnostics['positive_label']}")

            if diagnostics.get("out_of_fold_roc_auc") is not None:
                lines.append(
                    f"- **Out-of-fold ROC-AUC:** {diagnostics['out_of_fold_roc_auc']}"
                )

            if diagnostics.get("out_of_fold_pr_auc") is not None:
                lines.append(
                    f"- **Out-of-fold PR-AUC:** {diagnostics['out_of_fold_pr_auc']}"
                )

            lines.append("")
            lines.append("**Labels:**")
            lines.append("")
            lines.append(_format_list(diagnostics.get("labels", [])))
            lines.append("")

            lines.append("**Confusion matrix:**")
            lines.append("")
            lines.append("```text")

            matrix = diagnostics.get("confusion_matrix", [])

            if matrix:
                for row in matrix:
                    lines.append(str(row))
            else:
                lines.append("Not available")

            lines.append("```")
            lines.append("")

            lines.append("**Class-level metrics:**")
            lines.append("")
            lines.append("| Class | Precision | Recall | F1 | Support |")
            lines.append("|---|---:|---:|---:|---:|")

            for label, values in diagnostics.get("class_level_metrics", {}).items():
                lines.append(
                    f"| {label} | "
                    f"{values.get('precision')} | "
                    f"{values.get('recall')} | "
                    f"{values.get('f1_score')} | "
                    f"{values.get('support')} |"
                )

        lines.append("")

    return "\n".join(lines)

def _format_leaderboard(leaderboard: list[dict[str, Any]]) -> str:
    """
    Format model leaderboard.
    """

    if not leaderboard:
        return "No leaderboard was created."

    lines = []
    lines.append("| Rank | Model | Primary Score | Beats Dummy | Reliability | Overfit Gap |")
    lines.append("|---:|---|---:|---|---|---:|")

    for item in leaderboard:
        beats_dummy = item["improvement_over_dummy"]["beats_dummy"]

        lines.append(
            f"| {item['rank']} | "
            f"{item['display_name']} | "
            f"{item['primary_score']} | "
            f"{beats_dummy} | "
            f"{item['reliability_level']} | "
            f"{item['overfit_gap']} |"
        )

    lines.append("")
    lines.append("## Leaderboard Reliability Flags")
    lines.append("")

    for item in leaderboard:
        lines.append(f"### {item['display_name']}")

        if item["reliability_flags"]:
            for flag in item["reliability_flags"]:
                lines.append(
                    f"- **[{flag['severity'].upper()}] {flag['issue']}:** "
                    f"{flag['finding']} Recommendation: {flag['recommendation']}"
                )
        else:
            lines.append("- No reliability flags.")

        lines.append("")

    return "\n".join(lines)


def _format_critic_report(critic_report: dict[str, Any]) -> str:
    """
    Format the critic report.
    """

    if not critic_report:
        return "No critic report was created."

    lines = []

    lines.append(f"- **Overall reliability:** {critic_report['overall_reliability']}")
    lines.append(f"- **Recommendation decision:** {critic_report['recommendation_decision']}")
    lines.append(
        "- **Candidate met tuning reliability gate:** "
        f"{critic_report['can_proceed_to_tuning']}"
    )
    lines.append("")

    lines.append("## Selected Candidate")
    lines.append("")

    selected_candidate = critic_report.get("selected_candidate")

    if selected_candidate:
        lines.append(f"- **Model:** {selected_candidate['display_name']}")
        lines.append(f"- **Model ID:** {selected_candidate['model_id']}")
        lines.append(f"- **Primary metric:** {selected_candidate['primary_metric']}")
        lines.append(f"- **Primary score:** {selected_candidate['primary_score']}")
        lines.append(f"- **Reliability level:** {selected_candidate['reliability_level']}")
    else:
        lines.append("- No model was selected.")

    lines.append("")
    lines.append("## Critic Findings")
    lines.append("")

    findings = critic_report.get("findings", [])

    if not findings:
        lines.append("- No critic findings.")
    else:
        for index, finding in enumerate(findings, start=1):
            lines.append(f"### {index}. {finding['issue']} [{finding['severity'].upper()}]")
            lines.append(f"**Evidence:** {finding['evidence']}")
            lines.append("")
            lines.append(f"**Recommendation:** {finding['recommendation']}")
            lines.append("")

    lines.append("## Next Actions")
    lines.append("")
    lines.append(_format_list(critic_report.get("next_actions", [])))

    return "\n".join(lines)


def _format_tool_history(tool_history: list[dict[str, Any]]) -> str:
    """
    Format tool execution history.
    """

    if not tool_history:
        return "- No tool history recorded."

    lines = []

    for index, event in enumerate(tool_history, start=1):
        lines.append(
            f"{index}. **[{event['status'].upper()}]** "
            f"{event['tool_name']} - {event['message']}"
        )

    return "\n".join(lines)

def _format_holdout_result(result: dict[str, Any]) -> str:
    """
    Format the final holdout evaluation result.
    """

    if not result:
        return "No final holdout evaluation result was recorded."

    lines = []

    lines.append(f"- **Status:** {result.get('status')}")
    lines.append(f"- **Holdout used:** {result.get('holdout_used')}")
    lines.append(f"- **Reason:** {result.get('reason')}")

    if result.get("model_id"):
        lines.append(f"- **Model:** {result.get('display_name')}")
        lines.append(f"- **Model ID:** {result.get('model_id')}")
        lines.append(
            f"- **Training rows:** {result.get('training_rows')}"
        )
        lines.append(
            f"- **Holdout rows:** {result.get('holdout_rows')}"
        )
        lines.append(
            f"- **Primary metric:** {result.get('primary_metric')}"
        )
        lines.append(
            f"- **Cross-validation score:** "
            f"{result.get('cv_primary_score')}"
        )
        lines.append(
            f"- **Holdout score:** "
            f"{result.get('holdout_primary_score')}"
        )
        lines.append(
            f"- **Passed holdout guardrail:** "
            f"{result.get('passes_holdout_guardrail')}"
        )
        lines.append(
            f"- **Guardrail explanation:** "
            f"{result.get('guardrail_reason')}"
        )

    metrics = result.get("metrics", {})

    if metrics:
        lines.append("")
        lines.append("| Holdout Metric | Value |")
        lines.append("|---|---:|")

        for metric_name, metric_value in metrics.items():
            lines.append(
                f"| {metric_name} | {metric_value} |"
            )

    return "\n".join(lines)

def _format_tuning_result(
    summary: dict[str, Any],
    result: dict[str, Any],
) -> str:
    """
    Format the guarded tuning decision and result.
    """

    if not summary and not result:
        return "No guarded tuning decision was recorded."

    lines = []

    lines.append(
        f"- **Eligible:** {summary.get('eligible')}"
    )
    lines.append(
        f"- **Eligibility reason:** {summary.get('reason')}"
    )

    blocking_reasons = summary.get(
        "blocking_reasons",
        [],
    )

    if blocking_reasons:
        lines.append("")
        lines.append("### Blocking Reasons")
        lines.append("")
        lines.append(_format_list(blocking_reasons))

    if result:
        lines.append("")
        lines.append(
            f"- **Tuning status:** {result.get('status')}"
        )
        lines.append(
            f"- **Model:** {result.get('display_name')}"
        )
        lines.append(
            f"- **Primary metric:** "
            f"{result.get('primary_metric')}"
        )
        lines.append(
            f"- **Baseline CV score:** "
            f"{result.get('baseline_cv_score')}"
        )
        lines.append(
            f"- **Tuned CV score:** "
            f"{result.get('tuned_cv_score')}"
        )
        lines.append(
            f"- **Improvement:** {result.get('improvement')}"
        )
        lines.append(
            f"- **Accepted for final evaluation:** "
            f"{result.get('accepted_for_final_evaluation')}"
        )
        lines.append(
            f"- **Successful trials:** "
            f"{result.get('trials_completed')}"
        )
        lines.append(
            f"- **Duration seconds:** "
            f"{result.get('duration_seconds')}"
        )
        lines.append(
            f"- **Holdout used during tuning:** "
            f"{result.get('holdout_used_during_tuning')}"
        )

        best_params = result.get("best_params", {})

        if best_params:
            lines.append("")
            lines.append("### Best Parameters")
            lines.append("")
            lines.append(_format_key_value_dict(best_params))

    return "\n".join(lines)

def create_final_report_summary(state: ExperimentState) -> dict[str, Any]:
    """
    Create a short JSON-safe report summary.
    """

    critic_report = state.critic_report
    comparison_summary = state.comparison_summary

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "report_type": "final_automl_advisor_report",
        "dataset_path": state.dataset_path,
        "target_column": state.target_column,
        "task_type": state.task_type,
        "overall_reliability": critic_report.get("overall_reliability"),
        "recommendation_decision": critic_report.get("recommendation_decision"),
        "can_proceed_to_tuning": critic_report.get("can_proceed_to_tuning"),
        "best_overall_model": comparison_summary.get("best_overall_display_name"),
        "best_useful_model": comparison_summary.get("best_useful_display_name"),
        "models_trained": state.training_summary.get("models_completed"),
        "run_id": state.provenance.get("run_id"),
        "dataset_sha256": state.provenance.get("dataset_sha256"),
        "git_revision": state.provenance.get("git_revision"),
        "final_status": state.status,
    }


def _result_status(result: dict[str, Any]) -> str:
    """Return a concise status for a structured workflow result."""

    status = str((result or {}).get("status", "")).strip().lower()
    return status or "not recorded"


def _format_final_conclusion(
    state: ExperimentState,
    decision: str,
) -> str:
    """Create a conclusion that reflects stages already executed."""

    tuning_status = _result_status(
        getattr(state, "tuning_result", {}) or {}
    )
    holdout_status = _result_status(
        getattr(state, "holdout_result", {}) or {}
    )
    evidence_summary = (
        "The controlled workflow completed through reliability review, model "
        "persistence assessment, and reporting. Guarded tuning status: "
        f"**{tuning_status}**. Final holdout status: **{holdout_status}**."
    )

    if decision == "do_not_recommend_model":
        return (
            f"{evidence_summary} The AutoML advisor should not recommend a final "
            "model from this run. Review the critic findings and improve the "
            "modelling evidence before rerunning the workflow."
        )

    if decision == "recommend_with_caution":
        return (
            f"{evidence_summary} The selected candidate remains provisional. "
            "Review the critic findings, domain suitability, and independent "
            "validation requirements before use."
        )

    if decision == "recommend_candidate_model":
        return (
            f"{evidence_summary} The critic approved the selected candidate based "
            "on the current evidence. Review feature importance, artifact "
            "compatibility, domain suitability, and deployment constraints before use."
        )

    return (
        f"{evidence_summary} No recognised final recommendation was recorded. "
        "Inspect the critic report and execution timeline before using any artifact."
    )


def create_final_report_markdown(state: ExperimentState) -> str:
    """
    Create the final Markdown report from experiment state.
    """

    profile = state.profile
    preprocessing_config = state.preprocessing_config
    training_summary = state.training_summary
    comparison_summary = state.comparison_summary
    critic_report = state.critic_report

    lines = []

    lines.append("# Agentic AutoML Advisor Report")
    lines.append("")
    lines.append(f"**Created at:** {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"**Dataset:** `{state.dataset_path}`")
    lines.append("")
    lines.append(f"**Target column:** `{state.target_column}`")
    lines.append("")
    lines.append(f"**User objective:** {_value(state.user_objective)}")
    lines.append("")

    lines.append("---")
    lines.append("")

    lines.append("# 1. Executive Summary")
    lines.append("")

    decision = critic_report.get("recommendation_decision", "Not available")
    reliability = critic_report.get("overall_reliability", "Not available")

    lines.append(f"The experiment completed with overall reliability marked as **{reliability}**.")
    lines.append("")
    lines.append(f"The final recommendation decision is: **{decision}**.")
    lines.append("")

    if decision == "do_not_recommend_model":
        lines.append(
            "No final model should be recommended from this run. The workflow executed "
            "successfully, but the evidence is not strong enough for real model selection."
        )
    elif decision == "recommend_with_caution":
        lines.append(
            "A tentative model candidate may be considered, but it should be treated "
            "with caution and validated further."
        )
    elif decision == "recommend_candidate_model":
        lines.append(
            "A candidate model can be recommended based on the current experiment evidence."
        )
    else:
        lines.append("No final decision was available.")

    lines.append("")

    lines.append("# 2. Dataset Overview")
    lines.append("")
    lines.append(f"- **Rows:** {profile.get('rows')}")
    lines.append(f"- **Columns:** {profile.get('columns')}")
    lines.append(f"- **Feature count:** {profile.get('feature_count')}")
    lines.append(f"- **Target column:** {profile.get('target_column')}")
    lines.append("")
    lines.append("## Numerical Features")
    lines.append("")
    lines.append(_format_list(profile.get("numerical_columns", [])))
    lines.append("")
    lines.append("## Categorical Features")
    lines.append("")
    lines.append(_format_list(profile.get("categorical_columns", [])))
    lines.append("")

    lines.append("# 3. Task Detection")
    lines.append("")
    lines.append(f"- **Task type:** {state.task_type}")
    lines.append(f"- **Base task:** {state.base_task}")
    lines.append(f"- **Reason:** {state.task_reason}")
    lines.append("")
    lines.append("## Target Summary")
    lines.append("")
    lines.append(_format_key_value_dict(state.target_summary))
    lines.append("")

    lines.append("# 4. Data Quality Findings")
    lines.append("")
    lines.append(_format_quality_issues(state.quality_issues))
    lines.append("")

    lines.append("# 5. Experiment Plan")
    lines.append("")
    lines.append(_format_experiment_plan(state.experiment_plan))
    lines.append("")

    lines.append("# 6. Preprocessing Strategy")
    lines.append("")
    lines.append("## Columns Dropped")
    lines.append("")
    lines.append(_format_list(preprocessing_config.get("columns_to_drop", [])))
    lines.append("")
    lines.append("## Final Numerical Features")
    lines.append("")
    lines.append(_format_list(preprocessing_config.get("numerical_features", [])))
    lines.append("")
    lines.append("## Final Categorical Features")
    lines.append("")
    lines.append(_format_list(preprocessing_config.get("categorical_features", [])))
    lines.append("")
    lines.append("## Missing Value Strategy")
    lines.append("")
    lines.append(
        f"- **Numerical:** "
        f"{preprocessing_config.get('missing_value_strategy', {}).get('numerical', {}).get('strategy')}"
    )
    lines.append(
        f"- **Categorical:** "
        f"{preprocessing_config.get('missing_value_strategy', {}).get('categorical', {}).get('strategy')}"
    )
    lines.append("")
    lines.append("## Leakage Control")
    lines.append("")
    lines.append(
        "The preprocessing pipeline is designed to be fitted inside cross-validation "
        "folds, not on the full dataset before evaluation."
    )
    lines.append("")

    lines.append("# 7. Training Summary")
    lines.append("")
    lines.append(f"- **Task type:** {training_summary.get('task_type')}")
    lines.append(f"- **Validation method:** {training_summary.get('validation_method')}")
    lines.append(f"- **CV folds:** {training_summary.get('cv_folds')}")
    lines.append(f"- **Primary metric:** {training_summary.get('primary_metric')}")
    lines.append(f"- **Models attempted:** {training_summary.get('models_attempted')}")
    lines.append(f"- **Models completed:** {training_summary.get('models_completed')}")
    lines.append(f"- **Models failed:** {training_summary.get('models_failed')}")
    lines.append("")
    lines.append(f"**Leakage control:** {training_summary.get('leakage_control')}")
    lines.append("")

    lines.append("## Validation Design")
    lines.append("")

    validation_summary = state.validation_summary or {}

    if validation_summary:
        lines.append(
            f"- **Strategy:** {validation_summary.get('strategy')}"
        )
        lines.append(
            f"- **Cross-validation used:** "
            f"{validation_summary.get('cross_validation_used')}"
        )
        lines.append(
            f"- **Rows used for cross-validation:** "
            f"{validation_summary.get('cross_validation_rows')}"
        )
        lines.append(
            f"- **Final holdout used:** "
            f"{validation_summary.get('holdout_used')}"
        )
        lines.append(
            f"- **Holdout rows:** "
            f"{validation_summary.get('holdout_rows')}"
        )
        lines.append(
            f"- **Reason:** {validation_summary.get('reason')}"
        )
    else:
        lines.append(
            "No validation strategy summary was recorded."
        )

    lines.append("")

    lines.append("## Guarded Hyperparameter Tuning")
    lines.append("")
    lines.append(
        _format_tuning_result(
            state.tuning_summary,
            state.tuning_result,
        )
    )
    lines.append("")

    lines.append("## Final Holdout Evaluation")
    lines.append("")
    lines.append(_format_holdout_result(state.holdout_result))
    lines.append("")

    

    unsupported_metrics = training_summary.get("unsupported_metrics_in_first_trainer", [])

    if unsupported_metrics:
        lines.append("## Metrics Planned But Not Yet Implemented")
        lines.append("")
        lines.append(_format_list(unsupported_metrics))
        lines.append("")


    lines.append("# 8. Model Results")
    lines.append("")
    lines.append(_format_model_results(state.model_results))
    lines.append("")

    lines.append("# 9. Leaderboard and Comparison")
    lines.append("")
    lines.append(f"- **Primary metric:** {comparison_summary.get('primary_metric')}")
    lines.append(f"- **Dummy baseline score:** {comparison_summary.get('dummy_baseline_score')}")
    lines.append(f"- **Best overall model:** {comparison_summary.get('best_overall_display_name')}")
    lines.append(f"- **Best useful model:** {comparison_summary.get('best_useful_display_name')}")
    lines.append(f"- **Recommendation status:** {comparison_summary.get('recommendation_status')}")
    lines.append("")
    lines.append("## Important Notes")
    lines.append("")
    lines.append(_format_list(comparison_summary.get("important_notes", [])))
    lines.append("")
    lines.append("## Leaderboard")
    lines.append("")
    lines.append(_format_leaderboard(state.leaderboard))
    lines.append("")

    lines.append("# 10. Reliability Critic Report")
    lines.append("")
    lines.append(_format_critic_report(state.critic_report))
    lines.append("")

    lines.append("# 11. Model Artifact and Explainability")
    lines.append("")
    lines.append("## Model Artifact Status")
    lines.append("")
    lines.append(_format_model_artifact_summary(state.model_artifact_summary))
    lines.append("")
    lines.append("## Feature Importance")
    lines.append("")
    lines.append(_format_feature_importance_summary(state.feature_importance_summary))
    lines.append("")

    lines.append("# 12. Tool Execution Timeline")
    lines.append("")
    lines.append(_format_tool_history(state.tool_history))
    lines.append("")

    lines.append("# 13. Reproducibility and Provenance")
    lines.append("")
    lines.append(_format_provenance(state.provenance))
    lines.append("")

    lines.append("# 14. Final Conclusion")
    lines.append("")
    lines.append(_format_final_conclusion(state, decision))

    lines.append("")

    return "\n".join(lines)

def _format_model_artifact_summary(summary: dict[str, Any]) -> str:
    """
    Format final model artifact status for the report.
    """

    if not summary:
        return "No model artifact decision was recorded."

    lines = []

    lines.append(f"- **Status:** {summary.get('status')}")
    lines.append(f"- **Reason:** {summary.get('reason')}")
    lines.append(f"- **Critic decision:** {summary.get('critic_decision')}")

    if summary.get("artifact_path"):
        lines.append(f"- **Artifact path:** `{summary.get('artifact_path')}`")

    if summary.get("metadata_path"):
        lines.append(f"- **Metadata path:** `{summary.get('metadata_path')}`")

    return "\n".join(lines)


def _format_feature_importance_summary(summary: dict[str, Any]) -> str:
    """
    Format basic feature importance summary.
    """

    if not summary:
        return "No feature importance summary was recorded."

    if summary.get("status") != "available":
        return (
            f"Feature importance status: **{summary.get('status')}**. "
            f"Reason: {summary.get('reason')}"
        )

    lines = []

    lines.append(f"- **Importance type:** {summary.get('importance_type')}")
    lines.append("")
    lines.append("| Rank | Feature | Importance |")
    lines.append("|---:|---|---:|")

    for item in summary.get("top_features", []):
        lines.append(
            f"| {item.get('rank')} | "
            f"{item.get('feature')} | "
            f"{item.get('importance')} |"
        )

    return "\n".join(lines)


def save_final_report(markdown_text: str, output_path: str) -> str:
    """
    Save the final report to disk.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write(markdown_text)

    return str(path)