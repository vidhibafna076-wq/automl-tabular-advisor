from datetime import datetime
from typing import Any

from src.state import ExperimentState


def _add_tool_event(
    state: ExperimentState,
    tool_name: str,
    status: str,
    message: str,
) -> None:
    """
    Add a factual event to the experiment tool history.
    """

    state.tool_history.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tool_name": tool_name,
            "status": status,
            "message": message,
        }
    )


def _add_finding(
    findings: list[dict[str, Any]],
    severity: str,
    issue: str,
    evidence: str,
    recommendation: str,
) -> None:
    """
    Add one structured critic finding.
    """

    findings.append(
        {
            "severity": severity,
            "issue": issue,
            "evidence": evidence,
            "recommendation": recommendation,
        }
    )


def _get_non_dummy_models(
    leaderboard: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Return leaderboard entries that are not dummy baselines.
    """

    return [
        item
        for item in leaderboard
        if not item.get("is_dummy_baseline", False)
    ]


def _get_best_useful_model(
    leaderboard: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Return the first non-dummy model that beats the dummy baseline.
    """

    for item in leaderboard:
        if item.get("is_dummy_baseline", False):
            continue

        improvement = item.get("improvement_over_dummy", {})

        if improvement.get("beats_dummy"):
            return item

    return None


def _count_findings_by_severity(
    findings: list[dict[str, Any]],
    severity: str,
) -> int:
    """
    Count critic findings with a specific severity.
    """

    return sum(
        1
        for finding in findings
        if finding.get("severity") == severity
    )


def _normalise_flag_severity(severity: str | None) -> str:
    """
    Convert a reliability flag severity into a supported critic severity.
    """

    supported_severities = {
        "high",
        "medium",
        "low",
        "info",
    }

    normalised = str(severity or "medium").strip().lower()

    if normalised not in supported_severities:
        return "medium"

    return normalised


def _add_selected_candidate_flags(
    findings: list[dict[str, Any]],
    selected_candidate: dict[str, Any],
) -> None:
    """
    Add reliability findings that directly affect the selected candidate.

    Candidate-specific warnings can affect the final recommendation.
    """

    display_name = selected_candidate.get(
        "display_name",
        selected_candidate.get("model_id", "Selected candidate"),
    )

    for flag in selected_candidate.get("reliability_flags", []):
        issue = flag.get("issue", "Candidate reliability warning")
        evidence = (
            flag.get("finding")
            or flag.get("evidence")
            or "A reliability warning was recorded for the selected candidate."
        )
        recommendation = flag.get(
            "recommendation",
            "Review this warning before recommending or tuning the candidate.",
        )

        _add_finding(
            findings=findings,
            severity=_normalise_flag_severity(flag.get("severity")),
            issue=f"Selected candidate: {issue}",
            evidence=(
                f"{display_name} received this reliability warning: "
                f"{evidence}"
            ),
            recommendation=recommendation,
        )


def _add_non_selected_model_findings(
    findings: list[dict[str, Any]],
    non_dummy_models: list[dict[str, Any]],
    selected_model_id: str | None,
) -> None:
    """
    Record warnings from non-selected models as informational findings.

    These warnings explain why alternative models were not preferred, but they
    must not automatically downgrade the selected candidate.
    """

    overfit_models: list[str] = []
    weak_balanced_accuracy_models: list[str] = []

    for item in non_dummy_models:
        if item.get("model_id") == selected_model_id:
            continue

        for flag in item.get("reliability_flags", []):
            issue = flag.get("issue")

            if issue == "Possible overfitting":
                overfit_models.append(item.get("display_name", "Unknown model"))

            if issue == "Weak balanced accuracy":
                weak_balanced_accuracy_models.append(
                    item.get("display_name", "Unknown model")
                )

    if overfit_models:
        model_names = ", ".join(sorted(set(overfit_models)))

        _add_finding(
            findings=findings,
            severity="info",
            issue="Non-selected models showed possible overfitting",
            evidence=(
                "Possible overfitting was flagged for the following "
                f"non-selected model(s): {model_names}. The selected candidate "
                "was not included in this warning."
            ),
            recommendation=(
                "Keep these models out of the final recommendation unless "
                "additional regularisation or stronger evidence becomes available."
            ),
        )

    if weak_balanced_accuracy_models:
        model_names = ", ".join(
            sorted(set(weak_balanced_accuracy_models))
        )

        _add_finding(
            findings=findings,
            severity="info",
            issue="Non-selected models showed weak balanced accuracy",
            evidence=(
                "Weak balanced accuracy was flagged for the following "
                f"non-selected model(s): {model_names}."
            ),
            recommendation=(
                "Do not prefer these alternatives unless their class-level "
                "performance improves."
            ),
        )


def _add_leakage_findings(
    findings: list[dict[str, Any]],
    leakage_warnings: list[dict[str, Any]],
) -> None:
    """
    Convert target leakage warnings into critic findings.
    """

    if not leakage_warnings:
        return

    high_warnings = [
        warning
        for warning in leakage_warnings
        if warning.get("severity") == "high"
    ]

    severity = "high" if high_warnings else "medium"

    flagged_columns = sorted(
        {
            str(warning.get("column"))
            for warning in leakage_warnings
            if warning.get("column")
        }
    )

    column_text = (
        ", ".join(flagged_columns)
        if flagged_columns
        else "one or more feature columns"
    )

    _add_finding(
        findings=findings,
        severity=severity,
        issue="Potential target leakage detected",
        evidence=(
            f"The dataset inspection recorded {len(leakage_warnings)} "
            f"target leakage warning(s). Flagged columns: {column_text}."
        ),
        recommendation=(
            "Review the flagged features before trusting the scores. Remove any "
            "feature that contains future information, post-outcome values, or a "
            "direct copy or proxy of the target, then rerun the experiment."
        ),
    )


def _add_holdout_findings(
    findings: list[dict[str, Any]],
    validation_summary: dict[str, Any],
    holdout_result: dict[str, Any],
    rows: int,
) -> None:
    """
    Add findings about the final untouched holdout evaluation.
    """

    holdout_used = bool(validation_summary.get("holdout_used"))

    if holdout_used:
        holdout_status = holdout_result.get("status")

        if holdout_status != "completed":
            _add_finding(
                findings=findings,
                severity="high",
                issue="Reserved holdout was not successfully evaluated",
                evidence=(
                    "The validation strategy reserved a final holdout set, but "
                    f"the holdout result status is '{holdout_status}'."
                ),
                recommendation=(
                    "Do not recommend or persist a final model until the reserved "
                    "holdout has been evaluated successfully."
                ),
            )
            return

        if holdout_result.get("passes_holdout_guardrail") is not True:
            _add_finding(
                findings=findings,
                severity="high",
                issue="Final holdout guardrail failed",
                evidence=(
                    holdout_result.get("guardrail_reason")
                    or (
                        "The selected candidate did not pass the comparison "
                        "between cross-validation and holdout performance."
                    )
                ),
                recommendation=(
                    "Do not recommend or persist this candidate. Investigate "
                    "overfitting, instability, data drift, or weak generalisation."
                ),
            )
            return

        _add_finding(
            findings=findings,
            severity="info",
            issue="Final holdout evidence passed",
            evidence=(
                f"{holdout_result.get('display_name')} was evaluated on "
                f"{holdout_result.get('holdout_rows')} untouched row(s). "
                f"Cross-validation {holdout_result.get('primary_metric')} was "
                f"{holdout_result.get('cv_primary_score')}, and holdout "
                f"{holdout_result.get('primary_metric')} was "
                f"{holdout_result.get('holdout_primary_score')}. "
                f"{holdout_result.get('guardrail_reason')}"
            ),
            recommendation=(
                "The untouched holdout supports treating the selected model as "
                "the current candidate, subject to normal domain and deployment review."
            ),
        )
        return

    if rows >= 200:
        _add_finding(
            findings=findings,
            severity="medium",
            issue="No final holdout was available",
            evidence=(
                "The dataset had enough total rows to consider a final holdout, "
                "but the validation policy did not create one. This may happen "
                "when the target classes are too small for a safe stratified split."
            ),
            recommendation=(
                "Collect more examples for the smaller target classes or review "
                "the validation policy before making a strong recommendation."
            ),
        )
    else:
        _add_finding(
            findings=findings,
            severity="info",
            issue="Final holdout was not used for the small dataset",
            evidence=(
                f"The dataset contains {rows} row(s), so all usable rows were "
                "retained for cross-validation under the current validation policy."
            ),
            recommendation=(
                "Use a larger and more representative dataset before expecting "
                "strong final holdout evidence."
            ),
        )


def _tuning_follow_up_action(state: ExperimentState) -> str:
    """Describe the guarded-tuning evidence that already exists.

    The reliability critic runs after guarded tuning and final holdout
    evaluation. Its actions must therefore review the recorded tuning outcome
    instead of suggesting that tuning still needs to be started.
    """

    tuning_result = getattr(state, "tuning_result", {}) or {}
    tuning_summary = getattr(state, "tuning_summary", {}) or {}
    status = str(
        tuning_result.get("status")
        or tuning_summary.get("status")
        or "not_recorded"
    ).strip().lower()

    if status == "completed":
        if tuning_result.get("accepted_for_final_evaluation") is True:
            return (
                "Review the accepted guarded-tuning evidence and do not reuse "
                "the final holdout for further model selection."
            )

        return (
            "Retain the baseline candidate unless later evidence justifies "
            "revisiting the guarded-tuning result."
        )

    if status == "skipped":
        return (
            "Guarded tuning was assessed and skipped under the current policy; "
            "revisit it only if the modelling evidence changes."
        )

    return (
        "Review the guarded-tuning record before using the candidate because "
        "no terminal tuning result was recorded."
    )


def create_critic_report(
    state: ExperimentState,
) -> dict[str, Any]:
    """
    Create a reliability critique of the completed AutoML experiment.

    The critic does not train or tune models. It reviews the experiment evidence
    and decides whether the selected useful candidate can be recommended.
    """

    profile = state.profile or {}
    leaderboard = state.leaderboard or []
    comparison_summary = state.comparison_summary or {}
    training_summary = state.training_summary or {}
    validation_summary = getattr(
        state,
        "validation_summary",
        {},
    ) or {}
    holdout_result = getattr(
        state,
        "holdout_result",
        {},
    ) or {}
    leakage_warnings = getattr(
        state,
        "leakage_warnings",
        [],
    ) or []

    rows = int(profile.get("rows", 0) or 0)

    findings: list[dict[str, Any]] = []

    non_dummy_models = _get_non_dummy_models(leaderboard)
    best_useful_model = _get_best_useful_model(leaderboard)

    selected_model_id = (
        best_useful_model.get("model_id")
        if best_useful_model
        else None
    )

    # 1. Dataset-size evidence
    if rows < 100:
        _add_finding(
            findings=findings,
            severity="high",
            issue="Dataset too small for reliable model selection",
            evidence=(
                f"The experiment used only {rows} rows. This may be enough to "
                "test the pipeline, but not enough to confidently select a model."
            ),
            recommendation=(
                "Use a larger dataset before making a real modelling decision. "
                "Treat the current results as a technical workflow test only."
            ),
        )

    elif rows < 500:
        _add_finding(
            findings=findings,
            severity="medium",
            issue="Dataset is relatively small",
            evidence=f"The experiment used {rows} rows.",
            recommendation=(
                "Interpret model differences carefully and collect more "
                "representative data where possible."
            ),
        )

    # 2. Dummy-baseline evidence
    recommendation_status = comparison_summary.get(
        "recommendation_status"
    )

    if recommendation_status == "no_model_clearly_beats_dummy":
        _add_finding(
            findings=findings,
            severity="high",
            issue="No useful model clearly beats the dummy baseline",
            evidence=(
                "The comparison summary reports that no non-dummy model clearly "
                "beats the dummy baseline on the primary metric."
            ),
            recommendation=(
                "Do not recommend a final model. Review the data, features, "
                "target definition, and sample size before rerunning the workflow."
            ),
        )

    best_overall_model_id = comparison_summary.get(
        "best_overall_model_id"
    )

    if (
        best_overall_model_id
        and "dummy" in str(best_overall_model_id).lower()
    ):
        _add_finding(
            findings=findings,
            severity="high",
            issue="Best ranked model is the dummy baseline",
            evidence=(
                "The best overall model is listed as "
                f"{comparison_summary.get('best_overall_display_name')}."
            ),
            recommendation=(
                "A dummy baseline must not be treated as a useful final model. "
                "A real candidate must beat it clearly."
            ),
        )

    # 3. Planned metric coverage
    unsupported_metrics = training_summary.get(
        "unsupported_metrics_in_first_trainer",
        [],
    )

    if unsupported_metrics:
        _add_finding(
            findings=findings,
            severity="medium",
            issue="Some planned metrics are not implemented",
            evidence=(
                "The trainer did not implement: "
                + ", ".join(unsupported_metrics)
                + "."
            ),
            recommendation=(
                "Add the missing metrics before making a stronger recommendation."
            ),
        )

    # 4. Selected-candidate reliability
    if best_useful_model:
        selected_reliability = best_useful_model.get(
            "reliability_level"
        )

        if selected_reliability == "low":
            _add_finding(
                findings=findings,
                severity="high",
                issue="Selected candidate has low reliability",
                evidence=(
                    f"The selected candidate is "
                    f"{best_useful_model.get('display_name')}, but its "
                    "reliability level is low."
                ),
                recommendation=(
                    "Do not recommend or tune this model until the experiment "
                    "reliability improves."
                ),
            )

        elif selected_reliability == "medium":
            _add_finding(
                findings=findings,
                severity="medium",
                issue="Selected candidate has medium reliability",
                evidence=(
                    f"The selected candidate is "
                    f"{best_useful_model.get('display_name')}, and its "
                    "reliability level is medium."
                ),
                recommendation=(
                    "Treat the model as tentative and investigate its reliability "
                    "flags before persistence or deployment."
                ),
            )

        _add_selected_candidate_flags(
            findings=findings,
            selected_candidate=best_useful_model,
        )

    # 5. Warnings from alternative models
    _add_non_selected_model_findings(
        findings=findings,
        non_dummy_models=non_dummy_models,
        selected_model_id=selected_model_id,
    )

    # 6. Target leakage evidence
    _add_leakage_findings(
        findings=findings,
        leakage_warnings=leakage_warnings,
    )

    # 7. Training completion evidence
    models_failed = int(
        training_summary.get("models_failed", 0) or 0
    )

    if models_failed > 0:
        _add_finding(
            findings=findings,
            severity="medium",
            issue="Some models failed during training",
            evidence=(
                f"{models_failed} model(s) failed during baseline training."
            ),
            recommendation=(
                "Inspect the model errors before trusting the leaderboard."
            ),
        )

    # 8. Final untouched holdout evidence
    _add_holdout_findings(
        findings=findings,
        validation_summary=validation_summary,
        holdout_result=holdout_result,
        rows=rows,
    )

    high_count = _count_findings_by_severity(
        findings,
        "high",
    )
    medium_count = _count_findings_by_severity(
        findings,
        "medium",
    )

    # 9. Final recommendation decision
    if best_useful_model is None:
        selected_candidate = None
        overall_reliability = "low"
        recommendation_decision = "do_not_recommend_model"
        can_proceed_to_tuning = False

    else:
        selected_candidate = {
            "model_id": best_useful_model.get("model_id"),
            "display_name": best_useful_model.get("display_name"),
            "primary_metric": best_useful_model.get("primary_metric"),
            "primary_score": best_useful_model.get("primary_score"),
            "reliability_level": best_useful_model.get(
                "reliability_level"
            ),
        }

        if high_count > 0:
            overall_reliability = "low"
            recommendation_decision = "do_not_recommend_model"
            can_proceed_to_tuning = False

        elif medium_count > 0:
            overall_reliability = "medium"
            recommendation_decision = "recommend_with_caution"
            can_proceed_to_tuning = True

        else:
            overall_reliability = "high"
            recommendation_decision = "recommend_candidate_model"
            can_proceed_to_tuning = True

    # 10. Recommended next actions
    if recommendation_decision == "do_not_recommend_model":
        next_actions = [
            "Do not select or persist a final model from this run.",
            "Resolve the high-severity critic findings.",
            "Review the dataset, target, features, and validation evidence.",
            "Rerun the workflow after improving the evidence.",
        ]

    elif recommendation_decision == "recommend_with_caution":
        next_actions = [
            "Treat the selected useful model as a tentative candidate.",
            "Review the candidate-specific medium-severity findings.",
            _tuning_follow_up_action(state),
            "Do not treat the current result as deployment approval.",
        ]

    else:
        next_actions = [
            "Proceed with the selected candidate as the current recommended model.",
            "Save the critic-approved fitted pipeline and its metadata.",
            "Review feature importance and domain suitability.",
            _tuning_follow_up_action(state),
        ]

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "overall_reliability": overall_reliability,
        "recommendation_decision": recommendation_decision,
        "can_proceed_to_tuning": can_proceed_to_tuning,
        "selected_candidate": selected_candidate,
        "finding_counts": {
            "high": high_count,
            "medium": medium_count,
            "low": _count_findings_by_severity(
                findings,
                "low",
            ),
            "info": _count_findings_by_severity(
                findings,
                "info",
            ),
        },
        "findings": findings,
        "next_actions": next_actions,
    }


def critic_agent(
    state: ExperimentState,
) -> dict[str, Any]:
    """
    Review the completed model comparison and holdout evaluation evidence.
    """

    tool_name = "critic_agent"
    state.iteration += 1

    _add_tool_event(
        state=state,
        tool_name=tool_name,
        status="started",
        message="Started reliability critique.",
    )

    try:
        if state.status != "holdout_evaluation_completed":
            raise ValueError(
                "Cannot run the reliability critic before the final holdout "
                "evaluation check is completed."
            )

        if not state.leaderboard:
            raise ValueError(
                "No leaderboard found in experiment state."
            )

        if not state.comparison_summary:
            raise ValueError(
                "No comparison summary found in experiment state."
            )

        critic_report = create_critic_report(state)

        state.critic_report = critic_report
        state.status = "reliability_critique_completed"
        state.completed_steps.append("created_critic_report")

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="completed",
            message=(
                "Reliability critique completed with decision: "
                f"{critic_report['recommendation_decision']}."
            ),
        )

        return {
            "success": True,
            "state": state,
            "critic_report": critic_report,
            "error": None,
        }

    except Exception as error:
        state.status = "reliability_critique_failed"
        state.warnings.append(str(error))
        state.completed_steps.append("failed_critic_report")

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="failed",
            message=str(error),
        )

        return {
            "success": False,
            "state": state,
            "critic_report": {},
            "error": str(error),
        }