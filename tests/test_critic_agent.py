from src.agents.critic_agent import critic_agent
from src.state import ExperimentState


def test_reliable_candidate_is_recommended() -> None:
    """
    A useful model with high reliability and successful untouched holdout
    evidence should be recommended.

    Informational findings are allowed and must not downgrade the final
    recommendation.
    """

    state = ExperimentState(
        dataset_path="reliable_candidate.csv",
        target_column="Target",
        user_objective="Test reliable candidate recommendation",
    )

    state.status = "holdout_evaluation_completed"
    state.task_type = "binary_classification"
    state.base_task = "classification"

    # At least 500 rows avoids dataset-size warnings.
    state.profile = {
        "rows": 500,
        "columns": 6,
    }

    state.leaderboard = [
        {
            "rank": 1,
            "model_id": "logistic_regression",
            "display_name": "Logistic Regression",
            "primary_metric": "f1",
            "primary_score": 0.82,
            "is_dummy_baseline": False,
            "improvement_over_dummy": {
                "beats_dummy": True,
                "absolute_improvement": 0.22,
            },
            "reliability_level": "high",
            "reliability_flags": [],
            "status": "completed",
        },
        {
            "rank": 2,
            "model_id": "dummy_classifier",
            "display_name": "Dummy Classifier",
            "primary_metric": "f1",
            "primary_score": 0.60,
            "is_dummy_baseline": True,
            "improvement_over_dummy": {
                "beats_dummy": False,
                "absolute_improvement": 0.0,
            },
            "reliability_level": "baseline",
            "reliability_flags": [],
            "status": "completed",
        },
    ]

    state.comparison_summary = {
        "recommendation_status": "useful_model_selected",
        "best_overall_model_id": "logistic_regression",
        "best_overall_display_name": "Logistic Regression",
        "best_useful_model_id": "logistic_regression",
        "best_useful_display_name": "Logistic Regression",
        "primary_metric": "f1",
    }

    state.training_summary = {
        "models_completed": 4,
        "models_failed": 0,
        "unsupported_metrics_in_first_trainer": [],
        "primary_metric": "f1",
    }

    state.validation_summary = {
        "holdout_used": True,
        "training_rows": 400,
        "holdout_rows": 100,
    }

    state.holdout_result = {
        "status": "completed",
        "holdout_used": True,
        "model_id": "logistic_regression",
        "display_name": "Logistic Regression",
        "primary_metric": "f1",
        "cv_primary_score": 0.82,
        "holdout_primary_score": 0.81,
        "passes_holdout_guardrail": True,
        "holdout_rows": 100,
        "guardrail_reason": (
            "Holdout f1 was 0.8100. The minimum permitted value "
            "under the current guardrail was 0.7200."
        ),
    }

    state.tuning_result = {
        "status": "completed",
        "accepted_for_final_evaluation": True,
    }

    state.leakage_warnings = []

    result = critic_agent(state)

    assert result["success"] is True
    assert result["error"] is None

    updated_state = result["state"]
    critic_report = result["critic_report"]

    # State transition
    assert (
        updated_state.status
        == "reliability_critique_completed"
    )

    assert (
        "created_critic_report"
        in updated_state.completed_steps
    )

    # Final critic decision
    assert (
        critic_report["overall_reliability"]
        == "high"
    )

    assert (
        critic_report["recommendation_decision"]
        == "recommend_candidate_model"
    )

    assert (
        critic_report["can_proceed_to_tuning"]
        is True
    )

    # Selected candidate
    selected_candidate = critic_report[
        "selected_candidate"
    ]

    assert selected_candidate == {
        "model_id": "logistic_regression",
        "display_name": "Logistic Regression",
        "primary_metric": "f1",
        "primary_score": 0.82,
        "reliability_level": "high",
    }

    # No blocking findings
    finding_counts = critic_report["finding_counts"]

    assert finding_counts["high"] == 0
    assert finding_counts["medium"] == 0

    # Passing holdout evidence is informational.
    assert finding_counts["info"] >= 1

    holdout_findings = [
        finding
        for finding in critic_report["findings"]
        if finding.get("issue")
        == "Final holdout evidence passed"
    ]

    assert len(holdout_findings) == 1
    assert holdout_findings[0]["severity"] == "info"

    assert (
        "100 untouched row(s)"
        in holdout_findings[0]["evidence"]
    )

    # Next actions match an approved candidate.
    assert (
        "Proceed with the selected candidate as the current "
        "recommended model."
        in critic_report["next_actions"]
    )

    assert (
        "Save the critic-approved fitted pipeline and its metadata."
        in critic_report["next_actions"]
    )

    assert (
        "Review the accepted guarded-tuning evidence and do not reuse "
        "the final holdout for further model selection."
        in critic_report["next_actions"]
    )

    assert not any(
        "consider limited tuning" in action.lower()
        for action in critic_report["next_actions"]
    )

    # State stores the same report.
    assert updated_state.critic_report == critic_report

    # Tool history
    started_events = [
        event
        for event in updated_state.tool_history
        if (
            event.get("tool_name") == "critic_agent"
            and event.get("status") == "started"
        )
    ]

    completed_events = [
        event
        for event in updated_state.tool_history
        if (
            event.get("tool_name") == "critic_agent"
            and event.get("status") == "completed"
        )
    ]

    assert started_events
    assert completed_events

    assert (
        "recommend_candidate_model"
        in completed_events[-1]["message"]
    )

def test_candidate_is_recommended_with_caution_for_medium_concern() -> None:
    """
    A useful candidate with no high-severity concerns should be recommended
    with caution when at least one medium-severity finding exists.

    In this test, the dataset contains 300 rows, which triggers the critic's
    medium-severity small-dataset finding.
    """

    state = ExperimentState(
        dataset_path="cautious_candidate.csv",
        target_column="Target",
        user_objective="Test cautious candidate recommendation",
    )

    state.status = "holdout_evaluation_completed"
    state.task_type = "regression"
    state.base_task = "regression"

    # Between 100 and 499 rows triggers a medium dataset-size finding.
    state.profile = {
        "rows": 300,
        "columns": 7,
    }

    state.leaderboard = [
        {
            "rank": 1,
            "model_id": "ridge_regression",
            "display_name": "Ridge Regression",
            "primary_metric": "rmse",
            "primary_score": 100.0,
            "is_dummy_baseline": False,
            "improvement_over_dummy": {
                "beats_dummy": True,
                "absolute_improvement": 35.0,
            },
            "reliability_level": "high",
            "reliability_flags": [],
            "status": "completed",
        },
        {
            "rank": 2,
            "model_id": "dummy_regressor",
            "display_name": "Dummy Regressor",
            "primary_metric": "rmse",
            "primary_score": 135.0,
            "is_dummy_baseline": True,
            "improvement_over_dummy": {
                "beats_dummy": False,
                "absolute_improvement": 0.0,
            },
            "reliability_level": "baseline",
            "reliability_flags": [],
            "status": "completed",
        },
    ]

    state.comparison_summary = {
        "recommendation_status": "useful_model_selected",
        "best_overall_model_id": "ridge_regression",
        "best_overall_display_name": "Ridge Regression",
        "best_useful_model_id": "ridge_regression",
        "best_useful_display_name": "Ridge Regression",
        "primary_metric": "rmse",
    }

    state.training_summary = {
        "models_completed": 4,
        "models_failed": 0,
        "unsupported_metrics_in_first_trainer": [],
        "primary_metric": "rmse",
    }

    state.validation_summary = {
        "holdout_used": True,
        "training_rows": 240,
        "holdout_rows": 60,
    }

    state.holdout_result = {
        "status": "completed",
        "holdout_used": True,
        "model_id": "ridge_regression",
        "display_name": "Ridge Regression",
        "primary_metric": "rmse",
        "cv_primary_score": 100.0,
        "holdout_primary_score": 110.0,
        "passes_holdout_guardrail": True,
        "holdout_rows": 60,
        "guardrail_reason": (
            "Holdout rmse was 110.0000. The maximum permitted value "
            "under the current guardrail was 120.0000."
        ),
    }

    state.leakage_warnings = []

    result = critic_agent(state)

    assert result["success"] is True
    assert result["error"] is None

    updated_state = result["state"]
    critic_report = result["critic_report"]

    # State transition
    assert (
        updated_state.status
        == "reliability_critique_completed"
    )

    assert (
        "created_critic_report"
        in updated_state.completed_steps
    )

    # Final critic decision
    assert (
        critic_report["overall_reliability"]
        == "medium"
    )

    assert (
        critic_report["recommendation_decision"]
        == "recommend_with_caution"
    )

    assert (
        critic_report["can_proceed_to_tuning"]
        is True
    )

    # Selected candidate remains available.
    assert critic_report["selected_candidate"] == {
        "model_id": "ridge_regression",
        "display_name": "Ridge Regression",
        "primary_metric": "rmse",
        "primary_score": 100.0,
        "reliability_level": "high",
    }

    finding_counts = critic_report["finding_counts"]

    # No blocking high-severity concern exists.
    assert finding_counts["high"] == 0

    # The 300-row dataset creates one medium concern.
    assert finding_counts["medium"] == 1

    dataset_findings = [
        finding
        for finding in critic_report["findings"]
        if finding.get("issue")
        == "Dataset is relatively small"
    ]

    assert len(dataset_findings) == 1
    assert dataset_findings[0]["severity"] == "medium"

    assert (
        "300 rows"
        in dataset_findings[0]["evidence"]
    )

    # Passing holdout evidence remains informational.
    holdout_findings = [
        finding
        for finding in critic_report["findings"]
        if finding.get("issue")
        == "Final holdout evidence passed"
    ]

    assert len(holdout_findings) == 1
    assert holdout_findings[0]["severity"] == "info"

    # Cautious next actions
    assert (
        "Treat the selected useful model as a tentative candidate."
        in critic_report["next_actions"]
    )

    assert (
        "Do not treat the current result as deployment approval."
        in critic_report["next_actions"]
    )

    # State stores the same report.
    assert updated_state.critic_report == critic_report

    # Tool history
    completed_events = [
        event
        for event in updated_state.tool_history
        if (
            event.get("tool_name") == "critic_agent"
            and event.get("status") == "completed"
        )
    ]

    assert completed_events

    assert (
        "recommend_with_caution"
        in completed_events[-1]["message"]
    )

def test_candidate_is_not_recommended_when_holdout_guardrail_fails() -> None:
    """
    A useful model must not be recommended when its final untouched holdout
    performance fails the reliability guardrail.

    The candidate remains visible in the report for diagnosis, but the final
    recommendation must be do_not_recommend_model.
    """

    state = ExperimentState(
        dataset_path="failed_holdout_candidate.csv",
        target_column="Target",
        user_objective="Test rejection after holdout failure",
    )

    state.status = "holdout_evaluation_completed"
    state.task_type = "binary_classification"
    state.base_task = "classification"

    # At least 500 rows avoids unrelated dataset-size findings.
    state.profile = {
        "rows": 500,
        "columns": 6,
    }

    state.leaderboard = [
        {
            "rank": 1,
            "model_id": "logistic_regression",
            "display_name": "Logistic Regression",
            "primary_metric": "f1",
            "primary_score": 0.90,
            "is_dummy_baseline": False,
            "improvement_over_dummy": {
                "beats_dummy": True,
                "absolute_improvement": 0.30,
            },
            "reliability_level": "high",
            "reliability_flags": [],
            "status": "completed",
        },
        {
            "rank": 2,
            "model_id": "dummy_classifier",
            "display_name": "Dummy Classifier",
            "primary_metric": "f1",
            "primary_score": 0.60,
            "is_dummy_baseline": True,
            "improvement_over_dummy": {
                "beats_dummy": False,
                "absolute_improvement": 0.0,
            },
            "reliability_level": "baseline",
            "reliability_flags": [],
            "status": "completed",
        },
    ]

    state.comparison_summary = {
        "recommendation_status": "useful_model_selected",
        "best_overall_model_id": "logistic_regression",
        "best_overall_display_name": "Logistic Regression",
        "best_useful_model_id": "logistic_regression",
        "best_useful_display_name": "Logistic Regression",
        "primary_metric": "f1",
    }

    state.training_summary = {
        "models_completed": 4,
        "models_failed": 0,
        "unsupported_metrics_in_first_trainer": [],
        "primary_metric": "f1",
    }

    state.validation_summary = {
        "holdout_used": True,
        "training_rows": 400,
        "holdout_rows": 100,
    }

    state.holdout_result = {
        "status": "completed",
        "holdout_used": True,
        "model_id": "logistic_regression",
        "display_name": "Logistic Regression",
        "primary_metric": "f1",
        "cv_primary_score": 0.90,
        "holdout_primary_score": 0.55,
        "passes_holdout_guardrail": False,
        "holdout_rows": 100,
        "guardrail_reason": (
            "Holdout f1 was 0.5500. The minimum permitted value "
            "under the current guardrail was 0.8000."
        ),
    }

    state.leakage_warnings = []

    result = critic_agent(state)

    assert result["success"] is True
    assert result["error"] is None

    updated_state = result["state"]
    critic_report = result["critic_report"]

    # State transition
    assert (
        updated_state.status
        == "reliability_critique_completed"
    )

    assert (
        "created_critic_report"
        in updated_state.completed_steps
    )

    # Final decision
    assert critic_report["overall_reliability"] == "low"

    assert (
        critic_report["recommendation_decision"]
        == "do_not_recommend_model"
    )

    assert (
        critic_report["can_proceed_to_tuning"]
        is False
    )

    # The candidate remains visible for diagnosis.
    assert critic_report["selected_candidate"] == {
        "model_id": "logistic_regression",
        "display_name": "Logistic Regression",
        "primary_metric": "f1",
        "primary_score": 0.90,
        "reliability_level": "high",
    }

    finding_counts = critic_report["finding_counts"]

    assert finding_counts["high"] == 1
    assert finding_counts["medium"] == 0

    holdout_failure_findings = [
        finding
        for finding in critic_report["findings"]
        if finding.get("issue")
        == "Final holdout guardrail failed"
    ]

    assert len(holdout_failure_findings) == 1

    holdout_finding = holdout_failure_findings[0]

    assert holdout_finding["severity"] == "high"

    assert (
        "Holdout f1 was 0.5500"
        in holdout_finding["evidence"]
    )

    assert (
        "minimum permitted value"
        in holdout_finding["evidence"]
    )

    assert (
        "Do not recommend or persist this candidate"
        in holdout_finding["recommendation"]
    )

    # Rejection-oriented next actions
    assert (
        "Do not select or persist a final model from this run."
        in critic_report["next_actions"]
    )

    assert (
        "Resolve the high-severity critic findings."
        in critic_report["next_actions"]
    )

    # State stores the same report.
    assert updated_state.critic_report == critic_report

    # Tool history
    completed_events = [
        event
        for event in updated_state.tool_history
        if (
            event.get("tool_name") == "critic_agent"
            and event.get("status") == "completed"
        )
    ]

    assert completed_events

    assert (
        "do_not_recommend_model"
        in completed_events[-1]["message"]
    )