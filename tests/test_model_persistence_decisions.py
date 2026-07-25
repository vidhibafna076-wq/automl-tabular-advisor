from src.state import ExperimentState
from src.tools.model_persistence_tool import model_persistence_tool


def make_persistence_state(
    decision: str,
    selected_candidate: dict | None,
) -> ExperimentState:
    """
    Create the minimum valid state required to test a persistence decision.
    """

    state = ExperimentState(
        dataset_path="unused.csv",
        target_column="Target",
        user_objective="Test persistence decision",
    )

    state.status = "reliability_critique_completed"
    state.critic_report = {
        "recommendation_decision": decision,
        "selected_candidate": selected_candidate,
    }

    return state


def test_do_not_recommend_skips_model_persistence() -> None:
    """
    A rejected model must not be saved, but the persistence check itself
    must complete successfully.
    """

    state = make_persistence_state(
        decision="do_not_recommend_model",
        selected_candidate=None,
    )

    result = model_persistence_tool(state)

    assert result["success"] is True
    assert result["error"] is None

    updated_state = result["state"]
    artifact_summary = result["model_artifact_summary"]
    importance_summary = result["feature_importance_summary"]

    assert updated_state.status == "model_persistence_checked"
    assert "checked_model_persistence" in updated_state.completed_steps

    assert artifact_summary["status"] == "skipped"
    assert (
        artifact_summary["critic_decision"]
        == "do_not_recommend_model"
    )
    assert artifact_summary["model_id"] is None
    assert artifact_summary["artifact_path"] is None
    assert artifact_summary["metadata_path"] is None
    assert artifact_summary["tuning_applied"] is False
    assert artifact_summary["tuned_parameters"] == {}

    assert importance_summary["status"] == "skipped"
    assert importance_summary["top_features"] == []

    completed_events = [
        event
        for event in updated_state.tool_history
        if event.get("tool_name") == "model_persistence_tool"
        and event.get("status") == "completed"
    ]

    assert completed_events
    assert "No model was saved" in completed_events[-1]["message"]


def test_recommend_with_caution_skips_model_persistence() -> None:
    """
    A cautious candidate may be reported, but it must not be persisted as
    an approved deployable model.
    """

    selected_candidate = {
        "model_id": "ridge_regression",
        "display_name": "Ridge Regression",
        "primary_metric": "rmse",
        "primary_score": 45678.9,
    }

    state = make_persistence_state(
        decision="recommend_with_caution",
        selected_candidate=selected_candidate,
    )

    result = model_persistence_tool(state)

    assert result["success"] is True
    assert result["error"] is None

    updated_state = result["state"]
    artifact_summary = result["model_artifact_summary"]
    importance_summary = result["feature_importance_summary"]

    assert updated_state.status == "model_persistence_checked"
    assert "checked_model_persistence" in updated_state.completed_steps

    assert artifact_summary["status"] == "skipped"
    assert (
        artifact_summary["critic_decision"]
        == "recommend_with_caution"
    )
    assert artifact_summary["model_id"] == "ridge_regression"
    assert artifact_summary["display_name"] == "Ridge Regression"
    assert artifact_summary["artifact_path"] is None
    assert artifact_summary["metadata_path"] is None
    assert artifact_summary["tuning_applied"] is False
    assert artifact_summary["tuned_parameters"] == {}

    assert importance_summary["status"] == "skipped"
    assert importance_summary["top_features"] == []

    completed_events = [
        event
        for event in updated_state.tool_history
        if event.get("tool_name") == "model_persistence_tool"
        and event.get("status") == "completed"
    ]

    assert completed_events
    assert (
        "recommended the candidate with caution"
        in completed_events[-1]["message"]
    )