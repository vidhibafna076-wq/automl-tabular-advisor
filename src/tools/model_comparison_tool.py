from datetime import datetime
from typing import Any

from src.comparison import create_model_leaderboard
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


def model_comparison_tool(state: ExperimentState) -> dict[str, Any]:
    """
    Approved tool for creating a model leaderboard.

    This tool:
    - ranks trained models
    - compares models against the dummy baseline
    - adds reliability flags
    - stores leaderboard and comparison summary in state

    It does not tune models.
    It does not make a final production recommendation yet.
    """

    tool_name = "model_comparison_tool"
    state.iteration += 1

    _add_tool_event(
        state=state,
        tool_name=tool_name,
        status="started",
        message="Started model comparison and leaderboard creation.",
    )

    try:
        if state.status != "baseline_training_completed":
            raise ValueError(
                "Cannot compare models before baseline training is completed."
            )

        if not state.model_results:
            raise ValueError("No model results found in experiment state.")

        if not state.training_summary:
            raise ValueError("No training summary found in experiment state.")

        leaderboard, comparison_summary = create_model_leaderboard(
            model_results=state.model_results,
            training_summary=state.training_summary,
            profile=state.profile,
        )

        state.leaderboard = leaderboard
        state.comparison_summary = comparison_summary
        state.status = "model_comparison_completed"
        state.completed_steps.append("created_model_leaderboard")

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="completed",
            message=(
                f"Created model leaderboard with {len(leaderboard)} ranked model(s)."
            ),
        )

        return {
            "success": True,
            "state": state,
            "leaderboard": leaderboard,
            "comparison_summary": comparison_summary,
            "error": None,
        }

    except Exception as error:
        state.status = "model_comparison_failed"
        state.warnings.append(str(error))
        state.completed_steps.append("failed_model_comparison")

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="failed",
            message=str(error),
        )

        return {
            "success": False,
            "state": state,
            "leaderboard": [],
            "comparison_summary": {},
            "error": str(error),
        }