from datetime import datetime
from typing import Any

from src.model_registry import get_model_registry_summary
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


def model_registry_tool(state: ExperimentState) -> dict[str, Any]:
    """
    Approved tool for creating the model registry summary.

    This tool selects the approved model catalogue based on the detected task.

    It does not train models.
    It only records which models are available for the next training step.
    """

    tool_name = "model_registry_tool"
    state.iteration += 1

    _add_tool_event(
        state=state,
        tool_name=tool_name,
        status="started",
        message="Started model registry selection.",
    )

    try:
        if state.status != "preprocessing_pipeline_created":
            raise ValueError(
                "Cannot create model registry before preprocessing pipeline is created."
            )

        if state.base_task not in ["classification", "regression"]:
            raise ValueError(
                f"Unsupported base task: {state.base_task}"
            )

        registry_summary = get_model_registry_summary(
            base_task=state.base_task,
        )

        state.model_registry_summary = registry_summary
        state.status = "model_registry_created"
        state.completed_steps.append("created_model_registry")

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="completed",
            message=(
                f"Created {state.base_task} model registry with "
                f"{len(registry_summary)} candidate model(s)."
            ),
        )

        return {
            "success": True,
            "state": state,
            "model_registry_summary": registry_summary,
            "error": None,
        }

    except Exception as error:
        state.status = "model_registry_failed"
        state.warnings.append(str(error))
        state.completed_steps.append("failed_model_registry")

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="failed",
            message=str(error),
        )

        return {
            "success": False,
            "state": state,
            "model_registry_summary": [],
            "error": str(error),
        }