from datetime import datetime
from typing import Any, Callable

from src.agents.critic_agent import critic_agent
from src.agents.planner_agent import planner_agent
from src.state import ExperimentState
from src.tools.inspection_tool import inspect_dataset_tool
from src.tools.model_comparison_tool import model_comparison_tool
from src.tools.model_registry_tool import model_registry_tool
from src.tools.preprocessing_pipeline_tool import preprocessing_pipeline_tool
from src.tools.preprocessing_plan_tool import preprocessing_plan_tool
from src.tools.report_tool import final_report_tool
from src.tools.training_tool import baseline_training_tool
from src.tools.model_persistence_tool import model_persistence_tool


TERMINAL_STATUSES = {
    "final_report_created",
    "preprocessing_config_created_with_pending_approvals",
    "dataset_inspection_failed",
    "experiment_planning_failed",
    "preprocessing_config_failed",
    "preprocessing_pipeline_failed",
    "model_registry_failed",
    "baseline_training_failed",
    "model_comparison_failed",
    "reliability_critique_failed",
    "final_report_failed",
    "model_persistence_failed",
}


def _add_orchestrator_event(
    state: ExperimentState,
    status: str,
    message: str,
) -> None:
    """
    Add a factual orchestrator event to the tool history.
    """

    state.tool_history.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tool_name": "orchestrator_agent",
            "status": status,
            "message": message,
        }
    )


def _get_next_action(state: ExperimentState) -> str:
    """
    Decide the next approved action based on the current state.

    This is the main rule-based orchestration policy.
    """

    if state.status == "created":
        return "inspect_dataset"

    if state.status == "dataset_inspection_completed":
        return "create_experiment_plan"

    if state.status == "experiment_plan_created":
        return "create_preprocessing_config"

    if state.status == "preprocessing_config_created":
        return "create_preprocessing_pipeline"

    if state.status == "preprocessing_pipeline_created":
        return "create_model_registry"

    if state.status == "model_registry_created":
        return "train_baseline_models"

    if state.status == "baseline_training_completed":
        return "compare_models"

    if state.status == "model_comparison_completed":
        return "run_reliability_critic"

    if state.status == "reliability_critique_completed":
        return "check_model_persistence"

    if state.status == "model_persistence_checked":
        return "create_final_report"

    if state.status in TERMINAL_STATUSES:
        return "stop"

    return "unknown"


def _run_action(
    state: ExperimentState,
    action: str,
    approve_drop_id_columns: bool,
) -> dict[str, Any]:
    """
    Run one approved action.
    """

    if action == "inspect_dataset":
        return inspect_dataset_tool(state)

    if action == "create_experiment_plan":
        return planner_agent(state)

    if action == "create_preprocessing_config":
        return preprocessing_plan_tool(
            state=state,
            approve_drop_id_columns=approve_drop_id_columns,
        )

    if action == "create_preprocessing_pipeline":
        return preprocessing_pipeline_tool(state)

    if action == "create_model_registry":
        return model_registry_tool(state)

    if action == "train_baseline_models":
        return baseline_training_tool(state)

    if action == "compare_models":
        return model_comparison_tool(state)

    if action == "run_reliability_critic":
        return critic_agent(state)
    
    if action == "check_model_persistence":
        return model_persistence_tool(state)

    if action == "create_final_report":
        return final_report_tool(state)

    raise ValueError(f"Unknown orchestrator action: {action}")


def run_orchestrator(
    state: ExperimentState,
    approve_drop_id_columns: bool = False,
    max_steps: int = 20,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Run the controlled agentic AutoML workflow.

    The orchestrator repeatedly:
    1. Observes the current state status
    2. Chooses the next approved action
    3. Calls the relevant tool or agent
    4. Updates state
    5. Stops when complete or blocked

    An optional progress callback receives factual start, completion, and
    finish events. The Streamlit UI uses these events to report real workflow
    progress; the command-line interface does not need to provide a callback.
    """

    _add_orchestrator_event(
        state=state,
        status="started",
        message="Started orchestrated AutoML workflow.",
    )

    actions_taken: list[dict[str, Any]] = []

    for step_number in range(1, max_steps + 1):
        next_action = _get_next_action(state)

        actions_taken.append(
            {
                "step_number": step_number,
                "state_status_before": state.status,
                "selected_action": next_action,
            }
        )

        if next_action == "stop":
            _add_orchestrator_event(
                state=state,
                status="stopped",
                message=f"Stopped workflow at terminal status: {state.status}.",
            )

            response = {
                "success": True,
                "state": state,
                "actions_taken": actions_taken,
                "stop_reason": state.status,
                "error": None,
            }
            if progress_callback:
                progress_callback(
                    {
                        "phase": "finished",
                        "step_number": step_number,
                        "action": "stop",
                        "success": True,
                        "state_status": state.status,
                    }
                )
            return response

        if next_action == "unknown":
            message = f"No orchestration rule found for state status: {state.status}"

            state.warnings.append(message)

            _add_orchestrator_event(
                state=state,
                status="failed",
                message=message,
            )

            return {
                "success": False,
                "state": state,
                "actions_taken": actions_taken,
                "stop_reason": "unknown_status",
                "error": message,
            }

        _add_orchestrator_event(
            state=state,
            status="selected_action",
            message=f"Selected next action: {next_action}.",
        )

        if progress_callback:
            progress_callback(
                {
                    "phase": "started",
                    "step_number": step_number,
                    "action": next_action,
                    "state_status": state.status,
                }
            )

        result = _run_action(
            state=state,
            action=next_action,
            approve_drop_id_columns=approve_drop_id_columns,
        )

        state = result["state"]

        actions_taken[-1]["action_success"] = result["success"]
        actions_taken[-1]["state_status_after"] = state.status

        if progress_callback:
            progress_callback(
                {
                    "phase": "completed",
                    "step_number": step_number,
                    "action": next_action,
                    "success": result["success"],
                    "state_status": state.status,
                }
            )

        if not result["success"]:
            _add_orchestrator_event(
                state=state,
                status="failed",
                message=f"Action failed: {next_action}.",
            )

            return {
                "success": False,
                "state": state,
                "actions_taken": actions_taken,
                "stop_reason": state.status,
                "error": result.get("error"),
            }

        if state.status == "preprocessing_config_created_with_pending_approvals":
            _add_orchestrator_event(
                state=state,
                status="paused",
                message=(
                    "Workflow paused because preprocessing configuration has "
                    "pending approvals."
                ),
            )

            return {
                "success": True,
                "state": state,
                "actions_taken": actions_taken,
                "stop_reason": "pending_approvals",
                "error": None,
            }

    message = f"Maximum orchestrator steps reached: {max_steps}"

    state.warnings.append(message)

    _add_orchestrator_event(
        state=state,
        status="stopped",
        message=message,
    )

    return {
        "success": False,
        "state": state,
        "actions_taken": actions_taken,
        "stop_reason": "max_steps_reached",
        "error": message,
    }