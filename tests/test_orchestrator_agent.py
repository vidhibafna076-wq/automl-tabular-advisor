import pytest

import src.agents.orchestrator_agent as orchestrator_module
from src.agents.orchestrator_agent import (
    _get_next_action,
    run_orchestrator,
)
from src.state import ExperimentState


@pytest.mark.parametrize(
    (
        "current_status",
        "expected_action",
    ),
    [
        (
            "created",
            "inspect_dataset",
        ),
        (
            "dataset_inspection_completed",
            "create_experiment_plan",
        ),
        (
            "experiment_plan_created",
            "create_preprocessing_config",
        ),
        (
            "preprocessing_config_created",
            "create_preprocessing_pipeline",
        ),
        (
            "preprocessing_pipeline_created",
            "create_model_registry",
        ),
        (
            "model_registry_created",
            "train_baseline_models",
        ),
        (
            "baseline_training_completed",
            "compare_models",
        ),
        (
            "model_comparison_completed",
            "run_guarded_tuning",
        ),
        (
            "tuning_completed",
            "evaluate_final_holdout",
        ),
        (
            "holdout_evaluation_completed",
            "run_reliability_critic",
        ),
        (
            "reliability_critique_completed",
            "check_model_persistence",
        ),
        (
            "model_persistence_checked",
            "create_final_report",
        ),
        (
            "final_report_created",
            "stop",
        ),
        (
            "dataset_inspection_failed",
            "stop",
        ),
        (
            "tuning_failed",
            "stop",
        ),
        (
            "model_persistence_failed",
            "stop",
        ),
        (
            "unexpected_status",
            "unknown",
        ),
    ],
)
def test_next_action_matches_controlled_workflow(
    current_status: str,
    expected_action: str,
) -> None:
    """
    Every supported experiment status must select exactly one expected
    orchestrator action.

    Failed and completed terminal statuses must stop. Unsupported statuses
    must return unknown rather than executing an arbitrary tool.
    """

    state = ExperimentState(
        dataset_path="unused.csv",
        target_column="Target",
        user_objective="Test orchestrator action selection",
    )

    state.status = current_status

    selected_action = _get_next_action(state)

    assert selected_action == expected_action

def test_terminal_state_stops_without_running_another_tool(
    monkeypatch,
) -> None:
    """
    A workflow already at final_report_created must stop immediately.

    No AutoML tool may be dispatched after the terminal state has been
    reached.
    """

    state = ExperimentState(
        dataset_path="completed_run.csv",
        target_column="Target",
        user_objective="Test terminal orchestrator behaviour",
    )

    state.status = "final_report_created"

    def fail_if_action_runs(
        state,
        action,
        approve_drop_id_columns,
    ):
        raise AssertionError(
            "No workflow action should run after final_report_created."
        )

    monkeypatch.setattr(
        orchestrator_module,
        "_run_action",
        fail_if_action_runs,
    )

    progress_events = []

    result = run_orchestrator(
        state=state,
        approve_drop_id_columns=False,
        max_steps=20,
        progress_callback=progress_events.append,
    )

    assert result["success"] is True
    assert result["error"] is None
    assert result["stop_reason"] == "final_report_created"

    updated_state = result["state"]

    assert updated_state.status == "final_report_created"

    # The orchestrator records one loop iteration whose selected action is stop.
    assert result["actions_taken"] == [
        {
            "step_number": 1,
            "state_status_before": "final_report_created",
            "selected_action": "stop",
        }
    ]

    orchestrator_events = [
        event
        for event in updated_state.tool_history
        if event.get("tool_name") == "orchestrator_agent"
    ]

    assert len(orchestrator_events) == 2

    assert orchestrator_events[0]["status"] == "started"
    assert (
        orchestrator_events[0]["message"]
        == "Started orchestrated AutoML workflow."
    )

    assert orchestrator_events[1]["status"] == "stopped"
    assert (
        orchestrator_events[1]["message"]
        == (
            "Stopped workflow at terminal status: "
            "final_report_created."
        )
    )

    # No action-selection event should be added for the stop action.
    selected_action_events = [
        event
        for event in orchestrator_events
        if event.get("status") == "selected_action"
    ]

    assert selected_action_events == []

    # The progress callback receives only the final completion notification.
    assert progress_events == [
        {
            "phase": "finished",
            "step_number": 1,
            "action": "stop",
            "success": True,
            "state_status": "final_report_created",
        }
    ]

def test_failed_action_stops_workflow_immediately(
    monkeypatch,
) -> None:
    """
    When a workflow action fails, the orchestrator must stop immediately.

    It must preserve:
    - the failed state;
    - the original tool error;
    - the action history;
    - progress callback evidence.

    No later workflow action may run.
    """

    state = ExperimentState(
        dataset_path="failing_dataset.csv",
        target_column="Target",
        user_objective="Test orchestrator failure handling",
    )

    state.status = "created"

    actions_called = []

    def fake_run_action(
        state,
        action,
        approve_drop_id_columns,
    ):
        actions_called.append(action)

        assert action == "inspect_dataset"
        assert approve_drop_id_columns is False

        state.status = "dataset_inspection_failed"
        state.warnings.append(
            "Synthetic dataset inspection failure."
        )

        return {
            "success": False,
            "state": state,
            "error": "Synthetic dataset inspection failure.",
        }

    monkeypatch.setattr(
        orchestrator_module,
        "_run_action",
        fake_run_action,
    )

    progress_events = []

    result = run_orchestrator(
        state=state,
        approve_drop_id_columns=False,
        max_steps=20,
        progress_callback=progress_events.append,
    )

    # Overall orchestrator result
    assert result["success"] is False

    assert (
        result["stop_reason"]
        == "dataset_inspection_failed"
    )

    assert (
        result["error"]
        == "Synthetic dataset inspection failure."
    )

    updated_state = result["state"]

    assert (
        updated_state.status
        == "dataset_inspection_failed"
    )

    assert (
        "Synthetic dataset inspection failure."
        in updated_state.warnings
    )

    # Only the first workflow action may run.
    assert actions_called == [
        "inspect_dataset",
    ]

    # Action history must preserve failure evidence.
    assert result["actions_taken"] == [
        {
            "step_number": 1,
            "state_status_before": "created",
            "selected_action": "inspect_dataset",
            "action_success": False,
            "state_status_after": (
                "dataset_inspection_failed"
            ),
        }
    ]

    # Orchestrator audit history
    orchestrator_events = [
        event
        for event in updated_state.tool_history
        if (
            event.get("tool_name")
            == "orchestrator_agent"
        )
    ]

    assert len(orchestrator_events) == 3

    assert (
        orchestrator_events[0]["status"]
        == "started"
    )

    assert (
        orchestrator_events[1]["status"]
        == "selected_action"
    )

    assert (
        orchestrator_events[1]["message"]
        == "Selected next action: inspect_dataset."
    )

    assert (
        orchestrator_events[2]["status"]
        == "failed"
    )

    assert (
        orchestrator_events[2]["message"]
        == "Action failed: inspect_dataset."
    )

    # Progress callback receives start and failure completion only.
    assert progress_events == [
        {
            "phase": "started",
            "step_number": 1,
            "action": "inspect_dataset",
            "state_status": "created",
        },
        {
            "phase": "completed",
            "step_number": 1,
            "action": "inspect_dataset",
            "success": False,
            "state_status": (
                "dataset_inspection_failed"
            ),
        },
    ]

def test_pending_preprocessing_approval_pauses_workflow(
    monkeypatch,
) -> None:
    """
    When preprocessing identifies a change requiring user approval, the
    orchestrator must pause rather than continue to pipeline construction
    and model training.

    Pending approval is a controlled pause, not a workflow failure.
    """

    state = ExperimentState(
        dataset_path="approval_required.csv",
        target_column="Target",
        user_objective="Test pending approval handling",
    )

    # Begin immediately before preprocessing configuration.
    state.status = "experiment_plan_created"

    actions_called = []

    def fake_run_action(
        state,
        action,
        approve_drop_id_columns,
    ):
        actions_called.append(action)

        assert action == "create_preprocessing_config"
        assert approve_drop_id_columns is False

        state.status = (
            "preprocessing_config_created_with_pending_approvals"
        )

        state.pending_approvals = [
            {
                "approval_type": "drop_possible_id_column",
                "column": "Customer_ID",
                "reason": (
                    "The column appears to be an identifier."
                ),
            }
        ]

        return {
            "success": True,
            "state": state,
            "error": None,
        }

    monkeypatch.setattr(
        orchestrator_module,
        "_run_action",
        fake_run_action,
    )

    progress_events = []

    result = run_orchestrator(
        state=state,
        approve_drop_id_columns=False,
        max_steps=20,
        progress_callback=progress_events.append,
    )

    # A pending approval is a successful controlled pause.
    assert result["success"] is True
    assert result["error"] is None

    assert (
        result["stop_reason"]
        == "pending_approvals"
    )

    updated_state = result["state"]

    assert (
        updated_state.status
        == "preprocessing_config_created_with_pending_approvals"
    )

    # Only preprocessing configuration may run.
    assert actions_called == [
        "create_preprocessing_config",
    ]

    assert result["actions_taken"] == [
        {
            "step_number": 1,
            "state_status_before": (
                "experiment_plan_created"
            ),
            "selected_action": (
                "create_preprocessing_config"
            ),
            "action_success": True,
            "state_status_after": (
                "preprocessing_config_created_with_pending_approvals"
            ),
        }
    ]

    # The pending approval evidence remains in state.
    assert updated_state.pending_approvals == [
        {
            "approval_type": "drop_possible_id_column",
            "column": "Customer_ID",
            "reason": (
                "The column appears to be an identifier."
            ),
        }
    ]

    orchestrator_events = [
        event
        for event in updated_state.tool_history
        if (
            event.get("tool_name")
            == "orchestrator_agent"
        )
    ]

    assert len(orchestrator_events) == 3

    assert (
        orchestrator_events[0]["status"]
        == "started"
    )

    assert (
        orchestrator_events[1]["status"]
        == "selected_action"
    )

    assert (
        orchestrator_events[1]["message"]
        == (
            "Selected next action: "
            "create_preprocessing_config."
        )
    )

    assert (
        orchestrator_events[2]["status"]
        == "paused"
    )

    assert (
        orchestrator_events[2]["message"]
        == (
            "Workflow paused because preprocessing "
            "configuration has pending approvals."
        )
    )

    # The callback records the completed preprocessing step.
    # There is no finished callback because the workflow paused.
    assert progress_events == [
        {
            "phase": "started",
            "step_number": 1,
            "action": "create_preprocessing_config",
            "state_status": "experiment_plan_created",
        },
        {
            "phase": "completed",
            "step_number": 1,
            "action": "create_preprocessing_config",
            "success": True,
            "state_status": (
                "preprocessing_config_created_with_pending_approvals"
            ),
        },
    ]

def test_maximum_step_limit_prevents_infinite_workflow(
    monkeypatch,
) -> None:
    """
    The orchestrator must stop with a clear failure when successful actions
    repeatedly return a non-terminal state.

    This safety limit prevents an orchestration bug from creating an
    infinite workflow loop.
    """

    state = ExperimentState(
        dataset_path="looping_dataset.csv",
        target_column="Target",
        user_objective="Test maximum orchestrator step limit",
    )

    state.status = "created"

    actions_called = []

    def fake_run_action(
        state,
        action,
        approve_drop_id_columns,
    ):
        actions_called.append(action)

        assert action == "inspect_dataset"
        assert approve_drop_id_columns is False

        # Deliberately leave the state unchanged. This simulates a faulty
        # tool that reports success without advancing the workflow.
        state.status = "created"

        return {
            "success": True,
            "state": state,
            "error": None,
        }

    monkeypatch.setattr(
        orchestrator_module,
        "_run_action",
        fake_run_action,
    )

    progress_events = []

    result = run_orchestrator(
        state=state,
        approve_drop_id_columns=False,
        max_steps=3,
        progress_callback=progress_events.append,
    )

    assert result["success"] is False

    assert (
        result["stop_reason"]
        == "max_steps_reached"
    )

    assert (
        result["error"]
        == "Maximum orchestrator steps reached: 3"
    )

    updated_state = result["state"]

    # The faulty action ran exactly three times before the safety stop.
    assert actions_called == [
        "inspect_dataset",
        "inspect_dataset",
        "inspect_dataset",
    ]

    assert updated_state.status == "created"

    assert (
        "Maximum orchestrator steps reached: 3"
        in updated_state.warnings
    )

    # Each iteration records the same attempted transition.
    assert len(result["actions_taken"]) == 3

    for expected_step, action_record in enumerate(
        result["actions_taken"],
        start=1,
    ):
        assert action_record == {
            "step_number": expected_step,
            "state_status_before": "created",
            "selected_action": "inspect_dataset",
            "action_success": True,
            "state_status_after": "created",
        }

    orchestrator_events = [
        event
        for event in updated_state.tool_history
        if (
            event.get("tool_name")
            == "orchestrator_agent"
        )
    ]

    # One started event, three selected-action events, and one stopped event.
    assert len(orchestrator_events) == 5

    assert (
        orchestrator_events[0]["status"]
        == "started"
    )

    selected_events = [
        event
        for event in orchestrator_events
        if event.get("status") == "selected_action"
    ]

    assert len(selected_events) == 3

    assert all(
        event["message"]
        == "Selected next action: inspect_dataset."
        for event in selected_events
    )

    assert (
        orchestrator_events[-1]["status"]
        == "stopped"
    )

    assert (
        orchestrator_events[-1]["message"]
        == "Maximum orchestrator steps reached: 3"
    )

    # Every attempted action produces a started and completed callback.
    assert len(progress_events) == 6

    for index in range(0, 6, 2):
        expected_step = (index // 2) + 1

        assert progress_events[index] == {
            "phase": "started",
            "step_number": expected_step,
            "action": "inspect_dataset",
            "state_status": "created",
        }

        assert progress_events[index + 1] == {
            "phase": "completed",
            "step_number": expected_step,
            "action": "inspect_dataset",
            "success": True,
            "state_status": "created",
        }

    # No finished callback is produced because the workflow failed by
    # reaching its safety limit rather than a terminal success state.
    assert all(
        event.get("phase") != "finished"
        for event in progress_events
    )