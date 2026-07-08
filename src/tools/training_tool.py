from datetime import datetime
from typing import Any

from src.data_loader import load_dataset
from src.model_registry import get_candidate_models
from src.preprocessing import build_preprocessor, prepare_features_and_target
from src.state import ExperimentState
from src.training import train_candidate_models


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


def baseline_training_tool(state: ExperimentState) -> dict[str, Any]:
    """
    Approved tool for training baseline candidate models.

    This tool:
    - reloads the dataset
    - prepares X and y
    - rebuilds the preprocessor
    - gets approved candidate models
    - evaluates each model using cross-validation
    - saves model results to state

    It does not tune models yet.
    It does not select the final best model yet.
    """

    tool_name = "baseline_training_tool"
    state.iteration += 1

    _add_tool_event(
        state=state,
        tool_name=tool_name,
        status="started",
        message="Started baseline model training.",
    )

    try:
        if state.status != "model_registry_created":
            raise ValueError(
                "Cannot train baseline models before the model registry is created."
            )

        if not state.preprocessing_config:
            raise ValueError("No preprocessing configuration found.")

        if not state.model_registry_summary:
            raise ValueError("No model registry summary found.")

        dataset = load_dataset(
            file_path=state.dataset_path,
            target_column=state.target_column,
        )

        X, y = prepare_features_and_target(
            df=dataset.df,
            preprocessing_config=state.preprocessing_config,
        )

        preprocessor = build_preprocessor(
            preprocessing_config=state.preprocessing_config,
        )

        candidate_models = get_candidate_models(
            base_task=state.base_task,
        )

        model_results, training_summary = train_candidate_models(
            X=X,
            y=y,
            preprocessor=preprocessor,
            candidate_models=candidate_models,
            task_type=state.task_type,
            base_task=state.base_task,
            preprocessing_config=state.preprocessing_config,
        )

        state.model_results = model_results
        state.training_summary = training_summary
        state.status = "baseline_training_completed"
        state.completed_steps.append("trained_baseline_models")

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="completed",
            message=(
                f"Baseline training completed. "
                f"{training_summary['models_completed']} model(s) completed and "
                f"{training_summary['models_failed']} model(s) failed."
            ),
        )

        return {
            "success": True,
            "state": state,
            "model_results": model_results,
            "training_summary": training_summary,
            "error": None,
        }

    except Exception as error:
        state.status = "baseline_training_failed"
        state.warnings.append(str(error))
        state.completed_steps.append("failed_baseline_training")

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="failed",
            message=str(error),
        )

        return {
            "success": False,
            "state": state,
            "model_results": [],
            "training_summary": {},
            "error": str(error),
        }