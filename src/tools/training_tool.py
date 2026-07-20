from datetime import datetime
from typing import Any

from src.data_loader import load_dataset
from src.model_registry import get_candidate_models
from src.preprocessing import build_preprocessor, prepare_features_and_target
from src.state import ExperimentState
from src.training import train_candidate_models
from src.validation import create_validation_split


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


def baseline_training_tool(
    state: ExperimentState,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train and evaluate baseline candidate models.

    For sufficiently large datasets, this tool first reserves a final untouched
    holdout set. Cross-validation is then performed only on the remaining
    training portion.

    For small datasets, all usable rows remain available for cross-validation.
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
            raise ValueError(
                "No preprocessing configuration was found in the experiment state."
            )

        dataset = load_dataset(
            file_path=state.dataset_path,
            target_column=state.target_column,
        )

        X, y = prepare_features_and_target(
            df=dataset.df,
            preprocessing_config=state.preprocessing_config,
        )

        validation_split = create_validation_split(
            X=X,
            y=y,
            base_task=state.base_task,
            random_state=random_state,
        )

        state.validation_summary = validation_split.summary
        state.holdout_result = {}

        preprocessor = build_preprocessor(
            preprocessing_config=state.preprocessing_config,
        )

        candidate_models = get_candidate_models(
            base_task=state.base_task,
            random_state=random_state,
        )

        model_results, training_summary = train_candidate_models(
            X=validation_split.X_train,
            y=validation_split.y_train,
            preprocessor=preprocessor,
            candidate_models=candidate_models,
            task_type=state.task_type,
            base_task=state.base_task,
            preprocessing_config=state.preprocessing_config,
            random_state=random_state,
        )

        training_summary["validation_strategy_summary"] = (
            validation_split.summary
        )
        training_summary["rows_used_for_cross_validation"] = int(
            len(validation_split.y_train)
        )
        training_summary["holdout_rows_reserved"] = (
            int(len(validation_split.y_holdout))
            if validation_split.y_holdout is not None
            else 0
        )

        state.model_results = model_results
        state.training_summary = training_summary
        state.status = "baseline_training_completed"
        state.completed_steps.append("trained_baseline_models")

        holdout_message = (
            f" A final holdout of "
            f"{validation_split.summary['holdout_rows']} row(s) was reserved."
            if validation_split.summary["holdout_used"]
            else " No final holdout was reserved."
        )

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="completed",
            message=(
                "Baseline training completed. "
                f"{training_summary['models_completed']} model(s) completed and "
                f"{training_summary['models_failed']} model(s) failed. "
                f"Cross-validation used "
                f"{validation_split.summary['cross_validation_rows']} row(s)."
                f"{holdout_message}"
            ),
        )

        return {
            "success": True,
            "state": state,
            "model_results": model_results,
            "training_summary": training_summary,
            "validation_summary": validation_split.summary,
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
            "validation_summary": {},
            "error": str(error),
        }