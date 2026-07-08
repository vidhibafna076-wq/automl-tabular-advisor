from datetime import datetime
from typing import Any

from src.data_loader import load_dataset
from src.preprocessing import (
    build_preprocessor,
    create_preprocessing_pipeline_summary,
    prepare_features_and_target,
)
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


def preprocessing_pipeline_tool(state: ExperimentState) -> dict[str, Any]:
    """
    Approved tool for building the actual preprocessing pipeline.

    This tool:
    - reloads the dataset
    - prepares X and y using preprocessing_config
    - builds an unfitted scikit-learn ColumnTransformer
    - stores a JSON-safe summary in state

    It does not fit the preprocessor.
    It does not train models.
    """

    tool_name = "preprocessing_pipeline_tool"
    state.iteration += 1

    _add_tool_event(
        state=state,
        tool_name=tool_name,
        status="started",
        message="Started building preprocessing pipeline.",
    )

    try:
        if state.status != "preprocessing_config_created":
            raise ValueError(
                "Cannot build preprocessing pipeline before preprocessing configuration "
                "is created and all required approvals are resolved."
            )

        if state.pending_approvals:
            raise ValueError(
                "Cannot build preprocessing pipeline while approvals are pending."
            )

        if not state.preprocessing_config:
            raise ValueError(
                "No preprocessing configuration found in experiment state."
            )

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

        summary = create_preprocessing_pipeline_summary(
            preprocessing_config=state.preprocessing_config,
            rows_after_target_cleaning=len(y),
        )

        state.preprocessing_pipeline_summary = summary
        state.status = "preprocessing_pipeline_created"
        state.completed_steps.append("created_preprocessing_pipeline")

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="completed",
            message=(
                "Created unfitted preprocessing pipeline with "
                f"{len(state.preprocessing_config['numerical_features'])} numerical "
                f"feature(s) and {len(state.preprocessing_config['categorical_features'])} "
                "categorical feature(s)."
            ),
        )

        return {
            "success": True,
            "state": state,
            "X": X,
            "y": y,
            "preprocessor": preprocessor,
            "pipeline_summary": summary,
            "error": None,
        }

    except Exception as error:
        state.status = "preprocessing_pipeline_failed"
        state.warnings.append(str(error))
        state.completed_steps.append("failed_preprocessing_pipeline")

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="failed",
            message=str(error),
        )

        return {
            "success": False,
            "state": state,
            "X": None,
            "y": None,
            "preprocessor": None,
            "pipeline_summary": {},
            "error": str(error),
        }