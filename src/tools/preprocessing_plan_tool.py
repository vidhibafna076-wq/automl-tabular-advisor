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


def _get_missing_columns_by_type(
    profile: dict[str, Any],
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, list[str]]:
    """
    Identify which numerical and categorical columns have missing values.
    """

    missing_numeric = []
    missing_categorical = []

    for item in profile.get("missing_values", []):
        column = item["column"]
        missing_count = item["missing_count"]

        if missing_count <= 0:
            continue

        if column in numerical_columns:
            missing_numeric.append(column)

        if column in categorical_columns:
            missing_categorical.append(column)

    return {
        "numerical": missing_numeric,
        "categorical": missing_categorical,
    }


def _choose_validation_strategy(state: ExperimentState) -> dict[str, Any]:
    """
    Choose a validation strategy based on the task type and dataset size.

    This does not run validation yet. It only creates the plan.
    """

    rows = state.profile.get("rows", 0)

    if state.base_task == "classification":
        if rows < 500:
            return {
                "method": "stratified_cross_validation",
                "cv_folds": 3,
                "reason": (
                    "The dataset is small and the task is classification, so "
                    "stratified cross-validation is preferred."
                ),
            }

        return {
            "method": "stratified_train_test_split_plus_cv",
            "test_size": 0.2,
            "cv_folds": 5,
            "reason": (
                "The task is classification, so stratification helps preserve "
                "class distribution across splits."
            ),
        }

    if rows < 500:
        return {
            "method": "k_fold_cross_validation",
            "cv_folds": 3,
            "reason": (
                "The dataset is small, so K-fold cross-validation gives a more "
                "stable regression estimate than a single split."
            ),
        }

    return {
        "method": "train_test_split_plus_k_fold_cv",
        "test_size": 0.2,
        "cv_folds": 5,
        "reason": (
            "The task is regression, so K-fold cross-validation will be used "
            "to estimate model performance."
        ),
    }


def _choose_metric_strategy(state: ExperimentState) -> dict[str, Any]:
    """
    Choose primary and secondary metrics based on task type.
    """

    if state.task_type == "binary_classification":
        return {
            "primary_metric": "f1",
            "secondary_metrics": [
                "accuracy",
                "precision",
                "recall",
                "balanced_accuracy",
                "roc_auc",
                "pr_auc",
            ],
            "reason": (
                "Binary classification should not be judged by accuracy alone. "
                "F1 balances precision and recall. ROC-AUC and PR-AUC help evaluate "
                "ranking quality when probability estimates are available."
            ),
        }

    if state.task_type == "multiclass_classification":
        return {
            "primary_metric": "macro_f1",
            "secondary_metrics": [
                "weighted_f1",
                "accuracy",
                "balanced_accuracy",
            ],
            "reason": (
                "Macro F1 gives each class equal importance in multiclass problems."
            ),
        }

    return {
        "primary_metric": "rmse",
        "secondary_metrics": [
            "mae",
            "r2",
        ],
        "reason": (
            "Regression models should be evaluated using prediction error and "
            "explained variance."
        ),
    }


def create_preprocessing_config(
    state: ExperimentState,
    approve_drop_id_columns: bool = False,
) -> dict[str, Any]:
    """
    Create a machine-readable preprocessing configuration.

    This function does not fit any transformer.
    It only decides what the future preprocessing pipeline should do.
    """

    profile = state.profile

    feature_columns = profile.get("feature_columns", [])
    numerical_columns = profile.get("numerical_columns", [])
    categorical_columns = profile.get("categorical_columns", [])
    constant_columns = profile.get("constant_columns", [])
    possible_id_columns = profile.get("possible_id_columns", [])

    columns_to_drop = []
    drop_reasons = {}
    pending_approvals = []

    # Constant columns are safe to drop automatically.
    for column in constant_columns:
        columns_to_drop.append(column)
        drop_reasons[column] = "constant_column"

    # ID columns require user approval.
    if possible_id_columns:
        if approve_drop_id_columns:
            for column in possible_id_columns:
                columns_to_drop.append(column)
                drop_reasons[column] = "possible_identifier_user_approved"

            if "approved_drop_id_columns" not in state.approved_actions:
                state.approved_actions.append("approved_drop_id_columns")

        else:
            pending_approvals.append(
                {
                    "approval_id": "drop_id_columns",
                    "title": "Approve dropping possible ID columns",
                    "columns": possible_id_columns,
                    "reason": (
                        "These columns look like identifiers. They usually do not "
                        "generalise to new data, but the user should approve removal."
                    ),
                }
            )

    # Remove duplicates while preserving order.
    columns_to_drop = list(dict.fromkeys(columns_to_drop))

    final_feature_columns = [
        column
        for column in feature_columns
        if column not in columns_to_drop
    ]

    final_numerical_columns = [
        column
        for column in numerical_columns
        if column not in columns_to_drop
    ]

    final_categorical_columns = [
        column
        for column in categorical_columns
        if column not in columns_to_drop
    ]

    missing_by_type = _get_missing_columns_by_type(
        profile=profile,
        numerical_columns=final_numerical_columns,
        categorical_columns=final_categorical_columns,
    )

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target_column": state.target_column,
        "original_feature_columns": feature_columns,
        "final_feature_columns": final_feature_columns,
        "columns_to_drop": columns_to_drop,
        "drop_reasons": drop_reasons,
        "pending_approvals": pending_approvals,
        "numerical_features": final_numerical_columns,
        "categorical_features": final_categorical_columns,
        "missing_value_strategy": {
            "numerical": {
                "strategy": "median",
                "columns_with_missing_values": missing_by_type["numerical"],
            },
            "categorical": {
                "strategy": "most_frequent",
                "columns_with_missing_values": missing_by_type["categorical"],
            },
        },
        "scaling_strategy": {
            "apply_to_numerical": True,
            "scaler": "standard_scaler",
            "reason": (
                "Scaling helps distance-based and linear models. Tree-based models "
                "are less sensitive, but using a consistent pipeline keeps comparison fair."
            ),
        },
        "encoding_strategy": {
            "categorical_encoder": "one_hot_encoder",
            "handle_unknown": "ignore",
            "reason": (
                "One-hot encoding converts categorical values into numerical features. "
                "handle_unknown='ignore' prevents errors when new categories appear later."
            ),
        },
        "validation_strategy": _choose_validation_strategy(state),
        "metric_strategy": _choose_metric_strategy(state),
        "notes": [
            "This configuration does not fit preprocessing on the full dataset.",
            "The actual preprocessing pipeline should be fitted only inside training folds.",
            "This avoids data leakage.",
        ],
    }

    return config


def preprocessing_plan_tool(
    state: ExperimentState,
    approve_drop_id_columns: bool = False,
) -> dict[str, Any]:
    """
    Approved tool for creating a preprocessing configuration.

    This tool reads the experiment plan and profile, then creates a structured
    preprocessing configuration for the next ML stage.
    """

    tool_name = "preprocessing_plan_tool"
    state.iteration += 1

    _add_tool_event(
        state=state,
        tool_name=tool_name,
        status="started",
        message="Started preprocessing configuration planning.",
    )

    try:
        if state.status != "experiment_plan_created":
            raise ValueError(
                "Cannot create preprocessing configuration before the experiment plan is created."
            )

        config = create_preprocessing_config(
            state=state,
            approve_drop_id_columns=approve_drop_id_columns,
        )

        state.preprocessing_config = config
        state.pending_approvals = config["pending_approvals"]
        state.completed_steps.append("created_preprocessing_config")

        if state.pending_approvals:
            state.status = "preprocessing_config_created_with_pending_approvals"

            _add_tool_event(
                state=state,
                tool_name=tool_name,
                status="completed_with_pending_approvals",
                message=(
                    "Created preprocessing configuration, but some actions still "
                    "require user approval."
                ),
            )

        else:
            state.status = "preprocessing_config_created"

            _add_tool_event(
                state=state,
                tool_name=tool_name,
                status="completed",
                message="Created preprocessing configuration successfully.",
            )

        return {
            "success": True,
            "state": state,
            "preprocessing_config": config,
            "pending_approvals": state.pending_approvals,
            "error": None,
        }

    except Exception as error:
        state.status = "preprocessing_config_failed"
        state.warnings.append(str(error))
        state.completed_steps.append("failed_preprocessing_config")

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="failed",
            message=str(error),
        )

        return {
            "success": False,
            "state": state,
            "preprocessing_config": {},
            "pending_approvals": [],
            "error": str(error),
        }