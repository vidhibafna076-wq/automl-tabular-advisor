from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)


@dataclass
class TaskInfo:
    """
    Stores information about the detected machine learning task.

    task_type:
        Specific task type:
        - binary_classification
        - multiclass_classification
        - regression

    base_task:
        General task family:
        - classification
        - regression

    reason:
        Human-readable explanation of why this task type was selected.
    """

    task_type: str
    base_task: str
    reason: str
    unique_count: int
    missing_count: int
    unique_values_preview: list[Any]
    target_summary: dict[str, Any]


def _make_classification_summary(y_clean: pd.Series) -> dict[str, Any]:
    """
    Create a class distribution summary for classification targets.

    Example:
    If the target column has:
        Y, Y, Y, N

    This function returns:
        class_counts: Y = 3, N = 1
        class_percentages: Y = 75%, N = 25%
    """

    counts = y_clean.value_counts(dropna=False)
    percentages = (
        y_clean
        .value_counts(dropna=False, normalize=True)
        .mul(100)
        .round(2)
    )

    return {
        "class_counts": {
            str(label): int(count)
            for label, count in counts.items()
        },
        "class_percentages": {
            str(label): float(percentage)
            for label, percentage in percentages.items()
        },
    }


def _make_regression_summary(y_clean: pd.Series) -> dict[str, Any]:
    """
    Create a numeric summary for regression targets.

    This helps us understand the range and spread of a numeric target.
    """

    return {
        "min": float(y_clean.min()),
        "max": float(y_clean.max()),
        "mean": float(y_clean.mean()),
        "median": float(y_clean.median()),
        "std": float(y_clean.std()) if len(y_clean) > 1 else 0.0,
    }


def detect_task_type(y: pd.Series) -> TaskInfo:
    """
    Detect whether the ML task is:
    - binary classification
    - multiclass classification
    - regression

    This function uses simple, explainable rules.
    """

    missing_count = int(y.isna().sum())
    y_clean = y.dropna()

    if y_clean.empty:
        raise ValueError("Target column has no valid non-missing values.")

    unique_values = list(y_clean.unique())
    unique_count = len(unique_values)
    total_count = len(y_clean)
    unique_ratio = unique_count / total_count

    unique_values_preview = [
        str(value) for value in unique_values[:10]
    ]

    # Rule 1: Boolean target is binary classification
    if is_bool_dtype(y_clean):
        return TaskInfo(
            task_type="binary_classification",
            base_task="classification",
            reason=(
                "The target column is boolean, so it is treated as "
                "binary classification."
            ),
            unique_count=unique_count,
            missing_count=missing_count,
            unique_values_preview=unique_values_preview,
            target_summary=_make_classification_summary(y_clean),
        )

    # Rule 2: Non-numeric target is classification
    if not is_numeric_dtype(y_clean):
        if unique_count == 2:
            task_type = "binary_classification"
            reason = (
                f"The target column has 2 unique categories: "
                f"{unique_values_preview}. Therefore, it is treated as "
                "binary classification."
            )
        else:
            task_type = "multiclass_classification"
            reason = (
                f"The target column is categorical with {unique_count} "
                "unique classes. Therefore, it is treated as "
                "multiclass classification."
            )

        return TaskInfo(
            task_type=task_type,
            base_task="classification",
            reason=reason,
            unique_count=unique_count,
            missing_count=missing_count,
            unique_values_preview=unique_values_preview,
            target_summary=_make_classification_summary(y_clean),
        )

    # Rule 3: Numeric target with exactly two values is binary classification
    if unique_count == 2:
        return TaskInfo(
            task_type="binary_classification",
            base_task="classification",
            reason=(
                "The target column is numeric but has only 2 unique values. "
                "Therefore, it is treated as binary classification."
            ),
            unique_count=unique_count,
            missing_count=missing_count,
            unique_values_preview=unique_values_preview,
            target_summary=_make_classification_summary(y_clean),
        )

    # Rule 4: Integer target with few repeated values is probably multiclass classification
    if is_integer_dtype(y_clean) and unique_count <= 20 and unique_ratio <= 0.20:
        return TaskInfo(
            task_type="multiclass_classification",
            base_task="classification",
            reason=(
                f"The target column is integer-based with {unique_count} "
                "unique values that repeat across the dataset. Therefore, "
                "it is treated as multiclass classification."
            ),
            unique_count=unique_count,
            missing_count=missing_count,
            unique_values_preview=unique_values_preview,
            target_summary=_make_classification_summary(y_clean),
        )

    # Rule 5: Otherwise, numeric target is regression
    return TaskInfo(
        task_type="regression",
        base_task="regression",
        reason=(
            f"The target column is numeric with {unique_count} unique values. "
            "Therefore, it is treated as a regression problem."
        ),
        unique_count=unique_count,
        missing_count=missing_count,
        unique_values_preview=unique_values_preview,
        target_summary=_make_regression_summary(y_clean),
    )