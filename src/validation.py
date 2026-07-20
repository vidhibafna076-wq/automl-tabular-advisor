from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class ValidationSplit:
    """
    Stores the datasets used by the validation strategy.

    X_train and y_train are used for model comparison with cross-validation.

    X_holdout and y_holdout are reserved and must not be used during model
    comparison. They are evaluated only after a candidate model is selected.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_holdout: pd.DataFrame | None
    y_holdout: pd.Series | None
    summary: dict[str, Any]


def _cross_validation_only_split(
    X: pd.DataFrame,
    y: pd.Series,
    reason: str,
    random_state: int,
    holdout_fraction: float,
    minimum_rows_for_holdout: int,
) -> ValidationSplit:
    """
    Return the full dataset for cross-validation without creating a holdout.
    """

    summary = {
        "strategy": "cross_validation_only",
        "cross_validation_used": True,
        "holdout_used": False,
        "total_usable_rows": int(len(y)),
        "cross_validation_rows": int(len(y)),
        "holdout_rows": 0,
        "holdout_fraction": 0.0,
        "requested_holdout_fraction": float(holdout_fraction),
        "minimum_rows_for_holdout": int(minimum_rows_for_holdout),
        "random_state": int(random_state),
        "reason": reason,
    }

    return ValidationSplit(
        X_train=X,
        y_train=y,
        X_holdout=None,
        y_holdout=None,
        summary=summary,
    )


def create_validation_split(
    X: pd.DataFrame,
    y: pd.Series,
    base_task: str,
    holdout_fraction: float = 0.20,
    minimum_rows_for_holdout: int = 200,
    minimum_class_count_for_holdout: int = 10,
    random_state: int = 42,
) -> ValidationSplit:
    """
    Decide whether to reserve a final untouched holdout dataset.

    Policy
    ------
    Small datasets:
        Use all rows for cross-validation.

    Larger datasets:
        Reserve a deterministic holdout set and run cross-validation only on
        the remaining training portion.

    Classification:
        Use a stratified split so class proportions are approximately
        preserved.

    Regression:
        Use a standard random split.

    The holdout is not evaluated in this function. It is only reserved.
    """

    if len(X) != len(y):
        raise ValueError(
            "Feature and target row counts do not match during validation setup."
        )

    total_rows = int(len(y))

    if total_rows < 2:
        raise ValueError(
            "At least 2 usable rows are required to create a validation strategy."
        )

    if not 0 < holdout_fraction < 0.5:
        raise ValueError(
            "holdout_fraction must be greater than 0 and less than 0.5."
        )

    if total_rows < minimum_rows_for_holdout:
        return _cross_validation_only_split(
            X=X,
            y=y,
            reason=(
                f"The dataset has {total_rows} usable rows, which is below the "
                f"{minimum_rows_for_holdout}-row threshold for reserving a final "
                "holdout set. All usable rows will be evaluated with "
                "cross-validation."
            ),
            random_state=random_state,
            holdout_fraction=holdout_fraction,
            minimum_rows_for_holdout=minimum_rows_for_holdout,
        )

    stratify = None

    if base_task == "classification":
        class_counts = y.value_counts(dropna=False)

        if class_counts.empty:
            raise ValueError(
                "The classification target has no usable class values."
            )

        minimum_class_count = int(class_counts.min())

        if minimum_class_count < minimum_class_count_for_holdout:
            return _cross_validation_only_split(
                X=X,
                y=y,
                reason=(
                    "A final holdout was not created because the smallest target "
                    f"class contains only {minimum_class_count} row(s). At least "
                    f"{minimum_class_count_for_holdout} rows per class are required "
                    "by the current conservative holdout policy."
                ),
                random_state=random_state,
                holdout_fraction=holdout_fraction,
                minimum_rows_for_holdout=minimum_rows_for_holdout,
            )

        stratify = y

    elif base_task != "regression":
        raise ValueError(
            f"Unsupported base task for validation splitting: {base_task}"
        )

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=holdout_fraction,
        random_state=random_state,
        stratify=stratify,
    )

    summary = {
        "strategy": "cross_validation_with_final_holdout",
        "cross_validation_used": True,
        "holdout_used": True,
        "total_usable_rows": total_rows,
        "cross_validation_rows": int(len(y_train)),
        "holdout_rows": int(len(y_holdout)),
        "holdout_fraction": float(holdout_fraction),
        "requested_holdout_fraction": float(holdout_fraction),
        "minimum_rows_for_holdout": int(minimum_rows_for_holdout),
        "minimum_class_count_for_holdout": (
            int(minimum_class_count_for_holdout)
            if base_task == "classification"
            else None
        ),
        "random_state": int(random_state),
        "split_method": (
            "stratified_train_test_split"
            if base_task == "classification"
            else "random_train_test_split"
        ),
        "reason": (
            "The dataset is large enough to reserve a final untouched holdout "
            "set. Candidate models will be compared using cross-validation on "
            "the training portion only."
        ),
    }

    return ValidationSplit(
        X_train=X_train,
        y_train=y_train,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        summary=summary,
    )