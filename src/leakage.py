from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


SUSPICIOUS_TARGET_RELATED_WORDS = [
    "target",
    "label",
    "outcome",
    "result",
    "status",
    "approved",
    "approval",
    "decision",
    "class",
]


def _normalise_name(name: str) -> str:
    """
    Normalise a column name for simple leakage keyword checks.
    """

    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def _add_warning(
    warnings: list[dict[str, Any]],
    column: str,
    severity: str,
    warning_type: str,
    finding: str,
    recommendation: str,
) -> None:
    """
    Add a leakage warning in a consistent structure.
    """

    warnings.append(
        {
            "column": column,
            "severity": severity,
            "warning_type": warning_type,
            "finding": finding,
            "why_it_matters": (
                "Target leakage can make validation scores look unrealistically strong "
                "because the model may be using information that would not be available "
                "at prediction time."
            ),
            "recommendation": recommendation,
        }
    )


def _check_name_based_leakage(
    df: pd.DataFrame,
    target_column: str,
    warnings: list[dict[str, Any]],
) -> None:
    """
    Check whether feature names look suspiciously related to the target.
    """

    target_name = _normalise_name(target_column)

    for column in df.columns:
        if column == target_column:
            continue

        column_name = _normalise_name(column)

        if target_name and target_name in column_name:
            _add_warning(
                warnings=warnings,
                column=column,
                severity="high",
                warning_type="target_name_overlap",
                finding=(
                    f"Feature column '{column}' contains the target name "
                    f"'{target_column}'."
                ),
                recommendation=(
                    "Review this column carefully. If it contains the answer, a future "
                    "status, or a post-outcome value, remove it before modelling."
                ),
            )
            continue

        for suspicious_word in SUSPICIOUS_TARGET_RELATED_WORDS:
            if suspicious_word in column_name:
                _add_warning(
                    warnings=warnings,
                    column=column,
                    severity="medium",
                    warning_type="suspicious_outcome_name",
                    finding=(
                        f"Feature column '{column}' contains the suspicious word "
                        f"'{suspicious_word}'."
                    ),
                    recommendation=(
                        "Review whether this feature would genuinely be available before "
                        "the prediction is made. If it is only known after the outcome, "
                        "it may cause target leakage."
                    ),
                )
                break


def _encode_binary_target(y: pd.Series) -> pd.Series | None:
    """
    Convert a binary target to 0/1 codes for simple association checks.
    """

    y_clean = y.dropna()

    if y_clean.nunique(dropna=True) != 2:
        return None

    unique_values = list(y_clean.unique())

    mapping = {
        unique_values[0]: 0,
        unique_values[1]: 1,
    }

    return y.map(mapping)


def _check_numeric_near_perfect_association(
    df: pd.DataFrame,
    target_column: str,
    warnings: list[dict[str, Any]],
) -> None:
    """
    Check for simple near-perfect numeric association with a binary target.

    This is intentionally conservative. It does not prove leakage, but it can
    flag suspicious columns for human review.
    """

    y_encoded = _encode_binary_target(df[target_column])

    if y_encoded is None:
        return

    for column in df.columns:
        if column == target_column:
            continue

        if not is_numeric_dtype(df[column]):
            continue

        valid_data = pd.DataFrame(
            {
                "feature": df[column],
                "target": y_encoded,
            }
        ).dropna()

        if len(valid_data) < 10:
            continue

        if valid_data["feature"].nunique(dropna=True) < 2:
            continue

        try:
            correlation = valid_data["feature"].corr(valid_data["target"])
        except Exception:
            continue

        if correlation is None or np.isnan(correlation):
            continue

        if abs(correlation) >= 0.98:
            _add_warning(
                warnings=warnings,
                column=column,
                severity="high",
                warning_type="near_perfect_numeric_association",
                finding=(
                    f"Feature column '{column}' has near-perfect correlation "
                    f"with the binary target. Correlation: {round(float(correlation), 4)}."
                ),
                recommendation=(
                    "Investigate this feature before trusting model results. It may be "
                    "a direct proxy for the target or may only be available after the "
                    "outcome is known."
                ),
            )


def detect_target_leakage_warnings(
    df: pd.DataFrame,
    target_column: str,
) -> list[dict[str, Any]]:
    """
    Detect basic target leakage warning signs.

    These checks are warnings only. They do not automatically remove columns.

    Current checks:
    - feature name contains target name
    - feature name contains suspicious outcome-related words
    - numeric feature has near-perfect association with a binary target
    """

    warnings: list[dict[str, Any]] = []

    if target_column not in df.columns:
        return warnings

    _check_name_based_leakage(
        df=df,
        target_column=target_column,
        warnings=warnings,
    )

    _check_numeric_near_perfect_association(
        df=df,
        target_column=target_column,
        warnings=warnings,
    )

    return warnings