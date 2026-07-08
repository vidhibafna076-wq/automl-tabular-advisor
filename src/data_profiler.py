from typing import Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)


def _is_numeric_feature(series: pd.Series) -> bool:
    """
    Decide whether a feature should be treated as numerical.

    Boolean columns are not treated as numerical here because they are often
    better handled as categorical/binary features in preprocessing.
    """

    return is_numeric_dtype(series) and not is_bool_dtype(series)


def _detect_possible_date_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """
    Detect columns that are already datetime or look like dates.

    We keep this conservative. We do not want to wrongly convert random text
    columns into dates.
    """

    possible_date_columns = []

    for column in columns:
        series = df[column]

        if is_datetime64_any_dtype(series):
            possible_date_columns.append(column)
            continue

        if series.dtype == "object":
            non_missing = series.dropna().astype(str)

            if non_missing.empty:
                continue

            sample = non_missing.head(20)

            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                success_rate = parsed.notna().mean()

                if success_rate >= 0.8:
                    possible_date_columns.append(column)

            except Exception:
                continue

    return possible_date_columns


def _summarise_missing_values(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Create a missing-value summary for each column.
    """

    rows = len(df)
    missing_summary = []

    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        missing_percentage = round((missing_count / rows) * 100, 2) if rows > 0 else 0.0

        missing_summary.append(
            {
                "column": column,
                "missing_count": missing_count,
                "missing_percentage": float(missing_percentage),
            }
        )

    return missing_summary


def _detect_possible_id_columns(df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    """
    Detect columns that look like identifiers.

    ID-like columns usually do not help prediction. They can also cause
    overfitting because each row may have a unique ID.

    Important design choice:
    - A column name containing ID/UUID/key is strong evidence.
    - A mostly unique text column may also be an ID.
    - A mostly unique numeric column is NOT automatically treated as an ID,
      because useful numeric features like income, price, age, or amount can
      naturally have many unique values.
    """

    possible_id_columns = []

    for column in feature_columns:
        unique_count = int(df[column].nunique(dropna=True))
        unique_ratio = unique_count / len(df) if len(df) > 0 else 0

        column_lower = column.lower()

        name_suggests_id = (
            column_lower == "id"
            or column_lower.endswith("_id")
            or column_lower.startswith("id_")
            or "uuid" in column_lower
            or "identifier" in column_lower
            or column_lower.endswith("_key")
            or column_lower.endswith("key")
        )

        mostly_unique = unique_ratio >= 0.95
        is_text_column = df[column].dtype == "object"

        if name_suggests_id:
            possible_id_columns.append(column)
            continue

        if mostly_unique and is_text_column:
            possible_id_columns.append(column)

    return possible_id_columns

def _detect_high_cardinality_columns(
    df: pd.DataFrame,
    categorical_columns: list[str],
) -> list[dict[str, Any]]:
    """
    Detect categorical columns with too many unique values.

    High-cardinality columns can be difficult because one-hot encoding may
    create too many new columns.
    """

    high_cardinality = []

    for column in categorical_columns:
        unique_count = int(df[column].nunique(dropna=True))
        unique_ratio = unique_count / len(df) if len(df) > 0 else 0

        if unique_count > 50 and unique_ratio > 0.20:
            high_cardinality.append(
                {
                    "column": column,
                    "unique_count": unique_count,
                    "unique_ratio": round(float(unique_ratio), 4),
                }
            )

    return high_cardinality


def _summarise_numeric_columns(
    df: pd.DataFrame,
    numerical_columns: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Create basic numeric summaries and simple outlier counts.

    Outliers are detected using the IQR method:
    lower bound = Q1 - 1.5 * IQR
    upper bound = Q3 + 1.5 * IQR

    This does not automatically remove outliers. It only reports them.
    """

    summary = {}

    for column in numerical_columns:
        series = df[column].dropna()

        if series.empty:
            summary[column] = {
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "std": None,
                "outlier_count": 0,
                "outlier_percentage": 0.0,
            }
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            outlier_count = 0
        else:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_count = int(
                ((series < lower_bound) | (series > upper_bound)).sum()
            )

        outlier_percentage = round((outlier_count / len(df)) * 100, 2)

        summary[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()) if len(series) > 1 else 0.0,
            "outlier_count": outlier_count,
            "outlier_percentage": float(outlier_percentage),
        }

    return summary


def _summarise_categorical_columns(
    df: pd.DataFrame,
    categorical_columns: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Create summaries for categorical columns.
    """

    summary = {}

    for column in categorical_columns:
        value_counts = df[column].value_counts(dropna=False).head(10)

        summary[column] = {
            "unique_count": int(df[column].nunique(dropna=True)),
            "top_values": {
                str(label): int(count)
                for label, count in value_counts.items()
            },
        }

    return summary


def _summarise_target(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """
    Create a target-column summary.

    This is intentionally simple because task_detector.py already gives
    a more task-specific target summary.
    """

    target = df[target_column]

    return {
        "missing_count": int(target.isna().sum()),
        "unique_count": int(target.nunique(dropna=True)),
        "preview_values": [str(value) for value in target.dropna().unique()[:10]],
    }


def profile_dataset(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """
    Generate a structured profile of the dataset.

    This profile is used by the future planner, data quality checker,
    preprocessing engine, and model advisor.
    """

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    feature_df = df.drop(columns=[target_column])
    feature_columns = list(feature_df.columns)

    numerical_columns = [
        column
        for column in feature_columns
        if _is_numeric_feature(feature_df[column])
    ]

    categorical_columns = [
        column
        for column in feature_columns
        if column not in numerical_columns
    ]

    possible_date_columns = _detect_possible_date_columns(df, feature_columns)

    duplicate_rows = int(df.duplicated().sum())

    unique_counts = {
        column: int(df[column].nunique(dropna=True))
        for column in df.columns
    }

    constant_columns = [
        column
        for column in feature_columns
        if unique_counts[column] <= 1
    ]

    possible_id_columns = _detect_possible_id_columns(df, feature_columns)

    high_cardinality_columns = _detect_high_cardinality_columns(
        df=df,
        categorical_columns=categorical_columns,
    )

    profile = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "feature_count": int(len(feature_columns)),
        "target_column": target_column,
        "feature_columns": feature_columns,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "possible_date_columns": possible_date_columns,
        "missing_values": _summarise_missing_values(df),
        "duplicate_rows": duplicate_rows,
        "unique_counts": unique_counts,
        "constant_columns": constant_columns,
        "possible_id_columns": possible_id_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "numeric_summary": _summarise_numeric_columns(df, numerical_columns),
        "categorical_summary": _summarise_categorical_columns(
            df,
            categorical_columns,
        ),
        "target_summary": _summarise_target(df, target_column),
    }

    return profile