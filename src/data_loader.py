from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class DatasetBundle:
    """
    Stores the loaded dataset and its main components.

    df: full dataframe
    X: feature columns
    y: target column
    metadata: basic information about the loaded dataset
    """

    df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    metadata: dict[str, Any]


def load_dataset(file_path: str, target_column: str) -> DatasetBundle:
    """
    Load a CSV dataset, validate it, and separate features from target.

    Args:
        file_path: Path to the CSV file.
        target_column: Name of the column we want to predict.

    Returns:
        DatasetBundle containing:
        - full dataframe
        - feature dataframe X
        - target series y
        - dataset metadata
    """

    path = Path(file_path)

    # 1. Check whether the file exists
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # 2. Check whether it is a CSV file
    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            "Only CSV files are supported in this version."
        )

    # 3. Try to read the CSV
    try:
        df = pd.read_csv(path)
    except Exception as error:
        raise ValueError(f"Could not read CSV file. Original error: {error}")

    # 4. Check if the dataset is empty
    if df.empty:
        raise ValueError("The CSV file is empty.")

    # 5. Remove columns that are completely empty
    original_columns = list(df.columns)
    df = df.dropna(axis=1, how="all")
    removed_empty_columns = [
        col for col in original_columns if col not in df.columns
    ]

    # 6. Check whether the target column exists
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns are: {list(df.columns)}"
        )

    # 7. Check whether the target column is usable
    if df[target_column].isna().all():
        raise ValueError(
            f"Target column '{target_column}' contains only missing values."
        )

    # 8. Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 9. Check whether there are feature columns
    if X.shape[1] == 0:
        raise ValueError(
            "The dataset has no feature columns after removing the target column."
        )

    # 10. Create useful metadata
    metadata = {
        "file_path": str(path),
        "file_name": path.name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "feature_count": int(X.shape[1]),
        "target_column": target_column,
        "target_missing_count": int(y.isna().sum()),
        "removed_empty_columns": removed_empty_columns,
        "column_names": list(df.columns),
    }

    return DatasetBundle(
        df=df,
        X=X,
        y=y,
        metadata=metadata,
    )