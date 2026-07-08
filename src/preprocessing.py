from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _make_one_hot_encoder() -> OneHotEncoder:
    """
    Create a OneHotEncoder that works across different scikit-learn versions.

    Newer versions use sparse_output.
    Older versions use sparse.
    """

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def prepare_features_and_target(
    df: pd.DataFrame,
    preprocessing_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X and y according to the preprocessing configuration.

    This function:
    - removes rows with missing target values
    - selects final feature columns
    - separates X and y

    It does not fit preprocessing.
    """

    target_column = preprocessing_config["target_column"]
    final_feature_columns = preprocessing_config["final_feature_columns"]

    required_columns = [target_column] + final_feature_columns

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required columns are missing from the dataset: "
            + ", ".join(missing_columns)
        )

    working_df = df.dropna(subset=[target_column]).copy()

    X = working_df[final_feature_columns].copy()
    y = working_df[target_column].copy()

    if X.empty:
        raise ValueError("Feature matrix X is empty after preprocessing preparation.")

    if y.empty:
        raise ValueError("Target vector y is empty after preprocessing preparation.")

    return X, y


def build_preprocessor(
    preprocessing_config: dict[str, Any],
) -> ColumnTransformer:
    """
    Build an unfitted scikit-learn ColumnTransformer.

    Important:
    This function only builds the preprocessor.
    It does not fit it.

    The preprocessor should be fitted later inside cross-validation
    or inside a full model pipeline.
    """

    numerical_features = preprocessing_config["numerical_features"]
    categorical_features = preprocessing_config["categorical_features"]

    transformers = []

    if numerical_features:
        numerical_strategy = preprocessing_config["missing_value_strategy"]["numerical"]["strategy"]

        numerical_steps = [
            ("imputer", SimpleImputer(strategy=numerical_strategy)),
        ]

        if preprocessing_config["scaling_strategy"]["apply_to_numerical"]:
            numerical_steps.append(
                ("scaler", StandardScaler())
            )

        numerical_pipeline = Pipeline(
            steps=numerical_steps
        )

        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            )
        )

    if categorical_features:
        categorical_strategy = preprocessing_config["missing_value_strategy"]["categorical"]["strategy"]

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=categorical_strategy)),
                ("encoder", _make_one_hot_encoder()),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError(
            "No numerical or categorical features were provided for preprocessing."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor


def create_preprocessing_pipeline_summary(
    preprocessing_config: dict[str, Any],
    rows_after_target_cleaning: int,
) -> dict[str, Any]:
    """
    Create a JSON-safe summary of the preprocessing pipeline.

    This summary can be saved in ExperimentState.
    """

    numerical_features = preprocessing_config["numerical_features"]
    categorical_features = preprocessing_config["categorical_features"]

    return {
        "pipeline_type": "sklearn_column_transformer",
        "is_fitted": False,
        "rows_after_target_cleaning": rows_after_target_cleaning,
        "target_column": preprocessing_config["target_column"],
        "columns_to_drop": preprocessing_config["columns_to_drop"],
        "numerical_pipeline": {
            "features": numerical_features,
            "steps": [
                "SimpleImputer(strategy='median')",
                "StandardScaler()",
            ] if numerical_features else [],
        },
        "categorical_pipeline": {
            "features": categorical_features,
            "steps": [
                "SimpleImputer(strategy='most_frequent')",
                "OneHotEncoder(handle_unknown='ignore')",
            ] if categorical_features else [],
        },
        "final_feature_columns": preprocessing_config["final_feature_columns"],
        "leakage_note": (
            "This preprocessor is intentionally unfitted. It should be fitted only "
            "inside training folds or inside a full model pipeline."
        ),
    }