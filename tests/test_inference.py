import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.inference import predict_csv


def test_valid_classification_csv_generates_predictions_and_probabilities(
    tmp_path: Path,
) -> None:
    """
    A valid prediction CSV should:

    - reuse the saved preprocessing and model pipeline;
    - ignore unrelated extra columns;
    - remove the target from model input when present;
    - generate predictions;
    - generate probability columns for classification;
    - save the completed output CSV.
    """

    training_df = pd.DataFrame(
        {
            "Income": [
                25000,
                30000,
                35000,
                40000,
                70000,
                75000,
                80000,
                85000,
            ],
            "Credit_Score": [
                500,
                520,
                540,
                560,
                700,
                720,
                740,
                760,
            ],
            "Approved": [
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
            ],
        }
    )

    expected_features = [
        "Income",
        "Credit_Score",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                expected_features,
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(
        training_df[expected_features],
        training_df["Approved"],
    )

    model_path = (
        tmp_path
        / "classification_pipeline.joblib"
    )

    metadata_path = (
        tmp_path
        / "classification_metadata.json"
    )

    input_path = (
        tmp_path
        / "prediction_input.csv"
    )

    output_path = (
        tmp_path
        / "prediction_output.csv"
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    metadata = {
        "target_column": "Approved",
        "task_type": "binary_classification",
        "base_task": "classification",
        "model_id": "logistic_regression",
        "display_name": "Logistic Regression",
        "tuning_applied": True,
        "preprocessing_config": {
            "numerical_features": expected_features,
            "categorical_features": [],
        },
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=4,
        ),
        encoding="utf-8",
    )

    prediction_input = pd.DataFrame(
        {
            "Income": [
                32000,
                78000,
            ],
            "Credit_Score": [
                530,
                730,
            ],
            "Approved": [
                0,
                1,
            ],
            "Customer_Reference": [
                "TEST-001",
                "TEST-002",
            ],
        }
    )

    prediction_input.to_csv(
        input_path,
        index=False,
    )

    result = predict_csv(
        model_path=str(model_path),
        input_path=str(input_path),
        output_path=str(output_path),
    )

    assert result["success"] is True

    assert result["model_path"] == str(
        model_path
    )

    assert result["metadata_path"] == str(
        metadata_path
    )

    assert result["input_path"] == str(
        input_path
    )

    assert result["output_path"] == str(
        output_path
    )

    assert result["rows_predicted"] == 2

    assert (
        result["model_id"]
        == "logistic_regression"
    )

    assert (
        result["display_name"]
        == "Logistic Regression"
    )

    assert (
        result["task_type"]
        == "binary_classification"
    )

    assert result["target_column"] == "Approved"
    assert result["tuning_applied"] is True

    assert result["expected_features"] == [
        "Income",
        "Credit_Score",
    ]

    assert result["ignored_input_columns"] == [
        "Customer_Reference",
    ]

    assert set(
        result["probability_columns"]
    ) == {
        "probability_0",
        "probability_1",
    }

    assert output_path.exists()

    output_df = pd.read_csv(
        output_path
    )

    assert len(output_df) == 2

    # Original input columns are retained in the output.
    assert "Income" in output_df.columns
    assert "Credit_Score" in output_df.columns
    assert "Approved" in output_df.columns
    assert "Customer_Reference" in output_df.columns

    # Prediction fields are appended.
    assert "prediction" in output_df.columns
    assert "probability_0" in output_df.columns
    assert "probability_1" in output_df.columns
    assert "prediction_confidence" in output_df.columns

    assert output_df["prediction"].tolist() == [
        0,
        1,
    ]

    probability_totals = (
        output_df["probability_0"]
        + output_df["probability_1"]
    )

    assert all(
        abs(total - 1.0) < 1e-9
        for total in probability_totals
    )

    assert output_df[
        "prediction_confidence"
    ].between(
        0.0,
        1.0,
    ).all()

def test_missing_required_feature_raises_clear_error(
    tmp_path: Path,
) -> None:
    """
    Prediction must fail clearly when the input CSV is missing a feature
    required by the saved preprocessing pipeline.

    No prediction output file should be created.
    """

    training_df = pd.DataFrame(
        {
            "Income": [
                25000,
                30000,
                35000,
                40000,
                70000,
                75000,
                80000,
                85000,
            ],
            "Credit_Score": [
                500,
                520,
                540,
                560,
                700,
                720,
                740,
                760,
            ],
            "Approved": [
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
            ],
        }
    )

    expected_features = [
        "Income",
        "Credit_Score",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                expected_features,
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(
        training_df[expected_features],
        training_df["Approved"],
    )

    model_path = (
        tmp_path
        / "missing_feature_pipeline.joblib"
    )

    metadata_path = (
        tmp_path
        / "missing_feature_metadata.json"
    )

    input_path = (
        tmp_path
        / "missing_feature_input.csv"
    )

    output_path = (
        tmp_path
        / "missing_feature_output.csv"
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    metadata = {
        "target_column": "Approved",
        "task_type": "binary_classification",
        "base_task": "classification",
        "model_id": "logistic_regression",
        "display_name": "Logistic Regression",
        "preprocessing_config": {
            "numerical_features": expected_features,
            "categorical_features": [],
        },
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=4,
        ),
        encoding="utf-8",
    )

    # Credit_Score is deliberately missing.
    prediction_input = pd.DataFrame(
        {
            "Income": [
                32000,
                78000,
            ],
            "Customer_Reference": [
                "TEST-001",
                "TEST-002",
            ],
        }
    )

    prediction_input.to_csv(
        input_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match=(
            "The prediction dataset is missing required "
            r"feature column\(s\): Credit_Score"
        ),
    ):
        predict_csv(
            model_path=str(model_path),
            input_path=str(input_path),
            output_path=str(output_path),
        )

    assert output_path.exists() is False

def test_metadata_missing_required_field_raises_clear_error(
    tmp_path: Path,
) -> None:
    """
    Inference must reject incomplete model metadata before attempting
    prediction.

    The error should identify the missing field, and no output CSV should
    be created.
    """

    model_path = (
        tmp_path
        / "invalid_metadata_pipeline.joblib"
    )

    metadata_path = (
        tmp_path
        / "invalid_metadata_metadata.json"
    )

    input_path = (
        tmp_path
        / "invalid_metadata_input.csv"
    )

    output_path = (
        tmp_path
        / "invalid_metadata_output.csv"
    )

    # The pipeline contents are not needed because metadata validation occurs
    # before joblib loads the artifact.
    model_path.write_bytes(
        b"placeholder model artifact"
    )

    incomplete_metadata = {
        "target_column": "Approved",
        "task_type": "binary_classification",
        "base_task": "classification",
        "model_id": "logistic_regression",
        # preprocessing_config is intentionally missing.
    }

    metadata_path.write_text(
        json.dumps(
            incomplete_metadata,
            indent=4,
        ),
        encoding="utf-8",
    )

    prediction_input = pd.DataFrame(
        {
            "Income": [
                32000,
                78000,
            ],
            "Credit_Score": [
                530,
                730,
            ],
        }
    )

    prediction_input.to_csv(
        input_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match=(
            r"The metadata file is missing required field\(s\): "
            r"preprocessing_config"
        ),
    ):
        predict_csv(
            model_path=str(model_path),
            input_path=str(input_path),
            output_path=str(output_path),
        )

    assert output_path.exists() is False

def test_empty_prediction_csv_raises_clear_error(
    tmp_path: Path,
) -> None:
    """
    A prediction CSV with valid headers but no data rows must be rejected.

    No output prediction file should be created.
    """

    training_df = pd.DataFrame(
        {
            "Income": [
                25000,
                30000,
                35000,
                40000,
                70000,
                75000,
                80000,
                85000,
            ],
            "Credit_Score": [
                500,
                520,
                540,
                560,
                700,
                720,
                740,
                760,
            ],
            "Approved": [
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
            ],
        }
    )

    expected_features = [
        "Income",
        "Credit_Score",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                expected_features,
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(
        training_df[expected_features],
        training_df["Approved"],
    )

    model_path = (
        tmp_path
        / "empty_input_pipeline.joblib"
    )

    metadata_path = (
        tmp_path
        / "empty_input_metadata.json"
    )

    input_path = (
        tmp_path
        / "empty_prediction_input.csv"
    )

    output_path = (
        tmp_path
        / "empty_prediction_output.csv"
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    metadata = {
        "target_column": "Approved",
        "task_type": "binary_classification",
        "base_task": "classification",
        "model_id": "logistic_regression",
        "display_name": "Logistic Regression",
        "preprocessing_config": {
            "numerical_features": expected_features,
            "categorical_features": [],
        },
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=4,
        ),
        encoding="utf-8",
    )

    # Create a valid CSV structure with headers but no data rows.
    empty_prediction_input = pd.DataFrame(
        columns=[
            "Income",
            "Credit_Score",
        ]
    )

    empty_prediction_input.to_csv(
        input_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="The prediction input CSV contains no rows.",
    ):
        predict_csv(
            model_path=str(model_path),
            input_path=str(input_path),
            output_path=str(output_path),
        )

    assert output_path.exists() is False

def test_corrupted_model_artifact_does_not_create_output(
    tmp_path: Path,
) -> None:
    """
    A corrupted model artifact must fail during loading.

    Valid metadata and input data must not hide an invalid pipeline file, and
    no prediction output should be created.
    """

    model_path = (
        tmp_path
        / "corrupted_pipeline.joblib"
    )

    metadata_path = (
        tmp_path
        / "corrupted_metadata.json"
    )

    input_path = (
        tmp_path
        / "corrupted_model_input.csv"
    )

    output_path = (
        tmp_path
        / "corrupted_model_output.csv"
    )

    # Deliberately write invalid Joblib content.
    model_path.write_bytes(
        b"This is not a valid Joblib model artifact."
    )

    metadata = {
        "target_column": "Approved",
        "task_type": "binary_classification",
        "base_task": "classification",
        "model_id": "logistic_regression",
        "display_name": "Logistic Regression",
        "preprocessing_config": {
            "numerical_features": [
                "Income",
                "Credit_Score",
            ],
            "categorical_features": [],
        },
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=4,
        ),
        encoding="utf-8",
    )

    prediction_input = pd.DataFrame(
        {
            "Income": [
                32000,
                78000,
            ],
            "Credit_Score": [
                530,
                730,
            ],
        }
    )

    prediction_input.to_csv(
        input_path,
        index=False,
    )

    with pytest.raises(Exception):
        predict_csv(
            model_path=str(model_path),
            input_path=str(input_path),
            output_path=str(output_path),
        )

    assert output_path.exists() is False

def test_valid_regression_csv_generates_predictions_without_probabilities(
    tmp_path: Path,
) -> None:
    """
    A valid regression pipeline should generate numeric predictions without
    classification probability columns or prediction confidence.
    """

    training_df = pd.DataFrame(
        {
            "Square_Metres": [
                60.0,
                80.0,
                100.0,
                120.0,
                140.0,
                160.0,
                180.0,
                200.0,
            ],
            "Bedrooms": [
                1,
                2,
                2,
                3,
                3,
                4,
                4,
                5,
            ],
            "Sale_Price": [
                320000.0,
                410000.0,
                480000.0,
                570000.0,
                650000.0,
                740000.0,
                830000.0,
                920000.0,
            ],
        }
    )

    expected_features = [
        "Square_Metres",
        "Bedrooms",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                expected_features,
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                Ridge(
                    alpha=1.0,
                ),
            ),
        ]
    )

    pipeline.fit(
        training_df[expected_features],
        training_df["Sale_Price"],
    )

    model_path = (
        tmp_path
        / "regression_pipeline.joblib"
    )

    metadata_path = (
        tmp_path
        / "regression_metadata.json"
    )

    input_path = (
        tmp_path
        / "regression_input.csv"
    )

    output_path = (
        tmp_path
        / "regression_output.csv"
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    metadata = {
        "target_column": "Sale_Price",
        "task_type": "regression",
        "base_task": "regression",
        "model_id": "ridge_regression",
        "display_name": "Ridge Regression",
        "tuning_applied": False,
        "preprocessing_config": {
            "numerical_features": expected_features,
            "categorical_features": [],
        },
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=4,
        ),
        encoding="utf-8",
    )

    prediction_input = pd.DataFrame(
        {
            "Square_Metres": [
                110.0,
                175.0,
            ],
            "Bedrooms": [
                3,
                4,
            ],
            "Property_ID": [
                "PROP-001",
                "PROP-002",
            ],
        }
    )

    prediction_input.to_csv(
        input_path,
        index=False,
    )

    result = predict_csv(
        model_path=str(model_path),
        input_path=str(input_path),
        output_path=str(output_path),
    )

    assert result["success"] is True
    assert result["rows_predicted"] == 2

    assert result["model_id"] == "ridge_regression"
    assert result["display_name"] == "Ridge Regression"
    assert result["task_type"] == "regression"
    assert result["target_column"] == "Sale_Price"
    assert result["tuning_applied"] is False

    assert result["expected_features"] == [
        "Square_Metres",
        "Bedrooms",
    ]

    assert result["ignored_input_columns"] == [
        "Property_ID",
    ]

    # Regression models do not return class probabilities.
    assert result["probability_columns"] == []

    assert output_path.exists()

    output_df = pd.read_csv(
        output_path
    )

    assert len(output_df) == 2

    # Original input columns remain available.
    assert "Square_Metres" in output_df.columns
    assert "Bedrooms" in output_df.columns
    assert "Property_ID" in output_df.columns

    # Regression output contains only the prediction field.
    assert "prediction" in output_df.columns

    assert "prediction_confidence" not in output_df.columns
    assert "probability_0" not in output_df.columns
    assert "probability_1" not in output_df.columns

    assert output_df["prediction"].notna().all()

    assert pd.api.types.is_numeric_dtype(
        output_df["prediction"]
    )

    # The larger property should receive the larger predicted price.
    assert (
        output_df.loc[1, "prediction"]
        > output_df.loc[0, "prediction"]
    )