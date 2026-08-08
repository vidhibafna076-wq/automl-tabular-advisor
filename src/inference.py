from datetime import datetime
from pathlib import Path
from typing import Any
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from src.provenance import runtime_environment


def _major_version(value: str) -> str:
    """Return the numeric major component from a version string."""

    return value.split(".", maxsplit=1)[0]


def _validate_artifact_environment(metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate persisted-library compatibility before deserializing a model."""

    saved = metadata.get("artifact_environment")
    current = runtime_environment()

    if saved is None:
        return {
            "status": "legacy_unverified",
            "compatible": None,
            "warnings": [
                "This legacy artifact does not record its training environment."
            ],
        }

    if not isinstance(saved, dict) or not isinstance(
        saved.get("package_versions"), dict
    ):
        raise ValueError(
            "The model metadata contains malformed artifact environment information."
        )

    saved_packages = saved["package_versions"]
    current_packages = current["package_versions"]
    incompatible: list[str] = []
    compatibility_warnings: list[str] = []

    saved_sklearn = saved_packages.get("scikit-learn")
    current_sklearn = current_packages.get("scikit-learn")
    if saved_sklearn and saved_sklearn != current_sklearn:
        incompatible.append(
            "scikit-learn "
            f"(saved with {saved_sklearn}, running with {current_sklearn})"
        )

    for package in ("numpy", "joblib"):
        saved_version = saved_packages.get(package)
        current_version = current_packages.get(package)
        if not saved_version or not current_version:
            compatibility_warnings.append(
                f"Could not verify the saved {package} version."
            )
        elif _major_version(saved_version) != _major_version(current_version):
            incompatible.append(
                f"{package} (saved with {saved_version}, running with {current_version})"
            )
        elif saved_version != current_version:
            compatibility_warnings.append(
                f"{package} differs: saved with {saved_version}, "
                f"running with {current_version}."
            )

    saved_python = str(saved.get("python_version", ""))
    current_python = str(current["python_version"])
    if not saved_python:
        compatibility_warnings.append("Could not verify the saved Python version.")
    elif _major_version(saved_python) != _major_version(current_python):
        incompatible.append(
            f"Python (saved with {saved_python}, running with {current_python})"
        )
    elif saved_python != current_python:
        compatibility_warnings.append(
            f"Python differs: saved with {saved_python}, running with {current_python}."
        )

    if incompatible:
        raise ValueError(
            "The saved model artifact is incompatible with this runtime: "
            + "; ".join(incompatible)
            + ". Retrain the model in the current environment."
        )

    return {
        "status": "compatible",
        "compatible": True,
        "warnings": compatibility_warnings,
        "saved_environment": saved,
        "current_environment": current,
    }


def _resolve_metadata_path(
    model_path: Path,
    metadata_path: str | None,
) -> Path:
    """
    Resolve the metadata file associated with a saved pipeline.

    When the metadata path is not supplied, infer it from the standard artifact
    naming convention:

    *_pipeline.joblib
    -> *_metadata.json
    """

    if metadata_path:
        resolved_path = Path(metadata_path)
    else:
        model_name = model_path.name

        if not model_name.endswith("_pipeline.joblib"):
            raise ValueError(
                "Could not infer the metadata path because the model filename "
                "does not end with '_pipeline.joblib'. Pass --metadata explicitly."
            )

        metadata_name = model_name.replace(
            "_pipeline.joblib",
            "_metadata.json",
        )

        resolved_path = model_path.with_name(metadata_name)

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Model metadata file was not found: {resolved_path}"
        )

    return resolved_path


def _load_metadata(metadata_path: Path) -> dict[str, Any]:
    """
    Load and validate saved model metadata.
    """

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    required_fields = {
        "target_column",
        "task_type",
        "base_task",
        "model_id",
        "preprocessing_config",
    }

    missing_fields = sorted(
        field
        for field in required_fields
        if field not in metadata
    )

    if missing_fields:
        raise ValueError(
            "The metadata file is missing required field(s): "
            + ", ".join(missing_fields)
        )

    return metadata


def _get_expected_features(
    metadata: dict[str, Any],
) -> list[str]:
    """
    Determine the raw input features expected by the saved preprocessing
    pipeline.
    """

    preprocessing_config = metadata.get(
        "preprocessing_config",
        {},
    )

    numerical_features = preprocessing_config.get(
        "numerical_features",
        [],
    )

    categorical_features = preprocessing_config.get(
        "categorical_features",
        [],
    )

    expected_features: list[str] = []

    for column in [
        *numerical_features,
        *categorical_features,
    ]:
        if column not in expected_features:
            expected_features.append(column)

    if not expected_features:
        raise ValueError(
            "No expected input features were found in the saved metadata."
        )

    return expected_features


def _safe_probability_column_name(label: Any) -> str:
    """
    Convert a class label into a safe CSV column suffix.
    """

    label_text = str(label).strip()

    safe_characters = []

    for character in label_text:
        if character.isalnum():
            safe_characters.append(character)
        else:
            safe_characters.append("_")

    safe_label = "".join(safe_characters).strip("_")

    return safe_label or "unknown_class"


def _prepare_inference_features(
    input_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Validate the input dataset and return features in the exact order expected
    by the saved pipeline.

    Extra columns are ignored. Missing required columns cause a clear error.
    """

    expected_features = _get_expected_features(metadata)
    target_column = metadata.get("target_column")

    feature_df = input_df.copy()

    if target_column in feature_df.columns:
        feature_df = feature_df.drop(
            columns=[target_column]
        )

    missing_features = [
        column
        for column in expected_features
        if column not in feature_df.columns
    ]

    if missing_features:
        raise ValueError(
            "The prediction dataset is missing required feature column(s): "
            + ", ".join(missing_features)
        )

    ignored_columns = [
        column
        for column in feature_df.columns
        if column not in expected_features
    ]

    prepared_features = feature_df[
        expected_features
    ].copy()

    return prepared_features, ignored_columns


def _default_output_path(
    input_path: Path,
) -> Path:
    """
    Create a timestamped default prediction output path.
    """

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_directory = Path(
        "outputs/predictions"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory / (
        f"{input_path.stem}_{timestamp}_predictions.csv"
    )


def predict_csv(
    model_path: str,
    input_path: str,
    metadata_path: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """
    Load a saved AutoML pipeline and generate predictions for a CSV dataset.

    The saved preprocessing pipeline is reused automatically, ensuring that
    new data receives the same transformations used during model training.
    """

    resolved_model_path = Path(model_path)
    resolved_input_path = Path(input_path)

    if not resolved_model_path.exists():
        raise FileNotFoundError(
            f"Saved model pipeline was not found: {resolved_model_path}"
        )

    if not resolved_input_path.exists():
        raise FileNotFoundError(
            f"Prediction input CSV was not found: {resolved_input_path}"
        )

    resolved_metadata_path = _resolve_metadata_path(
        model_path=resolved_model_path,
        metadata_path=metadata_path,
    )

    metadata = _load_metadata(
        resolved_metadata_path
    )

    artifact_compatibility = _validate_artifact_environment(metadata)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
            module=r"joblib\.numpy_pickle",
        )
        pipeline = joblib.load(
            resolved_model_path
        )

    input_df = pd.read_csv(
        resolved_input_path
    )

    if input_df.empty:
        raise ValueError(
            "The prediction input CSV contains no rows."
        )

    prepared_features, ignored_columns = (
        _prepare_inference_features(
            input_df=input_df,
            metadata=metadata,
        )
    )

    predictions = pipeline.predict(
        prepared_features
    )

    result_df = input_df.copy()
    result_df["prediction"] = predictions

    probability_columns: list[str] = []

    if hasattr(pipeline, "predict_proba"):
        try:
            probabilities = pipeline.predict_proba(
                prepared_features
            )

            estimator = pipeline.named_steps.get(
                "model"
            )

            classes = getattr(
                estimator,
                "classes_",
                [],
            )

            if len(classes) == probabilities.shape[1]:
                for class_index, class_label in enumerate(
                    classes
                ):
                    safe_label = (
                        _safe_probability_column_name(
                            class_label
                        )
                    )

                    probability_column = (
                        f"probability_{safe_label}"
                    )

                    result_df[
                        probability_column
                    ] = probabilities[
                        :,
                        class_index,
                    ]

                    probability_columns.append(
                        probability_column
                    )

                result_df[
                    "prediction_confidence"
                ] = np.max(
                    probabilities,
                    axis=1,
                )

        except (AttributeError, ValueError):
            # Predictions remain valid even when probabilities are unavailable.
            probability_columns = []

    resolved_output_path = (
        Path(output_path)
        if output_path
        else _default_output_path(
            resolved_input_path
        )
    )

    resolved_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        resolved_output_path,
        index=False,
    )

    return {
        "success": True,
        "model_path": str(
            resolved_model_path
        ),
        "metadata_path": str(
            resolved_metadata_path
        ),
        "input_path": str(
            resolved_input_path
        ),
        "output_path": str(
            resolved_output_path
        ),
        "rows_predicted": int(
            len(result_df)
        ),
        "model_id": metadata.get(
            "model_id"
        ),
        "display_name": metadata.get(
            "display_name"
        ),
        "task_type": metadata.get(
            "task_type"
        ),
        "target_column": metadata.get(
            "target_column"
        ),
        "tuning_applied": metadata.get(
            "tuning_applied",
            False,
        ),
        "expected_features": (
            _get_expected_features(metadata)
        ),
        "ignored_input_columns": (
            ignored_columns
        ),
        "probability_columns": (
            probability_columns
        ),
        "artifact_compatibility": artifact_compatibility,
    }
