import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import src.tools.model_persistence_tool as persistence_module
from src.state import ExperimentState
from src.tools.model_persistence_tool import model_persistence_tool


def test_approved_candidate_is_saved_with_tuned_parameters(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """
    An approved candidate must be fitted, saved with its metadata, and use
    the tuned parameters accepted for final evaluation.

    All artifacts are written to pytest's temporary directory.
    """

    dataset = pd.DataFrame(
        {
            "feature_a": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.6,
                0.7,
                0.8,
                0.9,
            ],
            "feature_b": [
                1.0,
                1.2,
                1.4,
                1.6,
                2.0,
                2.2,
                2.4,
                2.6,
            ],
            "Target": [
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

    X = dataset[
        [
            "feature_a",
            "feature_b",
        ]
    ].copy()

    y = dataset["Target"].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                [
                    "feature_a",
                    "feature_b",
                ],
            ),
        ],
        remainder="drop",
    )

    candidate_model = SimpleNamespace(
        estimator=LogisticRegression(
            max_iter=1000,
            random_state=42,
        )
    )

    monkeypatch.setattr(
        persistence_module,
        "load_dataset",
        lambda file_path, target_column: SimpleNamespace(
            df=dataset
        ),
    )

    monkeypatch.setattr(
        persistence_module,
        "prepare_features_and_target",
        lambda df, preprocessing_config: (
            X,
            y,
        ),
    )

    monkeypatch.setattr(
        persistence_module,
        "build_preprocessor",
        lambda preprocessing_config: preprocessor,
    )

    monkeypatch.setattr(
        persistence_module,
        "get_model_by_id",
        lambda base_task, model_id, random_state: (
            candidate_model
        ),
    )

    state = ExperimentState(
        dataset_path="approved_test.csv",
        target_column="Target",
        user_objective="Test approved persistence",
    )

    state.status = "reliability_critique_completed"
    state.task_type = "binary_classification"
    state.base_task = "classification"

    state.preprocessing_config = {
        "numerical_features": [
            "feature_a",
            "feature_b",
        ],
        "categorical_features": [],
    }

    state.validation_summary = {
        "holdout_used": True,
    }

    state.holdout_result = {
        "status": "completed",
        "passes_holdout_guardrail": True,
    }

    state.tuning_result = {
        "status": "completed",
        "model_id": "logistic_regression",
        "accepted_for_final_evaluation": True,
        "best_params": {
            "C": 0.5,
            "class_weight": None,
        },
    }

    state.critic_report = {
        "recommendation_decision": (
            "recommend_candidate_model"
        ),
        "selected_candidate": {
            "model_id": "logistic_regression",
            "display_name": "Logistic Regression",
            "primary_metric": "f1",
            "primary_score": 0.8,
        },
    }

    model_directory = tmp_path / "models"

    result = model_persistence_tool(
        state=state,
        output_dir=str(model_directory),
        random_state=42,
    )

    assert result["success"] is True
    assert result["error"] is None

    updated_state = result["state"]
    artifact_summary = result[
        "model_artifact_summary"
    ]
    feature_importance = result[
        "feature_importance_summary"
    ]

    assert (
        updated_state.status
        == "model_persistence_checked"
    )

    assert (
        "checked_model_persistence"
        in updated_state.completed_steps
    )

    assert artifact_summary["status"] == "saved"
    assert (
        artifact_summary["critic_decision"]
        == "recommend_candidate_model"
    )
    assert (
        artifact_summary["model_id"]
        == "logistic_regression"
    )
    assert (
        artifact_summary["display_name"]
        == "Logistic Regression"
    )

    assert (
        artifact_summary["tuning_applied"]
        is True
    )

    assert artifact_summary["tuned_parameters"] == {
        "C": 0.5,
        "class_weight": None,
    }

    artifact_path = Path(
        artifact_summary["artifact_path"]
    )

    metadata_path = Path(
        artifact_summary["metadata_path"]
    )

    assert artifact_path.exists()
    assert metadata_path.exists()

    assert artifact_path.parent == model_directory
    assert metadata_path.parent == model_directory

    assert artifact_path.suffix == ".joblib"
    assert metadata_path.suffix == ".json"

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        metadata["model_id"]
        == "logistic_regression"
    )
    assert metadata["tuning_applied"] is True
    assert metadata["tuning_result"]["best_params"] == {
        "C": 0.5,
        "class_weight": None,
    }
    assert metadata["artifact_environment"]["schema_version"] == 1
    assert metadata["artifact_environment"]["python_version"]
    assert metadata["artifact_environment"]["package_versions"][
        "scikit-learn"
    ]

    assert (
        feature_importance["status"]
        == "available"
    )

    assert (
        feature_importance["importance_type"]
        == "linear_absolute_coefficient"
    )

    assert feature_importance["top_features"]

    completed_events = [
        event
        for event in updated_state.tool_history
        if (
            event.get("tool_name")
            == "model_persistence_tool"
            and event.get("status")
            == "completed"
        )
    ]

    assert completed_events

    assert (
        "Saved final fitted pipeline"
        in completed_events[-1]["message"]
    )
