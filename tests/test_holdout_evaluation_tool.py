from types import SimpleNamespace

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge

import src.tools.holdout_evaluation_tool as holdout_module
from src.state import ExperimentState
from src.tools.holdout_evaluation_tool import (
    holdout_evaluation_tool,
)

def test_holdout_evaluation_is_skipped_when_no_holdout_reserved(
    monkeypatch,
) -> None:
    """
    When baseline training did not reserve a final holdout, the holdout
    evaluation step must complete successfully with an honest skipped result.

    The tool must not load the dataset or train a model.
    """

    state = ExperimentState(
        dataset_path="unused.csv",
        target_column="Target",
        user_objective="Test skipped holdout evaluation",
    )

    state.status = "tuning_completed"

    state.validation_summary = {
        "holdout_used": False,
        "training_rows": 10,
        "holdout_rows": 0,
    }

    def fail_if_dataset_is_loaded(
        file_path,
        target_column,
    ):
        raise AssertionError(
            "The dataset must not be loaded when no holdout was reserved."
        )

    monkeypatch.setattr(
        holdout_module,
        "load_dataset",
        fail_if_dataset_is_loaded,
    )

    result = holdout_evaluation_tool(
        state=state,
        random_state=42,
    )

    assert result["success"] is True
    assert result["error"] is None

    updated_state = result["state"]
    holdout_result = result["holdout_result"]

    # State transition
    assert (
        updated_state.status
        == "holdout_evaluation_completed"
    )

    assert (
        "checked_final_holdout"
        in updated_state.completed_steps
    )

    # Result contents
    assert holdout_result["status"] == "skipped"
    assert holdout_result["holdout_used"] is False

    assert (
        holdout_result["passes_holdout_guardrail"]
        is None
    )

    assert holdout_result["model_id"] is None
    assert holdout_result["display_name"] is None
    assert holdout_result["metrics"] == {}

    assert holdout_result["reason"] == (
        "Holdout evaluation was skipped because no final holdout "
        "was reserved for this dataset."
    )

    # State must store the returned result.
    assert updated_state.holdout_result == holdout_result

    # Tool history
    started_events = [
        event
        for event in updated_state.tool_history
        if (
            event.get("tool_name")
            == "holdout_evaluation_tool"
            and event.get("status") == "started"
        )
    ]

    completed_events = [
        event
        for event in updated_state.tool_history
        if (
            event.get("tool_name")
            == "holdout_evaluation_tool"
            and event.get("status") == "completed"
        )
    ]

    assert started_events
    assert completed_events

    assert (
        "no final holdout was reserved"
        in completed_events[-1]["message"]
    )

def test_holdout_evaluation_skips_when_no_useful_candidate(
    monkeypatch,
) -> None:
    """
    A reserved holdout must not be evaluated when model comparison found no
    useful non-dummy candidate.

    This is a valid analytical outcome, so the tool should succeed while
    recording that the holdout guardrail was not passed.
    """

    state = ExperimentState(
        dataset_path="unused.csv",
        target_column="Target",
        user_objective="Test holdout with no useful candidate",
    )

    state.status = "tuning_completed"

    state.validation_summary = {
        "holdout_used": True,
        "training_rows": 240,
        "holdout_rows": 60,
        "requested_holdout_fraction": 0.20,
        "minimum_rows_for_holdout": 200,
        "minimum_class_count_for_holdout": 10,
        "random_state": 42,
    }

    state.comparison_summary = {
        "best_overall_model_id": "dummy_classifier",
        "best_overall_display_name": "Dummy Classifier",
        "best_useful_model_id": None,
        "best_useful_display_name": None,
        "primary_metric": "f1",
    }

    def fail_if_dataset_is_loaded(
        file_path,
        target_column,
    ):
        raise AssertionError(
            "The dataset must not be loaded when no useful candidate exists."
        )

    monkeypatch.setattr(
        holdout_module,
        "load_dataset",
        fail_if_dataset_is_loaded,
    )

    result = holdout_evaluation_tool(
        state=state,
        random_state=42,
    )

    assert result["success"] is True
    assert result["error"] is None

    updated_state = result["state"]
    holdout_result = result["holdout_result"]

    # State transition
    assert (
        updated_state.status
        == "holdout_evaluation_completed"
    )

    assert (
        "checked_final_holdout"
        in updated_state.completed_steps
    )

    # Result contents
    assert (
        holdout_result["status"]
        == "skipped_no_candidate"
    )

    assert holdout_result["holdout_used"] is True

    assert (
        holdout_result["passes_holdout_guardrail"]
        is False
    )

    assert holdout_result["model_id"] is None
    assert holdout_result["display_name"] is None
    assert holdout_result["metrics"] == {}

    assert holdout_result["reason"] == (
        "A holdout was reserved, but no useful non-dummy candidate "
        "was available for final evaluation."
    )

    # State must store the returned result.
    assert updated_state.holdout_result == holdout_result

    # Tool history
    completed_events = [
        event
        for event in updated_state.tool_history
        if (
            event.get("tool_name")
            == "holdout_evaluation_tool"
            and event.get("status") == "completed"
        )
    ]

    assert completed_events

    assert (
        "no useful non-dummy candidate"
        in completed_events[-1]["message"]
    )


def test_classification_holdout_passes_guardrail_with_tuned_model(
    monkeypatch,
) -> None:
    """
    An approved classification candidate should be fitted on the training
    portion and evaluated once on the untouched holdout.

    Accepted tuned parameters must be applied, and strong holdout performance
    must pass the current F1 guardrail.
    """

    dataset = pd.DataFrame(
        {
            "feature_a": [
                0.00,
                0.10,
                0.20,
                0.30,
                1.00,
                1.10,
                1.20,
                1.30,
                0.15,
                0.25,
                1.15,
                1.25,
            ],
            "feature_b": [
                0.05,
                0.15,
                0.25,
                0.35,
                1.05,
                1.15,
                1.25,
                1.35,
                0.20,
                0.30,
                1.20,
                1.30,
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
                0,
                0,
                1,
                1,
            ],
        }
    )

    feature_columns = [
        "feature_a",
        "feature_b",
    ]

    X = dataset[feature_columns].copy()
    y = dataset["Target"].copy()

    X_train = X.iloc[:8].copy()
    y_train = y.iloc[:8].copy()

    X_holdout = X.iloc[8:].copy()
    y_holdout = y.iloc[8:].copy()

    validation_split = SimpleNamespace(
        X_train=X_train,
        y_train=y_train,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        summary={
            "holdout_used": True,
        },
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                feature_columns,
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
        holdout_module,
        "load_dataset",
        lambda file_path, target_column: SimpleNamespace(
            df=dataset
        ),
    )

    monkeypatch.setattr(
        holdout_module,
        "prepare_features_and_target",
        lambda df, preprocessing_config: (
            X,
            y,
        ),
    )

    monkeypatch.setattr(
        holdout_module,
        "create_validation_split",
        lambda **kwargs: validation_split,
    )

    monkeypatch.setattr(
        holdout_module,
        "build_preprocessor",
        lambda preprocessing_config: preprocessor,
    )

    monkeypatch.setattr(
        holdout_module,
        "get_model_by_id",
        lambda base_task, model_id, random_state: (
            candidate_model
        ),
    )

    original_classification_metrics = (
        holdout_module._classification_metrics
    )

    inspected_model = {}

    def inspect_tuned_model_and_calculate_metrics(
        fitted_pipeline,
        X_holdout,
        y_holdout,
    ):
        """
        Verify that accepted tuned parameters reached the fitted estimator,
        then run the real classification metric calculation.
        """

        fitted_model = fitted_pipeline.named_steps["model"]

        inspected_model["C"] = fitted_model.C
        inspected_model["class_weight"] = (
            fitted_model.class_weight
        )

        return original_classification_metrics(
            fitted_pipeline=fitted_pipeline,
            X_holdout=X_holdout,
            y_holdout=y_holdout,
        )

    monkeypatch.setattr(
        holdout_module,
        "_classification_metrics",
        inspect_tuned_model_and_calculate_metrics,
    )

    state = ExperimentState(
        dataset_path="classification_holdout_test.csv",
        target_column="Target",
        user_objective="Test classification holdout guardrail",
    )

    state.status = "tuning_completed"
    state.task_type = "binary_classification"
    state.base_task = "classification"

    state.preprocessing_config = {
        "numerical_features": feature_columns,
        "categorical_features": [],
    }

    state.validation_summary = {
        "holdout_used": True,
        "training_rows": 8,
        "holdout_rows": 4,
        "requested_holdout_fraction": 0.20,
        "minimum_rows_for_holdout": 200,
        "minimum_class_count_for_holdout": 10,
        "random_state": 42,
    }

    state.comparison_summary = {
        "best_useful_model_id": "logistic_regression",
        "best_useful_display_name": "Logistic Regression",
        "primary_metric": "f1",
    }

    state.leaderboard = [
        {
            "rank": 1,
            "model_id": "logistic_regression",
            "display_name": "Logistic Regression",
            "primary_metric": "f1",
            "primary_score": 0.85,
            "status": "completed",
        }
    ]

    state.tuning_result = {
        "status": "completed",
        "model_id": "logistic_regression",
        "accepted_for_final_evaluation": True,
        "baseline_cv_score": 0.85,
        "tuned_cv_score": 0.90,
        "best_params": {
            "C": 10.0,
            "class_weight": None,
        },
    }

    result = holdout_evaluation_tool(
        state=state,
        random_state=42,
    )

    assert result["success"] is True
    assert result["error"] is None

    updated_state = result["state"]
    holdout_result = result["holdout_result"]

    # State transition
    assert (
        updated_state.status
        == "holdout_evaluation_completed"
    )

    assert (
        "checked_final_holdout"
        in updated_state.completed_steps
    )

    # Completed evaluation
    assert holdout_result["status"] == "completed"
    assert holdout_result["holdout_used"] is True

    assert (
        holdout_result["model_id"]
        == "logistic_regression"
    )

    assert (
        holdout_result["display_name"]
        == "Logistic Regression"
    )

    assert holdout_result["primary_metric"] == "f1"
    assert holdout_result["cv_primary_score"] == 0.90

    assert (
        holdout_result["passes_holdout_guardrail"]
        is True
    )

    assert holdout_result["training_rows"] == 8
    assert holdout_result["holdout_rows"] == 4

    # Real classification metrics
    assert holdout_result["holdout_primary_score"] == 1.0
    assert holdout_result["metrics"]["accuracy"] == 1.0
    assert holdout_result["metrics"]["f1"] == 1.0
    assert holdout_result["metrics"]["precision"] == 1.0
    assert holdout_result["metrics"]["recall"] == 1.0

    assert holdout_result["labels"] == [
        "0",
        "1",
    ]

    assert holdout_result["positive_label"] == "1"

    assert holdout_result["confusion_matrix"] == [
        [2, 0],
        [0, 2],
    ]

    # Guardrail: minimum allowed F1 = 0.90 - 0.10 = 0.80.
    assert (
        "minimum permitted value"
        in holdout_result["guardrail_reason"]
    )

    assert "0.8000" in holdout_result["guardrail_reason"]

    # Tuned parameters
    assert (
        holdout_result["used_tuned_parameters"]
        is True
    )

    assert holdout_result["tuned_parameters"] == {
        "C": 10.0,
        "class_weight": None,
    }

    assert holdout_result["baseline_cv_score"] == 0.85

    assert inspected_model["C"] == 10.0
    assert inspected_model["class_weight"] is None

    # State stores the result.
    assert updated_state.holdout_result == holdout_result

    # Tool history
    completed_events = [
        event
        for event in updated_state.tool_history
        if (
            event.get("tool_name")
            == "holdout_evaluation_tool"
            and event.get("status") == "completed"
        )
    ]

    assert completed_events

    completed_message = completed_events[-1]["message"]

    assert "Logistic Regression" in completed_message
    assert "4 untouched holdout row(s)" in completed_message
    assert "Guardrail passed: True" in completed_message

def test_classification_holdout_fails_guardrail_when_performance_drops(
    monkeypatch,
) -> None:
    """
    A completed holdout evaluation must fail the reliability guardrail when
    holdout F1 drops by more than 0.10 compared with the baseline CV score.

    The tool itself should still succeed because poor model performance is
    an analytical result rather than an execution failure.

    Tuned parameters that were not accepted must not be applied.
    """

    dataset = pd.DataFrame(
        {
            "feature_a": [
                0.00,
                0.10,
                0.20,
                0.30,
                1.00,
                1.10,
                1.20,
                1.30,
                0.15,
                0.25,
                1.15,
                1.25,
            ],
            "feature_b": [
                0.05,
                0.15,
                0.25,
                0.35,
                1.05,
                1.15,
                1.25,
                1.35,
                0.20,
                0.30,
                1.20,
                1.30,
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
                0,
                0,
                1,
                1,
            ],
        }
    )

    feature_columns = [
        "feature_a",
        "feature_b",
    ]

    X = dataset[feature_columns].copy()
    y = dataset["Target"].copy()

    X_train = X.iloc[:8].copy()
    y_train = y.iloc[:8].copy()

    X_holdout = X.iloc[8:].copy()
    y_holdout = y.iloc[8:].copy()

    validation_split = SimpleNamespace(
        X_train=X_train,
        y_train=y_train,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        summary={
            "holdout_used": True,
        },
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                feature_columns,
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
        holdout_module,
        "load_dataset",
        lambda file_path, target_column: SimpleNamespace(
            df=dataset
        ),
    )

    monkeypatch.setattr(
        holdout_module,
        "prepare_features_and_target",
        lambda df, preprocessing_config: (
            X,
            y,
        ),
    )

    monkeypatch.setattr(
        holdout_module,
        "create_validation_split",
        lambda **kwargs: validation_split,
    )

    monkeypatch.setattr(
        holdout_module,
        "build_preprocessor",
        lambda preprocessing_config: preprocessor,
    )

    monkeypatch.setattr(
        holdout_module,
        "get_model_by_id",
        lambda base_task, model_id, random_state: (
            candidate_model
        ),
    )

    inspected_model = {}

    def return_weak_holdout_metrics(
        fitted_pipeline,
        X_holdout,
        y_holdout,
    ):
        """
        Return deterministic weak performance so the guardrail must fail.

        Also inspect the estimator to confirm that rejected tuned parameters
        were not applied.
        """

        fitted_model = fitted_pipeline.named_steps["model"]

        inspected_model["C"] = fitted_model.C
        inspected_model["class_weight"] = (
            fitted_model.class_weight
        )

        return {
            "metrics": {
                "accuracy": 0.50,
                "balanced_accuracy": 0.50,
                "precision": 0.50,
                "recall": 0.50,
                "f1": 0.50,
            },
            "labels": [
                "0",
                "1",
            ],
            "positive_label": "1",
            "confusion_matrix": [
                [1, 1],
                [1, 1],
            ],
        }

    monkeypatch.setattr(
        holdout_module,
        "_classification_metrics",
        return_weak_holdout_metrics,
    )

    state = ExperimentState(
        dataset_path="failed_holdout_test.csv",
        target_column="Target",
        user_objective=(
            "Test classification holdout guardrail failure"
        ),
    )

    state.status = "tuning_completed"
    state.task_type = "binary_classification"
    state.base_task = "classification"

    state.preprocessing_config = {
        "numerical_features": feature_columns,
        "categorical_features": [],
    }

    state.validation_summary = {
        "holdout_used": True,
        "training_rows": 8,
        "holdout_rows": 4,
        "requested_holdout_fraction": 0.20,
        "minimum_rows_for_holdout": 200,
        "minimum_class_count_for_holdout": 10,
        "random_state": 42,
    }

    state.comparison_summary = {
        "best_useful_model_id": "logistic_regression",
        "best_useful_display_name": "Logistic Regression",
        "primary_metric": "f1",
    }

    state.leaderboard = [
        {
            "rank": 1,
            "model_id": "logistic_regression",
            "display_name": "Logistic Regression",
            "primary_metric": "f1",
            "primary_score": 0.90,
            "status": "completed",
        }
    ]

    state.tuning_result = {
        "status": "completed",
        "model_id": "logistic_regression",
        "accepted_for_final_evaluation": False,
        "baseline_cv_score": 0.90,
        "tuned_cv_score": 0.70,
        "best_params": {
            "C": 0.01,
            "class_weight": "balanced",
        },
    }

    result = holdout_evaluation_tool(
        state=state,
        random_state=42,
    )

    assert result["success"] is True
    assert result["error"] is None

    updated_state = result["state"]
    holdout_result = result["holdout_result"]

    # The evaluation completed successfully.
    assert (
        updated_state.status
        == "holdout_evaluation_completed"
    )

    assert (
        "checked_final_holdout"
        in updated_state.completed_steps
    )

    assert holdout_result["status"] == "completed"
    assert holdout_result["holdout_used"] is True

    assert (
        holdout_result["model_id"]
        == "logistic_regression"
    )

    assert (
        holdout_result["display_name"]
        == "Logistic Regression"
    )

    # The baseline remains the comparison score because tuning was rejected.
    assert holdout_result["primary_metric"] == "f1"
    assert holdout_result["cv_primary_score"] == 0.90
    assert holdout_result["holdout_primary_score"] == 0.50

    # Guardrail: minimum permitted F1 = 0.90 - 0.10 = 0.80.
    assert (
        holdout_result["passes_holdout_guardrail"]
        is False
    )

    assert (
        "minimum permitted value"
        in holdout_result["guardrail_reason"]
    )

    assert "0.8000" in holdout_result["guardrail_reason"]
    assert "0.5000" in holdout_result["guardrail_reason"]

    # Weak metrics are preserved for the critic and final report.
    assert holdout_result["metrics"]["accuracy"] == 0.50
    assert holdout_result["metrics"]["f1"] == 0.50
    assert holdout_result["metrics"]["precision"] == 0.50
    assert holdout_result["metrics"]["recall"] == 0.50

    assert holdout_result["confusion_matrix"] == [
        [1, 1],
        [1, 1],
    ]

    assert holdout_result["training_rows"] == 8
    assert holdout_result["holdout_rows"] == 4

    # Rejected tuned parameters must not be used.
    assert (
        holdout_result["used_tuned_parameters"]
        is False
    )

    assert holdout_result["tuned_parameters"] == {}
    assert holdout_result["baseline_cv_score"] == 0.90

    # The original Logistic Regression defaults remain in place.
    assert inspected_model["C"] == 1.0
    assert inspected_model["class_weight"] is None

    # The state stores the failed guardrail result.
    assert updated_state.holdout_result == holdout_result

    completed_events = [
        event
        for event in updated_state.tool_history
        if (
            event.get("tool_name")
            == "holdout_evaluation_tool"
            and event.get("status") == "completed"
        )
    ]

    assert completed_events

    completed_message = completed_events[-1]["message"]

    assert "Logistic Regression" in completed_message
    assert "4 untouched holdout row(s)" in completed_message
    assert "Guardrail passed: False" in completed_message

def test_regression_holdout_passes_lower_is_better_guardrail(
    monkeypatch,
) -> None:
    """
    Regression error metrics use a lower-is-better guardrail.

    A holdout RMSE within 20% of the cross-validation RMSE should pass.
    The tool must preserve the regression metrics and complete successfully.
    """

    dataset = pd.DataFrame(
        {
            "feature_a": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
            ],
            "feature_b": [
                2.0,
                4.0,
                6.0,
                8.0,
                10.0,
                12.0,
                14.0,
                16.0,
                18.0,
                20.0,
                22.0,
                24.0,
            ],
            "Target": [
                110.0,
                205.0,
                315.0,
                395.0,
                510.0,
                605.0,
                710.0,
                795.0,
                905.0,
                1010.0,
                1095.0,
                1210.0,
            ],
        }
    )

    feature_columns = [
        "feature_a",
        "feature_b",
    ]

    X = dataset[feature_columns].copy()
    y = dataset["Target"].copy()

    X_train = X.iloc[:8].copy()
    y_train = y.iloc[:8].copy()

    X_holdout = X.iloc[8:].copy()
    y_holdout = y.iloc[8:].copy()

    validation_split = SimpleNamespace(
        X_train=X_train,
        y_train=y_train,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        summary={
            "holdout_used": True,
        },
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                feature_columns,
            ),
        ],
        remainder="drop",
    )

    candidate_model = SimpleNamespace(
        estimator=Ridge(
            alpha=1.0,
        )
    )

    monkeypatch.setattr(
        holdout_module,
        "load_dataset",
        lambda file_path, target_column: SimpleNamespace(
            df=dataset
        ),
    )

    monkeypatch.setattr(
        holdout_module,
        "prepare_features_and_target",
        lambda df, preprocessing_config: (
            X,
            y,
        ),
    )

    monkeypatch.setattr(
        holdout_module,
        "create_validation_split",
        lambda **kwargs: validation_split,
    )

    monkeypatch.setattr(
        holdout_module,
        "build_preprocessor",
        lambda preprocessing_config: preprocessor,
    )

    monkeypatch.setattr(
        holdout_module,
        "get_model_by_id",
        lambda base_task, model_id, random_state: (
            candidate_model
        ),
    )

    inspected_model = {}

    def return_regression_metrics(
        fitted_pipeline,
        X_holdout,
        y_holdout,
    ):
        """
        Return deterministic regression metrics.

        CV RMSE is 100. The permitted maximum holdout RMSE is therefore
        120. A holdout RMSE of 115 should pass.
        """

        fitted_model = fitted_pipeline.named_steps["model"]
        inspected_model["alpha"] = fitted_model.alpha

        return {
            "metrics": {
                "mae": 90.0,
                "mse": 13225.0,
                "rmse": 115.0,
                "r2": 0.72,
            }
        }

    monkeypatch.setattr(
        holdout_module,
        "_regression_metrics",
        return_regression_metrics,
    )

    state = ExperimentState(
        dataset_path="regression_holdout_test.csv",
        target_column="Target",
        user_objective="Test regression holdout guardrail",
    )

    state.status = "tuning_completed"
    state.task_type = "regression"
    state.base_task = "regression"

    state.preprocessing_config = {
        "numerical_features": feature_columns,
        "categorical_features": [],
    }

    state.validation_summary = {
        "holdout_used": True,
        "training_rows": 8,
        "holdout_rows": 4,
        "requested_holdout_fraction": 0.20,
        "minimum_rows_for_holdout": 200,
        "minimum_class_count_for_holdout": 10,
        "random_state": 42,
    }

    state.comparison_summary = {
        "best_useful_model_id": "ridge_regression",
        "best_useful_display_name": "Ridge Regression",
        "primary_metric": "rmse",
    }

    state.leaderboard = [
        {
            "rank": 1,
            "model_id": "ridge_regression",
            "display_name": "Ridge Regression",
            "primary_metric": "rmse",
            "primary_score": 100.0,
            "status": "completed",
        }
    ]

    state.tuning_result = {
        "status": "completed",
        "model_id": "ridge_regression",
        "accepted_for_final_evaluation": False,
        "baseline_cv_score": 100.0,
        "tuned_cv_score": 125.0,
        "best_params": {
            "alpha": 0.01,
        },
    }

    result = holdout_evaluation_tool(
        state=state,
        random_state=42,
    )

    assert result["success"] is True
    assert result["error"] is None

    updated_state = result["state"]
    holdout_result = result["holdout_result"]

    # State transition
    assert (
        updated_state.status
        == "holdout_evaluation_completed"
    )

    assert (
        "checked_final_holdout"
        in updated_state.completed_steps
    )

    # Completed regression evaluation
    assert holdout_result["status"] == "completed"
    assert holdout_result["holdout_used"] is True

    assert (
        holdout_result["model_id"]
        == "ridge_regression"
    )

    assert (
        holdout_result["display_name"]
        == "Ridge Regression"
    )

    assert holdout_result["primary_metric"] == "rmse"

    # Rejected tuning means the original baseline score is used.
    assert holdout_result["cv_primary_score"] == 100.0
    assert holdout_result["holdout_primary_score"] == 115.0

    # Maximum permitted RMSE = 100 + 20% = 120.
    assert (
        holdout_result["passes_holdout_guardrail"]
        is True
    )

    assert (
        "maximum permitted value"
        in holdout_result["guardrail_reason"]
    )

    assert "115.0000" in holdout_result["guardrail_reason"]
    assert "120.0000" in holdout_result["guardrail_reason"]

    # Regression metrics are preserved.
    assert holdout_result["metrics"] == {
        "mae": 90.0,
        "mse": 13225.0,
        "rmse": 115.0,
        "r2": 0.72,
    }

    assert holdout_result["training_rows"] == 8
    assert holdout_result["holdout_rows"] == 4

    # Rejected tuned parameters must not be applied.
    assert (
        holdout_result["used_tuned_parameters"]
        is False
    )

    assert holdout_result["tuned_parameters"] == {}
    assert holdout_result["baseline_cv_score"] == 100.0

    # Ridge default remains because tuning was rejected.
    assert inspected_model["alpha"] == 1.0

    # Regression results do not contain classification fields.
    assert holdout_result["labels"] == []
    assert holdout_result["positive_label"] is None
    assert holdout_result["confusion_matrix"] == []

    assert updated_state.holdout_result == holdout_result

    completed_events = [
        event
        for event in updated_state.tool_history
        if (
            event.get("tool_name")
            == "holdout_evaluation_tool"
            and event.get("status") == "completed"
        )
    ]

    assert completed_events

    completed_message = completed_events[-1]["message"]

    assert "Ridge Regression" in completed_message
    assert "4 untouched holdout row(s)" in completed_message
    assert "Guardrail passed: True" in completed_message