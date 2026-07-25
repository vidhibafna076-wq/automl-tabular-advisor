from types import SimpleNamespace

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import src.tools.tuning_tool as tuning_module
from src.state import ExperimentState
from src.tools.tuning_tool import tuning_tool


def test_ineligible_tuning_is_skipped_successfully(
    monkeypatch,
) -> None:
    """
    An experiment that does not satisfy the tuning policy must be skipped
    safely rather than treated as a workflow failure.
    """

    state = ExperimentState(
        dataset_path="unused.csv",
        target_column="Target",
        user_objective="Test guarded tuning skip",
    )

    state.status = "model_comparison_completed"

    eligibility_result = {
        "eligible": False,
        "status": "ineligible",
        "reason": (
            "Guarded tuning is not appropriate for this experiment."
        ),
        "blocking_reasons": [
            "The dataset has fewer than 200 rows.",
            "No final holdout was reserved.",
            "No useful non-dummy candidate model was selected.",
        ],
        "candidate_model_id": None,
        "candidate_display_name": None,
        "candidate_primary_score": None,
        "max_trials": 12,
        "timeout_seconds": 180,
    }

    monkeypatch.setattr(
        tuning_module,
        "assess_tuning_eligibility",
        lambda state, max_trials, timeout_seconds: (
            eligibility_result
        ),
    )

    result = tuning_tool(
        state=state,
        max_trials=12,
        timeout_seconds=180,
        random_state=42,
    )

    assert result["success"] is True
    assert result["error"] is None

    updated_state = result["state"]
    tuning_summary = result["tuning_summary"]
    tuning_result = result["tuning_result"]

    # State transition
    assert updated_state.status == "tuning_completed"

    assert (
        "checked_guarded_tuning"
        in updated_state.completed_steps
    )

    # Eligibility summary
    assert tuning_summary["eligible"] is False

    assert tuning_summary["blocking_reasons"] == [
        "The dataset has fewer than 200 rows.",
        "No final holdout was reserved.",
        "No useful non-dummy candidate model was selected.",
    ]

    # Tuning result
    assert tuning_result["status"] == "skipped"

    assert tuning_result["reason"] == (
        "Guarded tuning is not appropriate for this experiment."
    )

    assert tuning_result["blocking_reasons"] == [
        "The dataset has fewer than 200 rows.",
        "No final holdout was reserved.",
        "No useful non-dummy candidate model was selected.",
    ]

    assert tuning_result["model_id"] is None
    assert tuning_result["display_name"] is None
    assert tuning_result["best_params"] == {}
    assert tuning_result["baseline_cv_score"] is None
    assert tuning_result["tuned_cv_score"] is None

    assert (
        tuning_result["accepted_for_final_evaluation"]
        is False
    )

    assert (
        tuning_result["holdout_used_during_tuning"]
        is False
    )

    # State stores the same result returned by the tool.
    assert updated_state.tuning_summary == tuning_summary
    assert updated_state.tuning_result == tuning_result

    # Tool history
    started_events = [
        event
        for event in updated_state.tool_history
        if (
            event.get("tool_name") == "tuning_tool"
            and event.get("status") == "started"
        )
    ]

    completed_events = [
        event
        for event in updated_state.tool_history
        if (
            event.get("tool_name") == "tuning_tool"
            and event.get("status") == "completed"
        )
    ]

    assert started_events
    assert completed_events

    completed_message = completed_events[-1]["message"]

    assert "Guarded tuning was skipped" in completed_message
    assert "fewer than 200 rows" in completed_message
    assert "No final holdout was reserved" in completed_message


def test_eligible_tuning_completes_and_is_accepted(
    monkeypatch,
) -> None:
    """
    An eligible candidate should complete guarded tuning using only the
    training portion.

    The tuned result should be accepted when its cross-validation score is
    better than the selected baseline candidate's score.
    """

    dataset = pd.DataFrame(
        {
            "feature_a": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
            ],
            "feature_b": [
                1.0,
                1.1,
                1.2,
                1.3,
                1.4,
                2.0,
                2.1,
                2.2,
                2.3,
                2.4,
            ],
            "Target": [
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
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

    eligibility_result = {
        "eligible": True,
        "status": "eligible",
        "reason": (
            "The experiment satisfies the guarded tuning policy."
        ),
        "blocking_reasons": [],
        "candidate_model_id": "logistic_regression",
        "candidate_display_name": "Logistic Regression",
        "candidate_primary_score": 0.80,
        "max_trials": 2,
        "timeout_seconds": 30,
    }

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
        tuning_module,
        "assess_tuning_eligibility",
        lambda state, max_trials, timeout_seconds: (
            eligibility_result
        ),
    )

    monkeypatch.setattr(
        tuning_module,
        "load_dataset",
        lambda file_path, target_column: SimpleNamespace(
            df=dataset
        ),
    )

    monkeypatch.setattr(
        tuning_module,
        "prepare_features_and_target",
        lambda df, preprocessing_config: (
            X,
            y,
        ),
    )

    monkeypatch.setattr(
        tuning_module,
        "create_validation_split",
        lambda **kwargs: validation_split,
    )

    monkeypatch.setattr(
        tuning_module,
        "build_preprocessor",
        lambda preprocessing_config: preprocessor,
    )

    monkeypatch.setattr(
        tuning_module,
        "get_model_by_id",
        lambda base_task, model_id, random_state: (
            candidate_model
        ),
    )

    monkeypatch.setattr(
        tuning_module,
        "build_scoring_strategy",
        lambda task_type, y: {
            "f1": "test_f1_scorer",
        },
    )

    monkeypatch.setattr(
        tuning_module,
        "build_cv_strategy",
        lambda
            base_task,
            preprocessing_config,
            y,
            random_state: "test_cv_strategy",
    )

    evaluated_training_indices = []

    def fake_calculate_cv_score(
        pipeline,
        X_train,
        y_train,
        scorer,
        cv,
        primary_metric,
    ) -> float:
        """
        Return a deterministic score better than the baseline.

        Also record the rows received so the test can prove that holdout
        rows were not passed into tuning.
        """

        evaluated_training_indices.extend(
            list(X_train.index)
        )

        assert scorer == "test_f1_scorer"
        assert cv == "test_cv_strategy"
        assert primary_metric == "f1"

        return 0.85

    monkeypatch.setattr(
        tuning_module,
        "_calculate_cv_score",
        fake_calculate_cv_score,
    )

    state = ExperimentState(
        dataset_path="eligible_tuning_test.csv",
        target_column="Target",
        user_objective="Test successful guarded tuning",
    )

    state.status = "model_comparison_completed"
    state.task_type = "binary_classification"
    state.base_task = "classification"

    state.preprocessing_config = {
        "numerical_features": feature_columns,
        "categorical_features": [],
    }

    state.validation_summary = {
        "holdout_used": True,
        "requested_holdout_fraction": 0.20,
        "minimum_rows_for_holdout": 200,
        "minimum_class_count_for_holdout": 10,
        "random_state": 42,
    }

    state.training_summary = {
        "primary_metric": "f1",
    }

    state.leaderboard = [
        {
            "rank": 1,
            "model_id": "logistic_regression",
            "display_name": "Logistic Regression",
            "primary_metric": "f1",
            "primary_score": 0.80,
            "status": "completed",
        }
    ]

    result = tuning_tool(
        state=state,
        max_trials=2,
        timeout_seconds=30,
        random_state=42,
    )

    assert result["success"] is True
    assert result["error"] is None

    updated_state = result["state"]
    tuning_summary = result["tuning_summary"]
    tuning_result = result["tuning_result"]

    # State transition
    assert updated_state.status == "tuning_completed"

    assert (
        "completed_guarded_tuning"
        in updated_state.completed_steps
    )

    # Completed tuning result
    assert tuning_result["status"] == "completed"

    assert (
        tuning_result["model_id"]
        == "logistic_regression"
    )

    assert (
        tuning_result["display_name"]
        == "Logistic Regression"
    )

    assert tuning_result["primary_metric"] == "f1"
    assert tuning_result["metric_direction"] == "maximize"

    assert tuning_result["baseline_cv_score"] == 0.80
    assert tuning_result["tuned_cv_score"] == 0.85

    assert (
        tuning_result["improvement"]
        == pytest.approx(0.05)
    )

    assert (
        tuning_result["accepted_for_final_evaluation"]
        is True
    )

    assert (
        tuning_result["holdout_used_during_tuning"]
        is False
    )

    assert tuning_result["trials_requested"] == 2
    assert tuning_result["trials_completed"] == 2

    assert tuning_result["training_rows"] == 8
    assert tuning_result["reserved_holdout_rows"] == 2

    assert tuning_result["best_params"] == {
        "C": 1.0,
        "class_weight": None,
    }

    # The CV evaluator must only receive training indices 0 through 7.
    assert evaluated_training_indices

    assert set(evaluated_training_indices).issubset(
        set(X_train.index)
    )

    assert set(evaluated_training_indices).isdisjoint(
        set(X_holdout.index)
    )

    # Tuning summary
    assert tuning_summary["eligible"] is True
    assert tuning_summary["status"] == "completed"
    assert tuning_summary["trials_completed"] == 2

    # Tool history
    completed_events = [
        event
        for event in updated_state.tool_history
        if (
            event.get("tool_name") == "tuning_tool"
            and event.get("status") == "completed"
        )
    ]

    assert completed_events

    completed_message = completed_events[-1]["message"]

    assert "Guarded tuning completed" in completed_message
    assert "Logistic Regression" in completed_message
    assert "2 successful trial(s)" in completed_message

    assert (
        "accepted for final holdout evaluation"
        in completed_message
    )

def test_completed_tuning_is_rejected_when_worse_than_baseline(
    monkeypatch,
) -> None:
    """
    Guarded tuning may complete successfully without its parameters being
    accepted.

    When the tuned cross-validation score is worse than the baseline score,
    the result must remain completed but must not be approved for final
    holdout evaluation.
    """

    dataset = pd.DataFrame(
        {
            "feature_a": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
            ],
            "feature_b": [
                1.0,
                1.1,
                1.2,
                1.3,
                1.4,
                2.0,
                2.1,
                2.2,
                2.3,
                2.4,
            ],
            "Target": [
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
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

    eligibility_result = {
        "eligible": True,
        "status": "eligible",
        "reason": (
            "The experiment satisfies the guarded tuning policy."
        ),
        "blocking_reasons": [],
        "candidate_model_id": "logistic_regression",
        "candidate_display_name": "Logistic Regression",
        "candidate_primary_score": 0.80,
        "max_trials": 1,
        "timeout_seconds": 30,
    }

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
        tuning_module,
        "assess_tuning_eligibility",
        lambda state, max_trials, timeout_seconds: (
            eligibility_result
        ),
    )

    monkeypatch.setattr(
        tuning_module,
        "load_dataset",
        lambda file_path, target_column: SimpleNamespace(
            df=dataset
        ),
    )

    monkeypatch.setattr(
        tuning_module,
        "prepare_features_and_target",
        lambda df, preprocessing_config: (
            X,
            y,
        ),
    )

    monkeypatch.setattr(
        tuning_module,
        "create_validation_split",
        lambda **kwargs: validation_split,
    )

    monkeypatch.setattr(
        tuning_module,
        "build_preprocessor",
        lambda preprocessing_config: preprocessor,
    )

    monkeypatch.setattr(
        tuning_module,
        "get_model_by_id",
        lambda base_task, model_id, random_state: (
            candidate_model
        ),
    )

    monkeypatch.setattr(
        tuning_module,
        "build_scoring_strategy",
        lambda task_type, y: {
            "f1": "test_f1_scorer",
        },
    )

    monkeypatch.setattr(
        tuning_module,
        "build_cv_strategy",
        lambda base_task, preprocessing_config, y, random_state: (
            "test_cv_strategy"
        ),
    )

    evaluated_training_indices = []

    def fake_calculate_cv_score(
        pipeline,
        X_train,
        y_train,
        scorer,
        cv,
        primary_metric,
    ) -> float:
        """
        Return a deterministic score below the baseline.

        The received indices are recorded to confirm that reserved holdout
        rows were never passed to the tuning evaluator.
        """

        evaluated_training_indices.extend(
            list(X_train.index)
        )

        assert scorer == "test_f1_scorer"
        assert cv == "test_cv_strategy"
        assert primary_metric == "f1"

        return 0.75

    monkeypatch.setattr(
        tuning_module,
        "_calculate_cv_score",
        fake_calculate_cv_score,
    )

    state = ExperimentState(
        dataset_path="worse_tuning_test.csv",
        target_column="Target",
        user_objective=(
            "Test rejection of worse tuned parameters"
        ),
    )

    state.status = "model_comparison_completed"
    state.task_type = "binary_classification"
    state.base_task = "classification"

    state.preprocessing_config = {
        "numerical_features": feature_columns,
        "categorical_features": [],
    }

    state.validation_summary = {
        "holdout_used": True,
        "requested_holdout_fraction": 0.20,
        "minimum_rows_for_holdout": 200,
        "minimum_class_count_for_holdout": 10,
        "random_state": 42,
    }

    state.training_summary = {
        "primary_metric": "f1",
    }

    state.leaderboard = [
        {
            "rank": 1,
            "model_id": "logistic_regression",
            "display_name": "Logistic Regression",
            "primary_metric": "f1",
            "primary_score": 0.80,
            "status": "completed",
        }
    ]

    result = tuning_tool(
        state=state,
        max_trials=1,
        timeout_seconds=30,
        random_state=42,
    )

    assert result["success"] is True
    assert result["error"] is None

    updated_state = result["state"]
    tuning_result = result["tuning_result"]
    tuning_summary = result["tuning_summary"]

    assert updated_state.status == "tuning_completed"

    assert (
        "completed_guarded_tuning"
        in updated_state.completed_steps
    )

    assert tuning_result["status"] == "completed"

    assert (
        tuning_result["model_id"]
        == "logistic_regression"
    )

    assert (
        tuning_result["display_name"]
        == "Logistic Regression"
    )

    assert tuning_result["primary_metric"] == "f1"
    assert tuning_result["metric_direction"] == "maximize"

    assert tuning_result["baseline_cv_score"] == 0.80
    assert tuning_result["tuned_cv_score"] == 0.75

    assert tuning_result["improvement"] == pytest.approx(
        -0.05
    )

    assert (
        tuning_result["accepted_for_final_evaluation"]
        is False
    )

    assert (
        tuning_result["holdout_used_during_tuning"]
        is False
    )

    assert tuning_result["trials_requested"] == 1
    assert tuning_result["trials_completed"] == 1

    assert tuning_result["training_rows"] == 8
    assert tuning_result["reserved_holdout_rows"] == 2

    assert tuning_result["best_params"] == {
        "C": 1.0,
        "class_weight": None,
    }

    # Prove that tuning received training rows only.
    assert evaluated_training_indices

    assert set(evaluated_training_indices).issubset(
        set(X_train.index)
    )

    assert set(evaluated_training_indices).isdisjoint(
        set(X_holdout.index)
    )

    assert tuning_summary["eligible"] is True
    assert tuning_summary["status"] == "completed"
    assert tuning_summary["trials_completed"] == 1

    completed_events = [
        event
        for event in updated_state.tool_history
        if (
            event.get("tool_name") == "tuning_tool"
            and event.get("status") == "completed"
        )
    ]

    assert completed_events

    completed_message = completed_events[-1]["message"]

    assert "Guarded tuning completed" in completed_message

    assert (
        "not accepted because it did not improve on the baseline"
        in completed_message
    )