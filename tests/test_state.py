import json
from pathlib import Path

import pytest

from src.state import (
    ExperimentState,
    load_state,
    save_state,
    state_to_dict,
)

def test_experiment_state_round_trip_preserves_all_fields(
    tmp_path: Path,
) -> None:
    """
    Saving and reloading an ExperimentState must preserve its complete
    experiment evidence.

    The test also confirms that save_state creates missing parent
    directories automatically.
    """

    state = ExperimentState(
        dataset_path="data/sample/holdout_demo.csv",
        target_column="Approved",
        user_objective=(
            "Predict whether a loan application should be approved"
        ),
    )

    state.task_type = "binary_classification"
    state.base_task = "classification"
    state.task_reason = (
        "The target contains two distinct categorical values."
    )
    state.target_unique_count = 2
    state.target_missing_count = 0

    state.target_summary = {
        "unique_values": 2,
        "classes": [
            "No",
            "Yes",
        ],
    }

    state.profile = {
        "rows": 500,
        "columns": 7,
        "feature_count": 6,
    }

    state.quality_issues = [
        {
            "severity": "medium",
            "issue": "Missing values detected",
        }
    ]

    state.leakage_warnings = []

    state.approved_actions = [
        "drop_possible_id_columns",
    ]

    state.pending_approvals = []

    state.experiment_plan = [
        {
            "title": "Train baseline models",
            "status": "completed",
        }
    ]

    state.preprocessing_config = {
        "columns_to_drop": [
            "Loan_ID",
        ],
        "numerical_features": [
            "ApplicantIncome",
            "LoanAmount",
        ],
        "categorical_features": [
            "Property_Area",
        ],
    }

    state.preprocessing_pipeline_summary = {
        "status": "created",
        "numerical_feature_count": 2,
        "categorical_feature_count": 1,
    }

    state.model_registry_summary = [
        {
            "model_id": "logistic_regression",
            "display_name": "Logistic Regression",
        }
    ]

    state.model_results = [
        {
            "model_id": "logistic_regression",
            "status": "completed",
            "primary_score": 0.82,
        }
    ]

    state.training_summary = {
        "models_completed": 4,
        "models_failed": 0,
        "primary_metric": "f1",
    }

    state.leaderboard = [
        {
            "rank": 1,
            "model_id": "logistic_regression",
            "primary_score": 0.82,
        }
    ]

    state.comparison_summary = {
        "best_overall_model_id": "logistic_regression",
        "best_useful_model_id": "logistic_regression",
        "primary_metric": "f1",
    }

    state.validation_summary = {
        "holdout_used": True,
        "cross_validation_rows": 400,
        "holdout_rows": 100,
    }

    state.tuning_summary = {
        "eligible": True,
        "status": "completed",
    }

    state.tuning_result = {
        "status": "completed",
        "model_id": "logistic_regression",
        "best_params": {
            "C": 0.5,
            "class_weight": None,
        },
        "accepted_for_final_evaluation": True,
    }

    state.holdout_result = {
        "status": "completed",
        "model_id": "logistic_regression",
        "primary_metric": "f1",
        "cv_primary_score": 0.82,
        "holdout_primary_score": 0.81,
        "passes_holdout_guardrail": True,
    }

    state.critic_report = {
        "overall_reliability": "high",
        "recommendation_decision": (
            "recommend_candidate_model"
        ),
        "can_proceed_to_tuning": True,
    }

    state.model_artifact_summary = {
        "status": "saved",
        "model_id": "logistic_regression",
        "artifact_path": (
            "outputs/models/"
            "logistic_regression_pipeline.joblib"
        ),
        "metadata_path": (
            "outputs/models/"
            "logistic_regression_metadata.json"
        ),
        "tuning_applied": True,
    }

    state.feature_importance_summary = {
        "status": "available",
        "importance_type": (
            "linear_absolute_coefficient"
        ),
        "top_features": [
            {
                "rank": 1,
                "feature": "ApplicantIncome",
                "importance": 1.25,
            }
        ],
    }

    state.final_report_path = (
        "outputs/reports/final_report.md"
    )

    state.final_report_summary = {
        "report_type": "final_automl_advisor_report",
        "recommendation_decision": (
            "recommend_candidate_model"
        ),
    }

    state.completed_steps = [
        "created_experiment_state",
        "completed_guarded_tuning",
        "checked_final_holdout",
        "created_critic_report",
        "checked_model_persistence",
        "created_final_report",
    ]

    state.tool_history = [
        {
            "timestamp": "2026-07-24T18:42:59",
            "tool_name": "final_report_tool",
            "status": "completed",
            "message": "Final report created.",
        }
    ]

    state.warnings = [
        "Synthetic warning retained for round-trip testing.",
    ]

    state.iteration = 12
    state.status = "final_report_created"

    output_path = (
        tmp_path
        / "nested"
        / "run"
        / "experiment_state.json"
    )

    # The nested parent directories do not exist yet.
    assert output_path.parent.exists() is False

    save_state(
        state=state,
        output_path=str(output_path),
    )

    # save_state must create the parent directories and JSON file.
    assert output_path.parent.exists() is True
    assert output_path.exists() is True

    raw_data = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    # The JSON content must match the dataclass conversion.
    assert raw_data == state_to_dict(state)

    loaded_state = load_state(
        str(output_path)
    )

    assert isinstance(
        loaded_state,
        ExperimentState,
    )

    # Dataclass equality compares every field.
    assert loaded_state == state

    # Explicitly protect the most important nested evidence.
    assert (
        loaded_state.dataset_path
        == "data/sample/holdout_demo.csv"
    )

    assert loaded_state.target_column == "Approved"

    assert (
        loaded_state.tuning_result["best_params"]
        == {
            "C": 0.5,
            "class_weight": None,
        }
    )

    assert (
        loaded_state.holdout_result[
            "passes_holdout_guardrail"
        ]
        is True
    )

    assert (
        loaded_state.critic_report[
            "recommendation_decision"
        ]
        == "recommend_candidate_model"
    )

    assert (
        loaded_state.model_artifact_summary[
            "status"
        ]
        == "saved"
    )

    assert (
        loaded_state.status
        == "final_report_created"
    )

    assert loaded_state.iteration == 12

def test_load_state_accepts_minimal_json_and_applies_defaults(
    tmp_path: Path,
) -> None:
    """
    A state file containing only the required constructor fields should load
    successfully and receive the dataclass defaults for optional fields.
    """

    input_path = tmp_path / "minimal_state.json"

    minimal_data = {
        "dataset_path": "data/minimal.csv",
        "target_column": "Target",
    }

    input_path.write_text(
        json.dumps(
            minimal_data,
            indent=4,
        ),
        encoding="utf-8",
    )

    loaded_state = load_state(
        str(input_path)
    )

    assert (
        loaded_state.dataset_path
        == "data/minimal.csv"
    )

    assert loaded_state.target_column == "Target"
    assert loaded_state.user_objective is None

    assert loaded_state.task_type is None
    assert loaded_state.base_task is None
    assert loaded_state.task_reason is None

    assert loaded_state.profile == {}
    assert loaded_state.quality_issues == []
    assert loaded_state.leakage_warnings == []

    assert loaded_state.completed_steps == []
    assert loaded_state.tool_history == []

    assert loaded_state.training_summary == {}
    assert loaded_state.leaderboard == []
    assert loaded_state.critic_report == {}

    assert loaded_state.warnings == []
    assert loaded_state.iteration == 0
    assert loaded_state.status == "created"

def test_experiment_states_do_not_share_mutable_defaults() -> None:
    """
    Separate ExperimentState objects must not share dictionaries or lists.

    Updating one experiment must never alter another experiment's memory.
    """

    first_state = ExperimentState(
        dataset_path="first.csv",
        target_column="First_Target",
    )

    second_state = ExperimentState(
        dataset_path="second.csv",
        target_column="Second_Target",
    )

    first_state.quality_issues.append(
        {
            "severity": "high",
            "issue": "Synthetic issue",
        }
    )

    first_state.completed_steps.append(
        "synthetic_step"
    )

    first_state.preprocessing_config[
        "numerical_features"
    ] = [
        "Feature_A",
    ]

    first_state.tuning_result[
        "status"
    ] = "completed"

    first_state.tool_history.append(
        {
            "tool_name": "synthetic_tool",
            "status": "completed",
        }
    )

    assert second_state.quality_issues == []
    assert second_state.completed_steps == []
    assert second_state.preprocessing_config == {}
    assert second_state.tuning_result == {}
    assert second_state.tool_history == []

    assert (
        first_state.quality_issues
        is not second_state.quality_issues
    )

    assert (
        first_state.preprocessing_config
        is not second_state.preprocessing_config
    )

    assert (
        first_state.completed_steps
        is not second_state.completed_steps
    )

def test_state_to_dict_returns_detached_nested_data() -> None:
    """
    Mutating the dictionary returned by state_to_dict must not mutate the
    original ExperimentState.

    This protects the live experiment memory when serialized data is edited.
    """

    state = ExperimentState(
        dataset_path="detached.csv",
        target_column="Target",
    )

    state.profile = {
        "rows": 100,
        "nested": {
            "feature_count": 4,
        },
    }

    state.completed_steps = [
        "created_experiment_state",
    ]

    state.tuning_result = {
        "status": "completed",
        "best_params": {
            "C": 0.5,
        },
    }

    converted = state_to_dict(state)

    converted["profile"]["rows"] = 999

    converted["profile"]["nested"][
        "feature_count"
    ] = 50

    converted["completed_steps"].append(
        "modified_outside_state"
    )

    converted["tuning_result"][
        "best_params"
    ]["C"] = 100.0

    assert state.profile["rows"] == 100

    assert (
        state.profile["nested"]["feature_count"]
        == 4
    )

    assert state.completed_steps == [
        "created_experiment_state",
    ]

    assert (
        state.tuning_result["best_params"]["C"]
        == 0.5
    )

def test_separate_state_files_do_not_overwrite_each_other(
    tmp_path: Path,
) -> None:
    """
    Two experiments saved to different paths must remain independent.

    Saving the second experiment must not alter the first experiment's JSON.
    """

    first_state = ExperimentState(
        dataset_path="data/first_run.csv",
        target_column="First_Target",
        user_objective="First experiment",
    )

    first_state.status = "final_report_created"
    first_state.critic_report = {
        "recommendation_decision": (
            "recommend_candidate_model"
        ),
    }

    second_state = ExperimentState(
        dataset_path="data/second_run.csv",
        target_column="Second_Target",
        user_objective="Second experiment",
    )

    second_state.status = "reliability_critique_completed"
    second_state.critic_report = {
        "recommendation_decision": (
            "do_not_recommend_model"
        ),
    }

    first_path = (
        tmp_path
        / "run_001"
        / "experiment_state.json"
    )

    second_path = (
        tmp_path
        / "run_002"
        / "experiment_state.json"
    )

    save_state(
        first_state,
        str(first_path),
    )

    first_file_before_second_save = (
        first_path.read_text(
            encoding="utf-8"
        )
    )

    save_state(
        second_state,
        str(second_path),
    )

    first_file_after_second_save = (
        first_path.read_text(
            encoding="utf-8"
        )
    )

    assert first_path.exists()
    assert second_path.exists()
    assert first_path != second_path

    assert (
        first_file_after_second_save
        == first_file_before_second_save
    )

    loaded_first = load_state(
        str(first_path)
    )

    loaded_second = load_state(
        str(second_path)
    )

    assert (
        loaded_first.dataset_path
        == "data/first_run.csv"
    )

    assert (
        loaded_first.target_column
        == "First_Target"
    )

    assert (
        loaded_first.critic_report[
            "recommendation_decision"
        ]
        == "recommend_candidate_model"
    )

    assert (
        loaded_second.dataset_path
        == "data/second_run.csv"
    )

    assert (
        loaded_second.target_column
        == "Second_Target"
    )

    assert (
        loaded_second.critic_report[
            "recommendation_decision"
        ]
        == "do_not_recommend_model"
    )

def test_separate_state_files_do_not_overwrite_each_other(
    tmp_path: Path,
) -> None:
    """
    Two experiments saved to different paths must remain independent.

    Saving the second experiment must not alter the first experiment's JSON.
    """

    first_state = ExperimentState(
        dataset_path="data/first_run.csv",
        target_column="First_Target",
        user_objective="First experiment",
    )

    first_state.status = "final_report_created"
    first_state.critic_report = {
        "recommendation_decision": (
            "recommend_candidate_model"
        ),
    }

    second_state = ExperimentState(
        dataset_path="data/second_run.csv",
        target_column="Second_Target",
        user_objective="Second experiment",
    )

    second_state.status = "reliability_critique_completed"
    second_state.critic_report = {
        "recommendation_decision": (
            "do_not_recommend_model"
        ),
    }

    first_path = (
        tmp_path
        / "run_001"
        / "experiment_state.json"
    )

    second_path = (
        tmp_path
        / "run_002"
        / "experiment_state.json"
    )

    save_state(
        first_state,
        str(first_path),
    )

    first_file_before_second_save = (
        first_path.read_text(
            encoding="utf-8"
        )
    )

    save_state(
        second_state,
        str(second_path),
    )

    first_file_after_second_save = (
        first_path.read_text(
            encoding="utf-8"
        )
    )

    assert first_path.exists()
    assert second_path.exists()
    assert first_path != second_path

    assert (
        first_file_after_second_save
        == first_file_before_second_save
    )

    loaded_first = load_state(
        str(first_path)
    )

    loaded_second = load_state(
        str(second_path)
    )

    assert (
        loaded_first.dataset_path
        == "data/first_run.csv"
    )

    assert (
        loaded_first.target_column
        == "First_Target"
    )

    assert (
        loaded_first.critic_report[
            "recommendation_decision"
        ]
        == "recommend_candidate_model"
    )

    assert (
        loaded_second.dataset_path
        == "data/second_run.csv"
    )

    assert (
        loaded_second.target_column
        == "Second_Target"
    )

    assert (
        loaded_second.critic_report[
            "recommendation_decision"
        ]
        == "do_not_recommend_model"
    )

def test_load_state_rejects_malformed_json(
    tmp_path: Path,
) -> None:
    """
    A corrupted state file must raise a JSON decoding error rather than
    producing a partially loaded experiment.
    """

    input_path = (
        tmp_path
        / "malformed_state.json"
    )

    input_path.write_text(
        """
        {
            "dataset_path": "broken.csv",
            "target_column": "Target",
        """,
        encoding="utf-8",
    )

    with pytest.raises(
        json.JSONDecodeError,
    ):
        load_state(
            str(input_path)
        )

def test_load_state_rejects_missing_required_field(
    tmp_path: Path,
) -> None:
    """
    A state file without the required target_column field must not load.
    """

    input_path = (
        tmp_path
        / "missing_required_field.json"
    )

    incomplete_data = {
        "dataset_path": "incomplete.csv",
    }

    input_path.write_text(
        json.dumps(
            incomplete_data,
            indent=4,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        TypeError,
        match="target_column",
    ):
        load_state(
            str(input_path)
        )

def test_load_state_rejects_unknown_schema_field(
    tmp_path: Path,
) -> None:
    """
    Unexpected state fields must not be silently ignored.

    Failing clearly makes state-schema incompatibility visible.
    """

    input_path = (
        tmp_path
        / "unknown_field_state.json"
    )

    incompatible_data = {
        "dataset_path": "incompatible.csv",
        "target_column": "Target",
        "unexpected_experiment_field": (
            "This field is not part of ExperimentState."
        ),
    }

    input_path.write_text(
        json.dumps(
            incompatible_data,
            indent=4,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        TypeError,
        match="unexpected_experiment_field",
    ):
        load_state(
            str(input_path)
        )