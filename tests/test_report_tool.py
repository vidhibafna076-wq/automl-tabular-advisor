from pathlib import Path

import src.tools.report_tool as report_module
from src.state import ExperimentState
from src.tools.report_tool import final_report_tool

def test_final_report_is_created_with_completion_event(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """
    A valid completed experiment should produce a final report.

    The final_report_tool completion event must exist before Markdown
    rendering so the report can include its own completion event.
    """

    state = ExperimentState(
        dataset_path="approved_dataset.csv",
        target_column="Approved",
        user_objective="Test final report generation",
    )

    state.status = "model_persistence_checked"

    state.critic_report = {
        "overall_reliability": "high",
        "recommendation_decision": (
            "recommend_candidate_model"
        ),
        "can_proceed_to_tuning": True,
        "selected_candidate": {
            "model_id": "logistic_regression",
            "display_name": "Logistic Regression",
            "primary_metric": "f1",
            "primary_score": 0.82,
            "reliability_level": "high",
        },
        "finding_counts": {
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 1,
        },
        "findings": [],
        "next_actions": [],
    }

    state.model_artifact_summary = {
        "status": "saved",
        "critic_decision": "recommend_candidate_model",
        "model_id": "logistic_regression",
        "display_name": "Logistic Regression",
        "artifact_path": "temporary_model.joblib",
        "metadata_path": "temporary_metadata.json",
        "tuning_applied": True,
        "tuned_parameters": {
            "C": 0.5,
        },
    }

    expected_summary = {
        "report_type": "final_automl_advisor_report",
        "dataset_path": "approved_dataset.csv",
        "target_column": "Approved",
        "recommendation_decision": (
            "recommend_candidate_model"
        ),
    }

    rendering_evidence = {}

    def fake_create_summary(
        received_state: ExperimentState,
    ) -> dict:
        assert (
            received_state.status
            == "final_report_created"
        )

        assert (
            "created_final_report"
            in received_state.completed_steps
        )

        return expected_summary

    def fake_create_markdown(
        received_state: ExperimentState,
    ) -> str:
        """
        Confirm the completion event already exists when Markdown is rendered.
        """

        completed_events = [
            event
            for event in received_state.tool_history
            if (
                event.get("tool_name")
                == "final_report_tool"
                and event.get("status")
                == "completed"
            )
        ]

        rendering_evidence[
            "completion_event_present"
        ] = bool(completed_events)

        rendering_evidence[
            "status_during_render"
        ] = received_state.status

        rendering_evidence[
            "summary_during_render"
        ] = received_state.final_report_summary

        return (
            "# Final AutoML Advisor Report\n\n"
            "Recommendation: recommend_candidate_model\n"
        )

    def fake_save_report(
        markdown_text: str,
        output_path: str,
    ) -> str:
        resolved_path = Path(output_path)

        resolved_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        resolved_path.write_text(
            markdown_text,
            encoding="utf-8",
        )

        return str(resolved_path)

    monkeypatch.setattr(
        report_module,
        "create_final_report_summary",
        fake_create_summary,
    )

    monkeypatch.setattr(
        report_module,
        "create_final_report_markdown",
        fake_create_markdown,
    )

    monkeypatch.setattr(
        report_module,
        "save_final_report",
        fake_save_report,
    )

    output_path = (
        tmp_path
        / "reports"
        / "final_report.md"
    )

    result = final_report_tool(
        state=state,
        output_path=str(output_path),
    )

    assert result["success"] is True
    assert result["error"] is None

    updated_state = result["state"]

    assert (
        updated_state.status
        == "final_report_created"
    )

    assert (
        "created_final_report"
        in updated_state.completed_steps
    )

    assert result["final_report_path"] == str(
        output_path
    )

    assert updated_state.final_report_path == str(
        output_path
    )

    assert (
        result["final_report_summary"]
        == expected_summary
    )

    assert (
        updated_state.final_report_summary
        == expected_summary
    )

    assert (
        result["report_markdown"]
        == (
            "# Final AutoML Advisor Report\n\n"
            "Recommendation: recommend_candidate_model\n"
        )
    )

    assert output_path.exists()

    assert output_path.read_text(
        encoding="utf-8"
    ) == result["report_markdown"]

    # Confirm ordering during report rendering.
    assert (
        rendering_evidence[
            "completion_event_present"
        ]
        is True
    )

    assert (
        rendering_evidence["status_during_render"]
        == "final_report_created"
    )

    assert (
        rendering_evidence["summary_during_render"]
        == expected_summary
    )

    completed_events = [
        event
        for event in updated_state.tool_history
        if (
            event.get("tool_name")
            == "final_report_tool"
            and event.get("status")
            == "completed"
        )
    ]

    assert len(completed_events) == 1

    assert (
        f"Final report saved to {output_path}."
        == completed_events[0]["message"]
    )

def test_rejected_model_report_is_created_without_saved_artifact_claim(
    tmp_path: Path,
) -> None:
    """
    A do-not-recommend decision is a valid completed experiment.

    The final report must:
    - still be created;
    - explain that no model should be recommended;
    - show persistence as skipped;
    - include the persistence reason;
    - avoid claiming that an artifact or metadata file was saved;
    - include the final_report_tool completion event.
    """

    state = ExperimentState(
        dataset_path="data/sample/loan_sample.csv",
        target_column="Loan_Status",
        user_objective=(
            "Predict whether a loan application will be approved"
        ),
    )

    state.status = "model_persistence_checked"
    state.task_type = "binary_classification"
    state.base_task = "classification"
    state.task_reason = (
        "The target contains two distinct categorical values."
    )

    state.target_summary = {
        "unique_values": 2,
        "classes": [
            "N",
            "Y",
        ],
    }

    state.profile = {
        "rows": 10,
        "columns": 8,
        "feature_count": 7,
        "target_column": "Loan_Status",
        "numerical_columns": [
            "ApplicantIncome",
            "LoanAmount",
            "Credit_History",
        ],
        "categorical_columns": [
            "Loan_ID",
            "Gender",
            "Married",
            "Property_Area",
        ],
    }

    state.quality_issues = []

    state.experiment_plan = []

    state.preprocessing_config = {
        "columns_to_drop": [
            "Loan_ID",
        ],
        "numerical_features": [
            "ApplicantIncome",
            "LoanAmount",
            "Credit_History",
        ],
        "categorical_features": [
            "Gender",
            "Married",
            "Property_Area",
        ],
        "missing_value_strategy": {
            "numerical": {
                "strategy": "median",
            },
            "categorical": {
                "strategy": "most_frequent",
            },
        },
    }

    state.training_summary = {
        "task_type": "binary_classification",
        "validation_method": "cross_validation",
        "cv_folds": 2,
        "primary_metric": "f1",
        "models_attempted": 4,
        "models_completed": 4,
        "models_failed": 0,
        "leakage_control": (
            "Preprocessing was fitted inside each cross-validation fold."
        ),
        "unsupported_metrics_in_first_trainer": [],
    }

    state.validation_summary = {
        "strategy": "cross_validation_only",
        "cross_validation_used": True,
        "cross_validation_rows": 10,
        "holdout_used": False,
        "holdout_rows": 0,
        "reason": (
            "The dataset was too small to reserve a final holdout."
        ),
    }

    state.model_results = []

    state.comparison_summary = {
        "primary_metric": "f1",
        "dummy_baseline_score": 0.67,
        "best_overall_model_id": "dummy_classifier",
        "best_overall_display_name": "Dummy Classifier",
        "best_useful_model_id": None,
        "best_useful_display_name": None,
        "recommendation_status": (
            "no_model_clearly_beats_dummy"
        ),
        "important_notes": [
            (
                "No non-dummy candidate clearly outperformed "
                "the dummy baseline."
            ),
        ],
    }

    state.leaderboard = []

    state.critic_report = {
        "overall_reliability": "low",
        "recommendation_decision": (
            "do_not_recommend_model"
        ),
        "can_proceed_to_tuning": False,
        "selected_candidate": None,
        "finding_counts": {
            "high": 2,
            "medium": 0,
            "low": 0,
            "info": 1,
        },
        "findings": [
            {
                "severity": "high",
                "issue": (
                    "No useful model clearly beats the dummy baseline"
                ),
                "evidence": (
                    "The comparison summary did not identify "
                    "a useful non-dummy candidate."
                ),
                "recommendation": (
                    "Do not recommend a final model from this run."
                ),
            },
        ],
        "next_actions": [
            (
                "Do not select or persist a final model "
                "from this run."
            ),
            (
                "Use a larger and more representative dataset."
            ),
        ],
    }

    persistence_reason = (
        "No model was saved because the critic determined that "
        "the current evidence does not justify recommending a model."
    )

    state.model_artifact_summary = {
        "status": "skipped",
        "reason": persistence_reason,
        "critic_decision": "do_not_recommend_model",
        "model_id": None,
        "display_name": None,
        "tuning_applied": False,
        "tuned_parameters": {},
        "artifact_path": None,
        "metadata_path": None,
    }

    state.feature_importance_summary = {
        "status": "skipped",
        "reason": (
            "Feature importance was skipped because "
            "no final model was saved."
        ),
        "top_features": [],
    }

    output_path = (
        tmp_path
        / "rejected_run"
        / "final_report.md"
    )

    result = final_report_tool(
        state=state,
        output_path=str(output_path),
    )

    assert result["success"] is True
    assert result["error"] is None

    updated_state = result["state"]
    report_markdown = result["report_markdown"]

    # State and physical report
    assert (
        updated_state.status
        == "final_report_created"
    )

    assert (
        "created_final_report"
        in updated_state.completed_steps
    )

    assert output_path.exists()

    assert (
        output_path.read_text(encoding="utf-8")
        == report_markdown
    )

    # Current run identity
    assert (
        "**Dataset:** `data/sample/loan_sample.csv`"
        in report_markdown
    )

    assert (
        "**Target column:** `Loan_Status`"
        in report_markdown
    )

    # Honest rejection conclusion
    assert (
        "The final recommendation decision is: "
        "**do_not_recommend_model**."
        in report_markdown
    )

    assert (
        "No final model should be recommended from this run."
        in report_markdown
    )

    assert (
        "The AutoML advisor should not recommend "
        "a final model from this run."
        in report_markdown
    )

    # Persistence is explicitly shown as skipped.
    assert (
        "- **Status:** skipped"
        in report_markdown
    )

    assert persistence_reason in report_markdown

    assert (
        "- **Critic decision:** do_not_recommend_model"
        in report_markdown
    )

    # Feature importance is also honestly skipped.
    assert (
        "Feature importance status: **skipped**."
        in report_markdown
    )

    assert (
        "no final model was saved"
        in report_markdown
    )

    # No artifact paths may be claimed for a rejected model.
    assert "**Artifact path:**" not in report_markdown
    assert "**Metadata path:**" not in report_markdown
    assert ".joblib" not in report_markdown

    # The completion event must appear inside the generated report.
    assert (
        "**[COMPLETED]** final_report_tool"
        in report_markdown
    )

    assert (
        "Final report saved to"
        in report_markdown
    )

    # Report summary belongs to this same run.
    report_summary = result["final_report_summary"]

    assert (
        report_summary["dataset_path"]
        == "data/sample/loan_sample.csv"
    )

    assert (
        report_summary["target_column"]
        == "Loan_Status"
    )

    assert (
        report_summary["overall_reliability"]
        == "low"
    )

    assert (
        report_summary["recommendation_decision"]
        == "do_not_recommend_model"
    )

    assert (
        report_summary["best_overall_model"]
        == "Dummy Classifier"
    )

    assert report_summary["best_useful_model"] is None

    assert (
        report_summary["final_status"]
        == "final_report_created"
    )

def test_approved_model_report_includes_artifacts_tuning_and_importance(
    tmp_path: Path,
) -> None:
    """
    An approved experiment report must preserve evidence that:

    - the critic recommended a candidate;
    - guarded tuning was completed and accepted;
    - the untouched holdout passed;
    - the fitted pipeline and metadata were saved;
    - feature importance was extracted;
    - the report belongs to the current dataset and target.
    """

    state = ExperimentState(
        dataset_path="data/sample/holdout_demo.csv",
        target_column="Approved",
        user_objective=(
            "Predict whether a loan application should be approved"
        ),
    )

    state.status = "model_persistence_checked"
    state.task_type = "binary_classification"
    state.base_task = "classification"
    state.task_reason = (
        "The target contains two distinct categorical values."
    )

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
        "target_column": "Approved",
        "numerical_columns": [
            "ApplicantIncome",
            "LoanAmount",
            "Credit_History",
            "Age",
        ],
        "categorical_columns": [
            "Employment_Type",
            "Property_Area",
        ],
    }

    state.quality_issues = []

    state.experiment_plan = []

    state.preprocessing_config = {
        "columns_to_drop": [],
        "numerical_features": [
            "ApplicantIncome",
            "LoanAmount",
            "Credit_History",
            "Age",
        ],
        "categorical_features": [
            "Employment_Type",
            "Property_Area",
        ],
        "missing_value_strategy": {
            "numerical": {
                "strategy": "median",
            },
            "categorical": {
                "strategy": "most_frequent",
            },
        },
    }

    state.training_summary = {
        "task_type": "binary_classification",
        "validation_method": "cross_validation_with_holdout",
        "cv_folds": 5,
        "primary_metric": "f1",
        "models_attempted": 4,
        "models_completed": 4,
        "models_failed": 0,
        "leakage_control": (
            "Preprocessing was fitted inside each cross-validation fold."
        ),
        "unsupported_metrics_in_first_trainer": [],
    }

    state.validation_summary = {
        "strategy": "cross_validation_with_final_holdout",
        "cross_validation_used": True,
        "cross_validation_rows": 400,
        "holdout_used": True,
        "holdout_rows": 100,
        "reason": (
            "The dataset contained enough rows to reserve "
            "an untouched final holdout."
        ),
    }

    state.tuning_summary = {
        "eligible": True,
        "reason": (
            "The experiment satisfied the guarded tuning policy."
        ),
        "blocking_reasons": [],
        "status": "completed",
        "trials_completed": 12,
        "duration_seconds": 2.5,
    }

    state.tuning_result = {
        "status": "completed",
        "model_id": "logistic_regression",
        "display_name": "Logistic Regression",
        "primary_metric": "f1",
        "baseline_cv_score": 0.80,
        "tuned_cv_score": 0.82,
        "improvement": 0.02,
        "accepted_for_final_evaluation": True,
        "trials_completed": 12,
        "duration_seconds": 2.5,
        "holdout_used_during_tuning": False,
        "best_params": {
            "C": 0.5,
            "class_weight": None,
        },
    }

    state.holdout_result = {
        "status": "completed",
        "holdout_used": True,
        "reason": (
            "The selected useful candidate was fitted on the training "
            "portion and evaluated once on the untouched holdout."
        ),
        "model_id": "logistic_regression",
        "display_name": "Logistic Regression",
        "training_rows": 400,
        "holdout_rows": 100,
        "primary_metric": "f1",
        "cv_primary_score": 0.82,
        "holdout_primary_score": 0.81,
        "passes_holdout_guardrail": True,
        "guardrail_reason": (
            "Holdout f1 was 0.8100. The minimum permitted value "
            "under the current guardrail was 0.7200."
        ),
        "metrics": {
            "accuracy": 0.83,
            "balanced_accuracy": 0.82,
            "precision": 0.84,
            "recall": 0.79,
            "f1": 0.81,
        },
    }

    state.model_results = []

    state.comparison_summary = {
        "primary_metric": "f1",
        "dummy_baseline_score": 0.60,
        "best_overall_model_id": "logistic_regression",
        "best_overall_display_name": "Logistic Regression",
        "best_useful_model_id": "logistic_regression",
        "best_useful_display_name": "Logistic Regression",
        "recommendation_status": "useful_model_selected",
        "important_notes": [
            (
                "Logistic Regression clearly outperformed "
                "the dummy baseline."
            ),
        ],
    }

    state.leaderboard = [
        {
            "rank": 1,
            "model_id": "logistic_regression",
            "display_name": "Logistic Regression",
            "primary_score": 0.82,
            "improvement_over_dummy": {
                "beats_dummy": True,
            },
            "reliability_level": "high",
            "overfit_gap": 0.03,
            "reliability_flags": [],
        },
        {
            "rank": 2,
            "model_id": "dummy_classifier",
            "display_name": "Dummy Classifier",
            "primary_score": 0.60,
            "improvement_over_dummy": {
                "beats_dummy": False,
            },
            "reliability_level": "baseline",
            "overfit_gap": 0.0,
            "reliability_flags": [],
        },
    ]

    state.critic_report = {
        "overall_reliability": "high",
        "recommendation_decision": (
            "recommend_candidate_model"
        ),
        "can_proceed_to_tuning": True,
        "selected_candidate": {
            "model_id": "logistic_regression",
            "display_name": "Logistic Regression",
            "primary_metric": "f1",
            "primary_score": 0.82,
            "reliability_level": "high",
        },
        "finding_counts": {
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 1,
        },
        "findings": [
            {
                "severity": "info",
                "issue": "Final holdout evidence passed",
                "evidence": (
                    "Logistic Regression was evaluated on "
                    "100 untouched rows."
                ),
                "recommendation": (
                    "Treat Logistic Regression as the current candidate."
                ),
            },
        ],
        "next_actions": [
            (
                "Proceed with the selected candidate as "
                "the current recommended model."
            ),
            (
                "Save the critic-approved fitted pipeline "
                "and its metadata."
            ),
        ],
    }

    artifact_path = (
        "outputs/models/"
        "approved_logistic_regression_pipeline.joblib"
    )

    metadata_path = (
        "outputs/models/"
        "approved_logistic_regression_metadata.json"
    )

    state.model_artifact_summary = {
        "status": "saved",
        "reason": (
            "Final fitted pipeline was saved because the critic "
            "approved the candidate model."
        ),
        "critic_decision": "recommend_candidate_model",
        "model_id": "logistic_regression",
        "display_name": "Logistic Regression",
        "tuning_applied": True,
        "tuned_parameters": {
            "C": 0.5,
            "class_weight": None,
        },
        "artifact_path": artifact_path,
        "metadata_path": metadata_path,
    }

    state.feature_importance_summary = {
        "status": "available",
        "importance_type": "linear_absolute_coefficient",
        "top_features": [
            {
                "rank": 1,
                "feature": "numerical__Credit_History",
                "importance": 1.42,
            },
            {
                "rank": 2,
                "feature": "numerical__ApplicantIncome",
                "importance": 0.86,
            },
        ],
    }

    output_path = (
        tmp_path
        / "approved_run"
        / "final_report.md"
    )

    result = final_report_tool(
        state=state,
        output_path=str(output_path),
    )

    assert result["success"] is True
    assert result["error"] is None

    updated_state = result["state"]
    report_markdown = result["report_markdown"]
    report_summary = result["final_report_summary"]

    # State and physical report
    assert (
        updated_state.status
        == "final_report_created"
    )

    assert (
        "created_final_report"
        in updated_state.completed_steps
    )

    assert output_path.exists()

    assert (
        output_path.read_text(encoding="utf-8")
        == report_markdown
    )

    # Current run identity
    assert (
        "**Dataset:** `data/sample/holdout_demo.csv`"
        in report_markdown
    )

    assert (
        "**Target column:** `Approved`"
        in report_markdown
    )

    # Approved critic decision
    assert (
        "The final recommendation decision is: "
        "**recommend_candidate_model**."
        in report_markdown
    )

    assert (
        "A candidate model can be recommended based on "
        "the current experiment evidence."
        in report_markdown
    )

    assert (
        "- **Overall reliability:** high"
        in report_markdown
    )

    assert (
        "- **Recommendation decision:** "
        "recommend_candidate_model"
        in report_markdown
    )

    assert (
        "- **Model:** Logistic Regression"
        in report_markdown
    )

    # Guarded tuning evidence
    assert (
        "- **Tuning status:** completed"
        in report_markdown
    )

    assert (
        "- **Accepted for final evaluation:** True"
        in report_markdown
    )

    assert (
        "- **Holdout used during tuning:** False"
        in report_markdown
    )

    assert "### Best Parameters" in report_markdown
    assert "- **C:** 0.5" in report_markdown
    assert "- **class_weight:** None" in report_markdown

    # Final untouched holdout evidence
    assert (
        "- **Passed holdout guardrail:** True"
        in report_markdown
    )

    assert (
        "- **Training rows:** 400"
        in report_markdown
    )

    assert (
        "- **Holdout rows:** 100"
        in report_markdown
    )

    assert (
        "| f1 | 0.81 |"
        in report_markdown
    )

    # Saved artifact evidence
    assert (
        "- **Status:** saved"
        in report_markdown
    )

    assert (
        "- **Critic decision:** recommend_candidate_model"
        in report_markdown
    )

    assert (
        f"- **Artifact path:** `{artifact_path}`"
        in report_markdown
    )

    assert (
        f"- **Metadata path:** `{metadata_path}`"
        in report_markdown
    )

    # Feature importance evidence
    assert (
        "- **Importance type:** "
        "linear_absolute_coefficient"
        in report_markdown
    )

    assert (
        "| 1 | numerical__Credit_History | 1.42 |"
        in report_markdown
    )

    assert (
        "| 2 | numerical__ApplicantIncome | 0.86 |"
        in report_markdown
    )

    # The report contains its own completion event.
    assert (
        "**[COMPLETED]** final_report_tool"
        in report_markdown
    )

    assert (
        "Final report saved to"
        in report_markdown
    )

    # Summary belongs to this approved run.
    assert (
        report_summary["dataset_path"]
        == "data/sample/holdout_demo.csv"
    )

    assert (
        report_summary["target_column"]
        == "Approved"
    )

    assert (
        report_summary["task_type"]
        == "binary_classification"
    )

    assert (
        report_summary["overall_reliability"]
        == "high"
    )

    assert (
        report_summary["recommendation_decision"]
        == "recommend_candidate_model"
    )

    assert (
        report_summary["best_overall_model"]
        == "Logistic Regression"
    )

    assert (
        report_summary["best_useful_model"]
        == "Logistic Regression"
    )

    assert report_summary["models_trained"] == 4

    assert (
        report_summary["final_status"]
        == "final_report_created"
    )