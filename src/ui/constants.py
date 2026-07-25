"""Shared UI constants.

The workflow rail describes controlled evaluation stages, not independent
agents. Each stage accepts legacy and current completion-marker names so this
UI can sit on top of more than one backend state version.
"""

from dataclasses import dataclass


APP_TITLE = "Agentic AutoML Advisor"
MAX_UPLOAD_BYTES = 150 * 1024 * 1024
PREVIEW_ROWS = 5_000


@dataclass(frozen=True)
class WorkflowStep:
    label: str
    action: str
    markers: tuple[str, ...]


WORKFLOW_STEPS: tuple[WorkflowStep, ...] = (
    WorkflowStep(
        "Inspect",
        "inspect_dataset",
        ("profiled_dataset", "dataset_inspection_completed"),
    ),
    WorkflowStep(
        "Plan",
        "create_experiment_plan",
        ("created_experiment_plan", "experiment_plan_created"),
    ),
    WorkflowStep(
        "Prepare",
        "create_preprocessing_config",
        ("created_preprocessing_config", "preprocessing_config_created"),
    ),
    WorkflowStep(
        "Pipeline",
        "create_preprocessing_pipeline",
        ("created_preprocessing_pipeline", "preprocessing_pipeline_created"),
    ),
    WorkflowStep(
        "Registry",
        "create_model_registry",
        ("created_model_registry", "model_registry_created"),
    ),
    WorkflowStep(
        "Train",
        "train_baseline_models",
        ("trained_baseline_models", "baseline_training_completed"),
    ),
    WorkflowStep(
        "Compare",
        "compare_models",
        ("created_model_leaderboard", "model_comparison_completed"),
    ),
    WorkflowStep(
        "Tune",
        "tune_candidate_model",
        (
            "tuned_candidate_model",
            "model_tuning_completed",
            "hyperparameter_tuning_completed",
            "tuning_completed",
            "model_tuning_skipped",
            "tuning_skipped",
        ),
    ),
    WorkflowStep(
        "Holdout",
        "evaluate_final_holdout",
        (
            "evaluated_final_holdout",
            "holdout_evaluation_completed",
            "final_holdout_evaluated",
            "holdout_evaluation_skipped",
            "final_holdout_skipped",
        ),
    ),
    WorkflowStep(
        "Critique",
        "run_reliability_critic",
        ("created_critic_report", "reliability_critique_completed"),
    ),
    WorkflowStep(
        "Persist",
        "check_model_persistence",
        ("checked_model_persistence", "model_persistence_checked"),
    ),
    WorkflowStep(
        "Report",
        "create_final_report",
        ("created_final_report", "final_report_created"),
    ),
)


ACTION_LABELS = {
    "inspect_dataset": "Inspecting dataset",
    "create_experiment_plan": "Creating experiment plan",
    "create_preprocessing_config": "Planning preprocessing",
    "create_preprocessing_pipeline": "Building preprocessing pipeline",
    "create_model_registry": "Preparing model registry",
    "train_baseline_models": "Training baseline models",
    "compare_models": "Comparing model evidence",
    "tune_candidate_model": "Tuning the candidate model",
    "tune_best_model": "Tuning the candidate model",
    "run_model_tuning": "Tuning the candidate model",
    "evaluate_final_holdout": "Evaluating the untouched holdout",
    "run_reliability_critic": "Reviewing reliability",
    "check_model_persistence": "Checking model persistence",
    "create_final_report": "Creating final report",
}


ACTION_STAGE = {
    alias: index
    for index, step in enumerate(WORKFLOW_STEPS, start=1)
    for alias in (
        step.action,
        *(
            ("tune_best_model", "run_model_tuning")
            if step.action == "tune_candidate_model"
            else ()
        ),
    )
}


RECOMMENDATION_LABELS = {
    "recommend_candidate_model": "Recommended candidate",
    "recommend_with_caution": "Recommended with caution",
    "do_not_recommend_model": "Recommendation withheld",
}


PAUSED_APPROVAL_STATUS = "preprocessing_config_created_with_pending_approvals"
COMPLETED_STATUS = "final_report_created"
