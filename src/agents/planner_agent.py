from datetime import datetime
from typing import Any

from src.state import ExperimentState


def _add_plan_step(
    plan: list[dict[str, Any]],
    step_id: str,
    title: str,
    action: str,
    reason: str,
    requires_user_approval: bool = False,
    priority: str = "medium",
) -> None:
    """
    Add one structured step to the experiment plan.

    A plan step should explain:
    - what action is recommended
    - why it is recommended
    - whether user approval is required
    - how important it is
    """

    plan.append(
        {
            "step_id": step_id,
            "title": title,
            "action": action,
            "reason": reason,
            "requires_user_approval": requires_user_approval,
            "priority": priority,
            "status": "planned",
        }
    )


def _has_quality_issue(state: ExperimentState, issue_name: str) -> bool:
    """
    Check whether a specific data quality issue exists in the state.
    """

    return any(
        issue["issue"] == issue_name
        for issue in state.quality_issues
    )


def create_experiment_plan(state: ExperimentState) -> list[dict[str, Any]]:
    """
    Create a rule-based experiment plan from the current experiment state.

    This is our first planner agent.

    It does not execute the plan.
    It only decides what should happen next.
    """

    if state.status != "dataset_inspection_completed":
        raise ValueError(
            "Cannot create experiment plan before dataset inspection is completed."
        )

    plan: list[dict[str, Any]] = []

    profile = state.profile

    # 1. Handle possible ID columns
    possible_id_columns = profile.get("possible_id_columns", [])

    if possible_id_columns:
        _add_plan_step(
            plan=plan,
            step_id="review_id_columns",
            title="Review possible ID columns",
            action=(
                "Review and exclude likely identifier columns before model training: "
                + ", ".join(possible_id_columns)
            ),
            reason=(
                "Identifier columns usually do not generalise to new data and may "
                "cause the model to memorise records instead of learning patterns."
            ),
            requires_user_approval=True,
            priority="high",
        )

    # 2. Handle missing values
    if _has_quality_issue(state, "Missing values detected"):
        _add_plan_step(
            plan=plan,
            step_id="apply_missing_value_imputation",
            title="Apply missing value imputation",
            action=(
                "Use median imputation for numerical columns and most-frequent "
                "imputation for categorical columns."
            ),
            reason=(
                "Most machine learning models cannot train directly on missing values."
            ),
            requires_user_approval=False,
            priority="high",
        )

    # 3. Handle constant columns
    constant_columns = profile.get("constant_columns", [])

    if constant_columns:
        _add_plan_step(
            plan=plan,
            step_id="drop_constant_columns",
            title="Drop constant columns",
            action="Drop columns with only one unique value: " + ", ".join(constant_columns),
            reason="Constant columns do not provide predictive information.",
            requires_user_approval=False,
            priority="medium",
        )

    # 4. Handle high-cardinality categorical columns
    high_cardinality_columns = profile.get("high_cardinality_columns", [])

    if high_cardinality_columns:
        affected = [
            item["column"]
            for item in high_cardinality_columns
        ]

        _add_plan_step(
            plan=plan,
            step_id="handle_high_cardinality_columns",
            title="Handle high-cardinality categorical columns",
            action=(
                "Review high-cardinality categorical columns before encoding: "
                + ", ".join(affected)
            ),
            reason=(
                "One-hot encoding high-cardinality columns can create too many features "
                "and increase overfitting risk."
            ),
            requires_user_approval=True,
            priority="medium",
        )

    # 5. Add preprocessing plan
    numerical_columns = profile.get("numerical_columns", [])
    categorical_columns = profile.get("categorical_columns", [])

    _add_plan_step(
        plan=plan,
        step_id="build_preprocessing_pipeline",
        title="Build preprocessing pipeline",
        action=(
            f"Prepare {len(numerical_columns)} numerical column(s) and "
            f"{len(categorical_columns)} categorical column(s) using a scikit-learn "
            "ColumnTransformer."
        ),
        reason=(
            "Numerical and categorical columns need different preprocessing before "
            "they can be used by machine learning models."
        ),
        requires_user_approval=False,
        priority="high",
    )

    # 6. Add validation strategy
    rows = profile.get("rows", 0)

    if state.base_task == "classification":
        if rows < 500:
            validation_action = (
                "Use stratified cross-validation because the dataset is small and the "
                "task is classification."
            )
        else:
            validation_action = (
                "Use stratified train/test split and cross-validation for classification."
            )

        _add_plan_step(
            plan=plan,
            step_id="choose_validation_strategy",
            title="Choose validation strategy",
            action=validation_action,
            reason=(
                "Stratification helps preserve class distribution across training and "
                "validation folds."
            ),
            requires_user_approval=False,
            priority="high",
        )

    elif state.base_task == "regression":
        if rows < 500:
            validation_action = (
                "Use K-fold cross-validation because the dataset is small."
            )
        else:
            validation_action = (
                "Use train/test split and K-fold cross-validation for regression."
            )

        _add_plan_step(
            plan=plan,
            step_id="choose_validation_strategy",
            title="Choose validation strategy",
            action=validation_action,
            reason=(
                "Cross-validation gives a more stable estimate of performance than a "
                "single split."
            ),
            requires_user_approval=False,
            priority="high",
        )

    # 7. Add metric strategy
    if state.task_type == "binary_classification":
        metric_action = (
            "Evaluate models using F1-score, recall, precision, balanced accuracy, "
            "and ROC-AUC where possible."
        )
        metric_reason = (
            "Binary classification should not be judged by accuracy alone, especially "
            "if class imbalance exists."
        )

    elif state.task_type == "multiclass_classification":
        metric_action = (
            "Evaluate models using macro F1, weighted F1, balanced accuracy, and accuracy."
        )
        metric_reason = (
            "Macro and weighted scores help compare multiclass performance more fairly."
        )

    else:
        metric_action = (
            "Evaluate models using MAE, RMSE, and R^2."
        )
        metric_reason = (
            "Regression models should be evaluated using error-based metrics and "
            "explained variance."
        )

    _add_plan_step(
        plan=plan,
        step_id="choose_evaluation_metrics",
        title="Choose evaluation metrics",
        action=metric_action,
        reason=metric_reason,
        requires_user_approval=False,
        priority="high",
    )

    # 8. Add baseline model training
    if state.base_task == "classification":
        model_action = (
            "Train baseline classification models: Dummy Classifier, Logistic Regression, "
            "Random Forest, and Histogram Gradient Boosting."
        )
    else:
        model_action = (
            "Train baseline regression models: Dummy Regressor, Ridge Regression, "
            "Random Forest Regressor, and Histogram Gradient Boosting Regressor."
        )

    _add_plan_step(
        plan=plan,
        step_id="train_baseline_models",
        title="Train baseline models",
        action=model_action,
        reason=(
            "Baseline models provide a fair starting point and help verify whether "
            "the dataset contains predictive signal."
        ),
        requires_user_approval=False,
        priority="high",
    )

    # 9. Add reliability check
    _add_plan_step(
        plan=plan,
        step_id="run_reliability_checks",
        title="Run reliability checks",
        action=(
            "Compare models against the dummy baseline, check score stability, and "
            "flag possible overfitting or unreliable results."
        ),
        reason=(
            "The highest-scoring model is not always the most trustworthy model."
        ),
        requires_user_approval=False,
        priority="high",
    )

    return plan


def planner_agent(state: ExperimentState) -> dict[str, Any]:
    """
    Planner agent entry point.

    This agent reads the current state and creates a structured experiment plan.
    """

    tool_name = "planner_agent"

    state.iteration += 1

    state.tool_history.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tool_name": tool_name,
            "status": "started",
            "message": "Started experiment planning.",
        }
    )

    try:
        plan = create_experiment_plan(state)

        state.experiment_plan = plan
        state.status = "experiment_plan_created"
        state.completed_steps.append("created_experiment_plan")

        state.tool_history.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "tool_name": tool_name,
                "status": "completed",
                "message": f"Created experiment plan with {len(plan)} step(s).",
            }
        )

        return {
            "success": True,
            "state": state,
            "plan": plan,
            "error": None,
        }

    except Exception as error:
        state.status = "experiment_planning_failed"
        state.warnings.append(str(error))
        state.completed_steps.append("failed_experiment_planning")

        state.tool_history.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "tool_name": tool_name,
                "status": "failed",
                "message": str(error),
            }
        )

        return {
            "success": False,
            "state": state,
            "plan": [],
            "error": str(error),
        }