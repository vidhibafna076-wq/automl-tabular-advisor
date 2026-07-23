from datetime import datetime
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.data_loader import load_dataset
from src.model_registry import get_model_by_id
from src.preprocessing import build_preprocessor, prepare_features_and_target
from src.state import ExperimentState
from src.validation import create_validation_split


LOWER_IS_BETTER_METRICS = {
    "mae",
    "mean_absolute_error",
    "mse",
    "mean_squared_error",
    "rmse",
    "root_mean_squared_error",
}


def _add_tool_event(
    state: ExperimentState,
    tool_name: str,
    status: str,
    message: str,
) -> None:
    """
    Add a factual event to the experiment tool history.
    """

    state.tool_history.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tool_name": tool_name,
            "status": status,
            "message": message,
        }
    )


def _safe_float(value: Any) -> float | None:
    """
    Convert a numeric value into a JSON-safe float.
    """

    try:
        converted = float(value)

        if np.isnan(converted) or np.isinf(converted):
            return None

        return converted

    except (TypeError, ValueError):
        return None


def _normalise_metric_name(metric_name: str | None) -> str:
    """
    Normalise metric names so training and holdout names can be matched.
    """

    if not metric_name:
        return ""

    aliases = {
        "f1_score": "f1",
        "f1_weighted_score": "f1_weighted",
        "balanced_accuracy_score": "balanced_accuracy",
        "precision_score": "precision",
        "recall_score": "recall",
        "roc_auc_score": "roc_auc",
        "average_precision": "pr_auc",
        "average_precision_score": "pr_auc",
        "r_squared": "r2",
        "r^2": "r2",
        "mean_absolute_error": "mae",
        "mean_squared_error": "mse",
        "root_mean_squared_error": "rmse",
    }

    normalised = str(metric_name).strip().lower().replace(" ", "_")

    return aliases.get(normalised, normalised)


def _find_candidate_summary(
    state: ExperimentState,
    model_id: str,
) -> dict[str, Any]:
    """
    Find the selected candidate in the leaderboard or model results.
    """

    for item in state.leaderboard:
        if item.get("model_id") == model_id:
            return item

    for item in state.model_results:
        if item.get("model_id") == model_id:
            return item

    return {}


def _classification_metrics(
    fitted_pipeline: Pipeline,
    X_holdout: Any,
    y_holdout: Any,
) -> dict[str, Any]:
    """
    Calculate classification metrics on the untouched holdout.
    """

    predictions = fitted_pipeline.predict(X_holdout)

    labels = list(np.unique(y_holdout))
    is_binary = len(labels) == 2

    metrics: dict[str, Any] = {
        "accuracy": _safe_float(
            accuracy_score(y_holdout, predictions)
        ),
        "balanced_accuracy": _safe_float(
            balanced_accuracy_score(y_holdout, predictions)
        ),
    }

    if is_binary:
        positive_label = sorted(labels, key=lambda value: str(value))[-1]

        metrics["precision"] = _safe_float(
            precision_score(
                y_holdout,
                predictions,
                pos_label=positive_label,
                zero_division=0,
            )
        )
        metrics["recall"] = _safe_float(
            recall_score(
                y_holdout,
                predictions,
                pos_label=positive_label,
                zero_division=0,
            )
        )
        metrics["f1"] = _safe_float(
            f1_score(
                y_holdout,
                predictions,
                pos_label=positive_label,
                zero_division=0,
            )
        )
    else:
        positive_label = None

        metrics["precision_weighted"] = _safe_float(
            precision_score(
                y_holdout,
                predictions,
                average="weighted",
                zero_division=0,
            )
        )
        metrics["recall_weighted"] = _safe_float(
            recall_score(
                y_holdout,
                predictions,
                average="weighted",
                zero_division=0,
            )
        )
        metrics["f1_weighted"] = _safe_float(
            f1_score(
                y_holdout,
                predictions,
                average="weighted",
                zero_division=0,
            )
        )

    try:
        probabilities = fitted_pipeline.predict_proba(X_holdout)
        model = fitted_pipeline.named_steps["model"]
        model_classes = list(model.classes_)

        if is_binary:
            positive_index = model_classes.index(positive_label)
            positive_probabilities = probabilities[:, positive_index]

            binary_target = np.array(
                [
                    1 if value == positive_label else 0
                    for value in y_holdout
                ]
            )

            metrics["roc_auc"] = _safe_float(
                roc_auc_score(binary_target, positive_probabilities)
            )
            metrics["pr_auc"] = _safe_float(
                average_precision_score(
                    binary_target,
                    positive_probabilities,
                )
            )
        else:
            metrics["roc_auc_ovr_weighted"] = _safe_float(
                roc_auc_score(
                    y_holdout,
                    probabilities,
                    labels=model_classes,
                    multi_class="ovr",
                    average="weighted",
                )
            )

    except (AttributeError, ValueError):
        # Some models or holdout class combinations do not support probability
        # metrics. The other classification metrics remain valid.
        pass

    return {
        "metrics": metrics,
        "labels": [str(label) for label in labels],
        "positive_label": (
            str(positive_label)
            if positive_label is not None
            else None
        ),
        "confusion_matrix": confusion_matrix(
            y_holdout,
            predictions,
            labels=labels,
        ).tolist(),
    }


def _regression_metrics(
    fitted_pipeline: Pipeline,
    X_holdout: Any,
    y_holdout: Any,
) -> dict[str, Any]:
    """
    Calculate regression metrics on the untouched holdout.
    """

    predictions = fitted_pipeline.predict(X_holdout)

    mse = mean_squared_error(y_holdout, predictions)

    return {
        "metrics": {
            "mae": _safe_float(
                mean_absolute_error(y_holdout, predictions)
            ),
            "mse": _safe_float(mse),
            "rmse": _safe_float(np.sqrt(mse)),
            "r2": _safe_float(r2_score(y_holdout, predictions)),
        }
    }


def _evaluate_holdout_guardrail(
    primary_metric: str,
    cv_score: float | None,
    holdout_score: float | None,
) -> tuple[bool, str]:
    """
    Compare the holdout score with the cross-validation estimate.

    This is a conservative prototype guardrail, not a universal statistical
    rule.

    Higher-is-better metrics:
        Allow an absolute reduction of at most 0.10.

    Error metrics:
        Allow an increase of at most 20%, with a minimum tolerance of 0.05.
    """

    if cv_score is None or holdout_score is None:
        return (
            False,
            "The primary cross-validation and holdout scores could not both be calculated.",
        )

    metric_name = _normalise_metric_name(primary_metric)

    if metric_name in LOWER_IS_BETTER_METRICS:
        comparable_cv_score = abs(cv_score)
        tolerance = max(abs(comparable_cv_score) * 0.20, 0.05)
        maximum_allowed_score = comparable_cv_score + tolerance

        passed = holdout_score <= maximum_allowed_score

        reason = (
            f"Holdout {metric_name} was {holdout_score:.4f}. "
            f"The maximum permitted value under the current guardrail was "
            f"{maximum_allowed_score:.4f}."
        )

        return passed, reason

    minimum_allowed_score = cv_score - 0.10
    passed = holdout_score >= minimum_allowed_score

    reason = (
        f"Holdout {metric_name} was {holdout_score:.4f}. "
        f"The minimum permitted value under the current guardrail was "
        f"{minimum_allowed_score:.4f}."
    )

    return passed, reason


def holdout_evaluation_tool(
    state: ExperimentState,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Evaluate the best useful candidate on an untouched final holdout.

    The split is recreated using the same deterministic validation policy and
    random state used during baseline training.

    When no holdout was reserved, the tool records an honest skipped result and
    allows the workflow to continue.
    """

    tool_name = "holdout_evaluation_tool"
    state.iteration += 1

    _add_tool_event(
        state=state,
        tool_name=tool_name,
        status="started",
        message="Started final holdout evaluation check.",
    )

    try:
        allowed_statuses = {
            "model_comparison_completed",
            "tuning_completed",
        }

        if state.status not in allowed_statuses:
            raise ValueError(
                "Cannot evaluate the final holdout before model comparison and "
                "guarded tuning are completed."
            )

        validation_summary = state.validation_summary or {}

        if not validation_summary.get("holdout_used"):
            result = {
                "status": "skipped",
                "holdout_used": False,
                "passes_holdout_guardrail": None,
                "reason": (
                    "Holdout evaluation was skipped because no final holdout "
                    "was reserved for this dataset."
                ),
                "model_id": None,
                "display_name": None,
                "metrics": {},
            }

            state.holdout_result = result
            state.status = "holdout_evaluation_completed"
            state.completed_steps.append("checked_final_holdout")

            _add_tool_event(
                state=state,
                tool_name=tool_name,
                status="completed",
                message=result["reason"],
            )

            return {
                "success": True,
                "state": state,
                "holdout_result": result,
                "error": None,
            }

        comparison_summary = state.comparison_summary or {}
        model_id = comparison_summary.get("best_useful_model_id")

        if not model_id:
            result = {
                "status": "skipped_no_candidate",
                "holdout_used": True,
                "passes_holdout_guardrail": False,
                "reason": (
                    "A holdout was reserved, but no useful non-dummy candidate "
                    "was available for final evaluation."
                ),
                "model_id": None,
                "display_name": None,
                "metrics": {},
            }

            state.holdout_result = result
            state.status = "holdout_evaluation_completed"
            state.completed_steps.append("checked_final_holdout")

            _add_tool_event(
                state=state,
                tool_name=tool_name,
                status="completed",
                message=result["reason"],
            )

            return {
                "success": True,
                "state": state,
                "holdout_result": result,
                "error": None,
            }

        dataset = load_dataset(
            file_path=state.dataset_path,
            target_column=state.target_column,
        )

        X, y = prepare_features_and_target(
            df=dataset.df,
            preprocessing_config=state.preprocessing_config,
        )

        validation_split = create_validation_split(
            X=X,
            y=y,
            base_task=state.base_task,
            holdout_fraction=float(
                validation_summary.get(
                    "requested_holdout_fraction",
                    0.20,
                )
            ),
            minimum_rows_for_holdout=int(
                validation_summary.get(
                    "minimum_rows_for_holdout",
                    200,
                )
            ),
            minimum_class_count_for_holdout=int(
                validation_summary.get(
                    "minimum_class_count_for_holdout",
                    10,
                )
                or 10
            ),
            random_state=int(
                validation_summary.get(
                    "random_state",
                    random_state,
                )
            ),
        )

        if (
            validation_split.X_holdout is None
            or validation_split.y_holdout is None
        ):
            raise ValueError(
                "The validation policy said a holdout was reserved, but the "
                "deterministic split could not recreate it."
            )

        expected_holdout_rows = validation_summary.get("holdout_rows")

        if (
            expected_holdout_rows is not None
            and len(validation_split.y_holdout)
            != int(expected_holdout_rows)
        ):
            raise ValueError(
                "The recreated holdout row count does not match the original "
                "validation summary."
            )

        preprocessor = build_preprocessor(
            preprocessing_config=state.preprocessing_config,
        )

        candidate_model = get_model_by_id(
            base_task=state.base_task,
            model_id=model_id,
            random_state=random_state,
        )

        tuning_result = state.tuning_result or {}

        use_tuned_parameters = (
            tuning_result.get("status") == "completed"
            and tuning_result.get("accepted_for_final_evaluation") is True
            and tuning_result.get("model_id") == model_id
        )

        final_estimator = clone(candidate_model.estimator)

        if use_tuned_parameters:
            final_estimator.set_params(
                **tuning_result.get("best_params", {})
            )

        fitted_pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", final_estimator),
            ]
        )

        fitted_pipeline.fit(
            validation_split.X_train,
            validation_split.y_train,
        )

        if state.base_task == "classification":
            evaluation = _classification_metrics(
                fitted_pipeline=fitted_pipeline,
                X_holdout=validation_split.X_holdout,
                y_holdout=validation_split.y_holdout,
            )
        elif state.base_task == "regression":
            evaluation = _regression_metrics(
                fitted_pipeline=fitted_pipeline,
                X_holdout=validation_split.X_holdout,
                y_holdout=validation_split.y_holdout,
            )
        else:
            raise ValueError(
                f"Unsupported base task for holdout evaluation: {state.base_task}"
            )

        candidate_summary = _find_candidate_summary(
            state=state,
            model_id=model_id,
        )

        primary_metric = _normalise_metric_name(
            comparison_summary.get("primary_metric")
            or candidate_summary.get("primary_metric")
        )

        if use_tuned_parameters:
            cv_primary_score = _safe_float(
                tuning_result.get("tuned_cv_score")
            )
        else:
            cv_primary_score = _safe_float(
                candidate_summary.get("primary_score")
            )

        holdout_primary_score = _safe_float(
            evaluation.get("metrics", {}).get(primary_metric)
        )

        passed_guardrail, guardrail_reason = (
            _evaluate_holdout_guardrail(
                primary_metric=primary_metric,
                cv_score=cv_primary_score,
                holdout_score=holdout_primary_score,
            )
        )

        result = {
            "status": "completed",
            "holdout_used": True,
            "model_id": model_id,
            "display_name": (
                candidate_summary.get("display_name")
                or comparison_summary.get("best_useful_display_name")
                or model_id
            ),
            "primary_metric": primary_metric,
            "cv_primary_score": cv_primary_score,
            "holdout_primary_score": holdout_primary_score,
            "passes_holdout_guardrail": passed_guardrail,
            "guardrail_reason": guardrail_reason,
            "training_rows": int(len(validation_split.y_train)),
            "holdout_rows": int(len(validation_split.y_holdout)),
            "metrics": evaluation.get("metrics", {}),
            "labels": evaluation.get("labels", []),
            "positive_label": evaluation.get("positive_label"),
            "confusion_matrix": evaluation.get(
                "confusion_matrix",
                [],
            ),
            "reason": (
                "The selected useful candidate was fitted on the training "
                "portion and evaluated once on the untouched holdout."
            ),

            "used_tuned_parameters": use_tuned_parameters,
            "tuned_parameters": (
                tuning_result.get("best_params", {})
                if use_tuned_parameters
                else {}
            ),
            "baseline_cv_score": _safe_float(
                tuning_result.get("baseline_cv_score")
            ),
        }

        state.holdout_result = result
        state.status = "holdout_evaluation_completed"
        state.completed_steps.append("checked_final_holdout")

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="completed",
            message=(
                f"Evaluated {result['display_name']} on "
                f"{result['holdout_rows']} untouched holdout row(s). "
                f"Guardrail passed: {passed_guardrail}."
            ),
        )

        return {
            "success": True,
            "state": state,
            "holdout_result": result,
            "error": None,
        }

    except Exception as error:
        state.status = "holdout_evaluation_failed"
        state.warnings.append(str(error))
        state.completed_steps.append("failed_holdout_evaluation")

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="failed",
            message=str(error),
        )

        return {
            "success": False,
            "state": state,
            "holdout_result": {},
            "error": str(error),
        }