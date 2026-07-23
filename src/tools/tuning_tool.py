from datetime import datetime
from time import perf_counter
from typing import Any

import numpy as np
import optuna
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.data_loader import load_dataset
from src.model_registry import get_model_by_id
from src.preprocessing import build_preprocessor, prepare_features_and_target
from src.state import ExperimentState
from src.training import build_cv_strategy, build_scoring_strategy
from src.tuning_policy import assess_tuning_eligibility
from src.validation import create_validation_split


LOWER_IS_BETTER_METRICS = {
    "rmse",
    "mae",
    "mse",
    "mean_squared_error",
    "mean_absolute_error",
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
    Convert a value to a JSON-safe float.
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
    Normalise metric names used by training and tuning.
    """

    if not metric_name:
        return ""

    aliases = {
        "f1_score": "f1",
        "f1_weighted": "weighted_f1",
        "f1_weighted_score": "weighted_f1",
        "r_squared": "r2",
        "r^2": "r2",
        "root_mean_squared_error": "rmse",
        "mean_absolute_error": "mae",
        "mean_squared_error": "mse",
    }

    normalised = (
        str(metric_name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    return aliases.get(normalised, normalised)


def _metric_direction(primary_metric: str) -> str:
    """
    Return the optimisation direction for a metric.
    """

    if _normalise_metric_name(primary_metric) in LOWER_IS_BETTER_METRICS:
        return "minimize"

    return "maximize"


def _suggest_model_params(
    trial: optuna.Trial,
    model_id: str,
) -> dict[str, Any]:
    """
    Define small, controlled search spaces for approved model families.

    The search spaces are deliberately limited to avoid expensive or
    uncontrolled tuning.
    """

    if model_id == "logistic_regression":
        return {
            "C": trial.suggest_float(
                "C",
                1e-3,
                100.0,
                log=True,
            ),
            "class_weight": trial.suggest_categorical(
                "class_weight",
                [None, "balanced"],
            ),
        }

    if model_id == "ridge_regression":
        return {
            "alpha": trial.suggest_float(
                "alpha",
                1e-4,
                100.0,
                log=True,
            ),
        }

    if model_id == "random_forest_classifier":
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                100,
                300,
                step=50,
            ),
            "max_depth": trial.suggest_categorical(
                "max_depth",
                [None, 5, 10, 15, 20],
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                2,
                12,
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                1,
                6,
            ),
            "max_features": trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2", None],
            ),
            "class_weight": trial.suggest_categorical(
                "class_weight",
                [None, "balanced", "balanced_subsample"],
            ),
        }

    if model_id == "random_forest_regressor":
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                100,
                300,
                step=50,
            ),
            "max_depth": trial.suggest_categorical(
                "max_depth",
                [None, 5, 10, 15, 20],
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                2,
                12,
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                1,
                6,
            ),
            "max_features": trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2", None],
            ),
        }

    if model_id in {
        "hist_gradient_boosting_classifier",
        "hist_gradient_boosting_regressor",
    }:
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.30,
                log=True,
            ),
            "max_iter": trial.suggest_int(
                "max_iter",
                100,
                300,
                step=50,
            ),
            "max_leaf_nodes": trial.suggest_int(
                "max_leaf_nodes",
                15,
                63,
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                10,
                40,
            ),
            "l2_regularization": trial.suggest_float(
                "l2_regularization",
                1e-8,
                10.0,
                log=True,
            ),
        }

    raise ValueError(
        f"No approved tuning search space exists for model '{model_id}'."
    )


def _default_trial_params(model_id: str) -> dict[str, Any] | None:
    """
    Return a baseline-like trial for the selected search space.

    Enqueuing this trial helps ensure the search evaluates settings close to
    the existing baseline before exploring alternatives.
    """

    defaults: dict[str, dict[str, Any]] = {
        "logistic_regression": {
            "C": 1.0,
            "class_weight": None,
        },
        "ridge_regression": {
            "alpha": 1.0,
        },
        "random_forest_classifier": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": None,
        },
        "random_forest_regressor": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
        },
        "hist_gradient_boosting_classifier": {
            "learning_rate": 0.1,
            "max_iter": 100,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 1e-8,
        },
        "hist_gradient_boosting_regressor": {
            "learning_rate": 0.1,
            "max_iter": 100,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 1e-8,
        },
    }

    return defaults.get(model_id)


def _find_candidate_summary(
    state: ExperimentState,
    model_id: str,
) -> dict[str, Any]:
    """
    Find the selected baseline candidate in the leaderboard.
    """

    for item in state.leaderboard:
        if item.get("model_id") == model_id:
            return item

    for item in state.model_results:
        if item.get("model_id") == model_id:
            return item

    return {}


def _calculate_cv_score(
    pipeline: Pipeline,
    X_train: Any,
    y_train: Any,
    scorer: Any,
    cv: Any,
    primary_metric: str,
) -> float:
    """
    Evaluate one trial with CV on the training portion only.
    """

    scores = cross_val_score(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        scoring=scorer,
        cv=cv,
        n_jobs=1,
        error_score="raise",
    )

    mean_score = float(np.nanmean(scores))

    if _normalise_metric_name(primary_metric) in LOWER_IS_BETTER_METRICS:
        # Scikit-learn error scorers return negative values.
        return -mean_score

    return mean_score


def _is_tuned_result_acceptable(
    baseline_score: float | None,
    tuned_score: float | None,
    primary_metric: str,
) -> tuple[bool, float | None]:
    """
    Ensure tuning does not replace the baseline with a worse CV result.
    """

    if baseline_score is None or tuned_score is None:
        return False, None

    if _metric_direction(primary_metric) == "minimize":
        improvement = baseline_score - tuned_score
        return tuned_score <= baseline_score, improvement

    improvement = tuned_score - baseline_score
    return tuned_score >= baseline_score, improvement


def tuning_tool(
    state: ExperimentState,
    max_trials: int = 12,
    timeout_seconds: int = 180,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Run guarded hyperparameter tuning on the selected useful candidate.

    Safety rules:
    - tuning runs only after baseline comparison;
    - eligibility policy must approve the experiment;
    - only the training portion is used;
    - the final holdout is never passed to Optuna;
    - tuned parameters are accepted only when CV does not become worse.
    """

    tool_name = "tuning_tool"
    state.iteration += 1

    _add_tool_event(
        state=state,
        tool_name=tool_name,
        status="started",
        message="Started guarded tuning eligibility check.",
    )

    try:
        if state.status != "model_comparison_completed":
            raise ValueError(
                "Cannot run guarded tuning before model comparison is completed."
            )

        eligibility = assess_tuning_eligibility(
            state=state,
            max_trials=max_trials,
            timeout_seconds=timeout_seconds,
        )

        state.tuning_summary = eligibility

        if not eligibility["eligible"]:
            result = {
                "status": "skipped",
                "reason": eligibility["reason"],
                "blocking_reasons": eligibility["blocking_reasons"],
                "model_id": eligibility.get("candidate_model_id"),
                "display_name": eligibility.get(
                    "candidate_display_name"
                ),
                "best_params": {},
                "baseline_cv_score": eligibility.get(
                    "candidate_primary_score"
                ),
                "tuned_cv_score": None,
                "accepted_for_final_evaluation": False,
                "holdout_used_during_tuning": False,
            }

            state.tuning_result = result
            state.status = "tuning_completed"
            state.completed_steps.append("checked_guarded_tuning")

            _add_tool_event(
                state=state,
                tool_name=tool_name,
                status="completed",
                message=(
                    "Guarded tuning was skipped. "
                    + " ".join(eligibility["blocking_reasons"])
                ),
            )

            return {
                "success": True,
                "state": state,
                "tuning_summary": eligibility,
                "tuning_result": result,
                "error": None,
            }

        model_id = eligibility["candidate_model_id"]

        if not model_id:
            raise ValueError(
                "Tuning was approved, but no candidate model ID was found."
            )

        dataset = load_dataset(
            file_path=state.dataset_path,
            target_column=state.target_column,
        )

        X, y = prepare_features_and_target(
            df=dataset.df,
            preprocessing_config=state.preprocessing_config,
        )

        validation_summary = state.validation_summary or {}

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

        if not validation_split.summary.get("holdout_used"):
            raise ValueError(
                "Tuning eligibility expected a reserved holdout, but the "
                "deterministic validation split did not recreate one."
            )

        preprocessor = build_preprocessor(
            preprocessing_config=state.preprocessing_config,
        )

        candidate_model = get_model_by_id(
            base_task=state.base_task,
            model_id=model_id,
            random_state=random_state,
        )

        candidate_summary = _find_candidate_summary(
            state=state,
            model_id=model_id,
        )

        primary_metric = _normalise_metric_name(
            candidate_summary.get("primary_metric")
            or state.training_summary.get("primary_metric")
        )

        if not primary_metric:
            raise ValueError(
                "No primary metric was available for guarded tuning."
            )

        scoring = build_scoring_strategy(
            task_type=state.task_type,
            y=validation_split.y_train,
        )

        scorer = scoring.get(primary_metric)

        if scorer is None:
            raise ValueError(
                f"The primary metric '{primary_metric}' is not available "
                "in the current scoring strategy."
            )

        cv = build_cv_strategy(
            base_task=state.base_task,
            preprocessing_config=state.preprocessing_config,
            y=validation_split.y_train,
            random_state=random_state,
        )

        study_direction = _metric_direction(primary_metric)

        sampler = optuna.samplers.TPESampler(
            seed=random_state,
        )

        study = optuna.create_study(
            direction=study_direction,
            sampler=sampler,
            study_name=f"{model_id}_guarded_tuning",
        )

        default_params = _default_trial_params(model_id)

        if default_params:
            study.enqueue_trial(default_params)

        def objective(trial: optuna.Trial) -> float:
            model_params = _suggest_model_params(
                trial=trial,
                model_id=model_id,
            )

            estimator = clone(candidate_model.estimator)
            estimator.set_params(**model_params)

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", clone(preprocessor)),
                    ("model", estimator),
                ]
            )

            return _calculate_cv_score(
                pipeline=pipeline,
                X_train=validation_split.X_train,
                y_train=validation_split.y_train,
                scorer=scorer,
                cv=cv,
                primary_metric=primary_metric,
            )

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        started_at = datetime.now().isoformat(timespec="seconds")
        start_time = perf_counter()

        study.optimize(
            objective,
            n_trials=max_trials,
            timeout=timeout_seconds,
            catch=(ValueError,),
            show_progress_bar=False,
        )

        duration_seconds = perf_counter() - start_time
        completed_trials = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]

        if not completed_trials:
            raise ValueError(
                "Guarded tuning completed without any successful trials."
            )

        tuned_score = _safe_float(study.best_value)
        baseline_score = _safe_float(
            candidate_summary.get("primary_score")
        )

        accepted, improvement = _is_tuned_result_acceptable(
            baseline_score=baseline_score,
            tuned_score=tuned_score,
            primary_metric=primary_metric,
        )

        result = {
            "status": "completed",
            "reason": (
                "Guarded tuning completed using cross-validation on the "
                "training portion only."
            ),
            "model_id": model_id,
            "display_name": (
                candidate_summary.get("display_name")
                or eligibility.get("candidate_display_name")
                or model_id
            ),
            "primary_metric": primary_metric,
            "metric_direction": study_direction,
            "baseline_cv_score": baseline_score,
            "tuned_cv_score": tuned_score,
            "improvement": _safe_float(improvement),
            "best_params": dict(study.best_params),
            "best_trial_number": int(study.best_trial.number),
            "trials_requested": int(max_trials),
            "trials_completed": int(len(completed_trials)),
            "timeout_seconds": int(timeout_seconds),
            "duration_seconds": _safe_float(duration_seconds),
            "started_at": started_at,
            "completed_at": datetime.now().isoformat(
                timespec="seconds"
            ),
            "accepted_for_final_evaluation": accepted,
            "holdout_used_during_tuning": False,
            "training_rows": int(
                len(validation_split.y_train)
            ),
            "reserved_holdout_rows": int(
                len(validation_split.y_holdout)
            ),
        }

        state.tuning_result = result
        state.tuning_summary = {
            **eligibility,
            "status": "completed",
            "trials_completed": len(completed_trials),
            "duration_seconds": _safe_float(duration_seconds),
        }
        state.status = "tuning_completed"
        state.completed_steps.append("completed_guarded_tuning")

        outcome_message = (
            "accepted for final holdout evaluation"
            if accepted
            else "not accepted because it did not improve on the baseline"
        )

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="completed",
            message=(
                f"Guarded tuning completed for "
                f"{result['display_name']} with "
                f"{len(completed_trials)} successful trial(s). "
                f"The tuned parameters were {outcome_message}."
            ),
        )

        return {
            "success": True,
            "state": state,
            "tuning_summary": state.tuning_summary,
            "tuning_result": result,
            "error": None,
        }

    except Exception as error:
        state.status = "tuning_failed"
        state.warnings.append(str(error))
        state.completed_steps.append("failed_guarded_tuning")

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="failed",
            message=str(error),
        )

        return {
            "success": False,
            "state": state,
            "tuning_summary": state.tuning_summary,
            "tuning_result": {},
            "error": str(error),
        }