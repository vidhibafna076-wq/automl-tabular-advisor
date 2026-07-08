from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline

from src.model_registry import CandidateModel


def _safe_float(value: Any) -> float | None:
    """
    Convert a value to a JSON-safe float.

    If the value is NaN or infinite, return None instead.
    """

    try:
        value = float(value)

        if np.isnan(value) or np.isinf(value):
            return None

        return value

    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    """
    Convert a value to a JSON-safe integer.
    """

    try:
        return int(value)
    except Exception:
        return None


def _infer_positive_label(y: pd.Series) -> Any:
    """
    Infer the positive class for binary classification.

    This matters because precision, recall, F1, ROC-AUC, and PR-AUC need to know
    which class should be treated as the positive class.

    We use common positive labels first.
    If none are found, we use the minority class as the positive class.
    """

    y_clean = y.dropna()
    labels = list(y_clean.unique())

    preferred_positive_labels = [
        "y",
        "yes",
        "true",
        "1",
        "approved",
        "positive",
        "churn",
        "fraud",
    ]

    label_lookup = {
        str(label).strip().lower(): label
        for label in labels
    }

    for preferred in preferred_positive_labels:
        if preferred in label_lookup:
            return label_lookup[preferred]

    value_counts = y_clean.value_counts()

    if len(value_counts) == 2:
        return value_counts.index[-1]

    return labels[0]


def _get_probability_for_positive_class(
    estimator: Any,
    X: pd.DataFrame,
    positive_label: Any,
) -> np.ndarray | None:
    """
    Get predicted probabilities for the positive class.

    This function is used by ROC-AUC and PR-AUC scorers.
    """

    if not hasattr(estimator, "predict_proba"):
        return None

    probabilities = estimator.predict_proba(X)

    if not hasattr(estimator, "classes_"):
        return None

    classes = list(estimator.classes_)

    if positive_label not in classes:
        return None

    positive_index = classes.index(positive_label)

    return probabilities[:, positive_index]


def _make_roc_auc_scorer(positive_label: Any):
    """
    Create a scorer for ROC-AUC.

    This scorer is compatible with cross_validate because it receives:
    estimator, X, y_true.
    """

    def scorer(estimator: Any, X: pd.DataFrame, y_true: pd.Series) -> float:
        positive_probabilities = _get_probability_for_positive_class(
            estimator=estimator,
            X=X,
            positive_label=positive_label,
        )

        if positive_probabilities is None:
            return np.nan

        y_binary = (pd.Series(y_true).reset_index(drop=True) == positive_label).astype(int)

        try:
            return roc_auc_score(y_binary, positive_probabilities)
        except Exception:
            return np.nan

    return scorer


def _make_pr_auc_scorer(positive_label: Any):
    """
    Create a scorer for PR-AUC.

    PR-AUC is calculated using average precision.
    """

    def scorer(estimator: Any, X: pd.DataFrame, y_true: pd.Series) -> float:
        positive_probabilities = _get_probability_for_positive_class(
            estimator=estimator,
            X=X,
            positive_label=positive_label,
        )

        if positive_probabilities is None:
            return np.nan

        y_binary = (pd.Series(y_true).reset_index(drop=True) == positive_label).astype(int)

        try:
            return average_precision_score(y_binary, positive_probabilities)
        except Exception:
            return np.nan

    return scorer


def build_cv_strategy(
    base_task: str,
    preprocessing_config: dict[str, Any],
    y: pd.Series,
    random_state: int = 42,
):
    """
    Build the cross-validation strategy based on task type and config.

    For classification:
        use StratifiedKFold where possible.

    For regression:
        use KFold.
    """

    validation_config = preprocessing_config["validation_strategy"]
    requested_folds = int(validation_config.get("cv_folds", 5))

    if base_task == "classification":
        class_counts = y.value_counts()
        minimum_class_count = int(class_counts.min())

        n_splits = min(requested_folds, minimum_class_count)

        if n_splits < 2:
            raise ValueError(
                "Not enough samples per class for cross-validation. "
                "At least 2 samples per class are required."
            )

        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )

    n_splits = min(requested_folds, len(y))

    if n_splits < 2:
        raise ValueError(
            "Not enough rows for cross-validation. At least 2 rows are required."
        )

    return KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def build_scoring_strategy(
    task_type: str,
    y: pd.Series,
) -> dict[str, Any]:
    """
    Build scoring functions for cross-validation.
    """

    if task_type == "binary_classification":
        positive_label = _infer_positive_label(y)

        return {
            "accuracy": make_scorer(accuracy_score),
            "precision": make_scorer(
                precision_score,
                pos_label=positive_label,
                zero_division=0,
            ),
            "recall": make_scorer(
                recall_score,
                pos_label=positive_label,
                zero_division=0,
            ),
            "f1": make_scorer(
                f1_score,
                pos_label=positive_label,
                zero_division=0,
            ),
            "balanced_accuracy": make_scorer(balanced_accuracy_score),
            "roc_auc": _make_roc_auc_scorer(positive_label),
            "pr_auc": _make_pr_auc_scorer(positive_label),
        }

    if task_type == "multiclass_classification":
        return {
            "accuracy": make_scorer(accuracy_score),
            "macro_f1": make_scorer(
                f1_score,
                average="macro",
                zero_division=0,
            ),
            "weighted_f1": make_scorer(
                f1_score,
                average="weighted",
                zero_division=0,
            ),
            "balanced_accuracy": make_scorer(balanced_accuracy_score),
        }

    if task_type == "regression":
        return {
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        }

    raise ValueError(f"Unsupported task type: {task_type}")


def _summarise_metric(
    cv_results: dict[str, Any],
    metric_name: str,
    base_task: str,
) -> dict[str, Any]:
    """
    Summarise one metric from cross-validation output.

    For regression error metrics, scikit-learn returns negative values because
    it assumes higher score is better. We convert RMSE and MAE back to positive.
    """

    test_key = f"test_{metric_name}"
    train_key = f"train_{metric_name}"

    test_scores = cv_results.get(test_key)
    train_scores = cv_results.get(train_key)

    if test_scores is None:
        return {
            "cv_mean": None,
            "cv_std": None,
            "train_mean": None,
            "train_std": None,
        }

    test_scores = np.array(test_scores, dtype=float)

    if train_scores is not None:
        train_scores = np.array(train_scores, dtype=float)

    if base_task == "regression" and metric_name in ["rmse", "mae"]:
        test_scores = -test_scores

        if train_scores is not None:
            train_scores = -train_scores

    return {
        "cv_mean": _safe_float(np.nanmean(test_scores)),
        "cv_std": _safe_float(np.nanstd(test_scores)),
        "train_mean": _safe_float(np.nanmean(train_scores)) if train_scores is not None else None,
        "train_std": _safe_float(np.nanstd(train_scores)) if train_scores is not None else None,
    }


def _get_metric_direction(metric_name: str) -> str:
    """
    Tell whether higher or lower is better for a metric.
    """

    if metric_name in ["rmse", "mae"]:
        return "lower_is_better"

    return "higher_is_better"


def _class_metrics_from_report(report: dict[str, Any], labels: list[Any]) -> dict[str, Any]:
    """
    Extract JSON-safe class-level metrics from classification_report output.
    """

    class_metrics = {}

    for label in labels:
        label_key = str(label)
        values = report.get(label_key, {})

        class_metrics[label_key] = {
            "precision": _safe_float(values.get("precision")),
            "recall": _safe_float(values.get("recall")),
            "f1_score": _safe_float(values.get("f1-score")),
            "support": _safe_int(values.get("support")),
        }

    return class_metrics


def _create_classification_diagnostics(
    full_pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: Any,
    task_type: str,
) -> dict[str, Any]:
    """
    Create out-of-fold classification diagnostics.

    This gives us:
    - out-of-fold predicted labels
    - confusion matrix
    - class-level precision, recall, F1, and support
    - optional ROC-AUC and PR-AUC from out-of-fold probabilities
    """

    if task_type not in ["binary_classification", "multiclass_classification"]:
        return {}

    labels = list(y.dropna().unique())

    y_pred = cross_val_predict(
        estimator=full_pipeline,
        X=X,
        y=y,
        cv=cv,
        method="predict",
    )

    matrix = confusion_matrix(
        y_true=y,
        y_pred=y_pred,
        labels=labels,
    )

    report = classification_report(
        y_true=y,
        y_pred=y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    diagnostics = {
        "labels": [str(label) for label in labels],
        "confusion_matrix": matrix.tolist(),
        "class_level_metrics": _class_metrics_from_report(report, labels),
        "macro_avg": {
            "precision": _safe_float(report.get("macro avg", {}).get("precision")),
            "recall": _safe_float(report.get("macro avg", {}).get("recall")),
            "f1_score": _safe_float(report.get("macro avg", {}).get("f1-score")),
        },
        "weighted_avg": {
            "precision": _safe_float(report.get("weighted avg", {}).get("precision")),
            "recall": _safe_float(report.get("weighted avg", {}).get("recall")),
            "f1_score": _safe_float(report.get("weighted avg", {}).get("f1-score")),
        },
    }

    if task_type == "binary_classification":
        positive_label = _infer_positive_label(y)
        diagnostics["positive_label"] = str(positive_label)

        try:
            probabilities = cross_val_predict(
                estimator=full_pipeline,
                X=X,
                y=y,
                cv=cv,
                method="predict_proba",
            )

            # Most sklearn classifiers store class order sorted by label.
            # For this binary target, this safely maps the positive label to
            # the correct probability column.
            probability_class_order = sorted(labels, key=lambda value: str(value))

            if positive_label in probability_class_order:
                positive_index = probability_class_order.index(positive_label)
                positive_probabilities = probabilities[:, positive_index]
                y_binary = (y.reset_index(drop=True) == positive_label).astype(int)

                diagnostics["out_of_fold_roc_auc"] = _safe_float(
                    roc_auc_score(y_binary, positive_probabilities)
                )
                diagnostics["out_of_fold_pr_auc"] = _safe_float(
                    average_precision_score(y_binary, positive_probabilities)
                )
            else:
                diagnostics["out_of_fold_roc_auc"] = None
                diagnostics["out_of_fold_pr_auc"] = None

        except Exception as error:
            diagnostics["out_of_fold_roc_auc"] = None
            diagnostics["out_of_fold_pr_auc"] = None
            diagnostics["probability_diagnostics_error"] = str(error)

    return diagnostics


def train_candidate_models(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: Any,
    candidate_models: list[CandidateModel],
    task_type: str,
    base_task: str,
    preprocessing_config: dict[str, Any],
    random_state: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Train and evaluate candidate models using cross-validation.

    This function builds a full pipeline for each model:

        preprocessor → model

    The pipeline is evaluated using cross-validation, which keeps preprocessing
    leakage-safe.
    """

    scoring = build_scoring_strategy(
        task_type=task_type,
        y=y,
    )

    cv = build_cv_strategy(
        base_task=base_task,
        preprocessing_config=preprocessing_config,
        y=y,
        random_state=random_state,
    )

    primary_metric = preprocessing_config["metric_strategy"]["primary_metric"]

    model_results = []

    for candidate in candidate_models:
        try:
            full_pipeline = Pipeline(
                steps=[
                    ("preprocessor", clone(preprocessor)),
                    ("model", clone(candidate.estimator)),
                ]
            )

            cv_results = cross_validate(
                estimator=full_pipeline,
                X=X,
                y=y,
                scoring=scoring,
                cv=cv,
                return_train_score=True,
                error_score=np.nan,
            )

            metrics = {
                metric_name: _summarise_metric(
                    cv_results=cv_results,
                    metric_name=metric_name,
                    base_task=base_task,
                )
                for metric_name in scoring.keys()
            }

            primary_score = None

            if primary_metric in metrics:
                primary_score = metrics[primary_metric]["cv_mean"]

            classification_diagnostics = _create_classification_diagnostics(
                full_pipeline=full_pipeline,
                X=X,
                y=y,
                cv=cv,
                task_type=task_type,
            )

            result = {
                "model_id": candidate.model_id,
                "display_name": candidate.display_name,
                "family": candidate.family,
                "complexity": candidate.complexity,
                "interpretability": candidate.interpretability,
                "is_dummy_baseline": candidate.is_dummy_baseline,
                "status": "completed",
                "metrics": metrics,
                "classification_diagnostics": classification_diagnostics,
                "primary_metric": primary_metric,
                "primary_score": primary_score,
                "primary_metric_direction": _get_metric_direction(primary_metric),
                "fit_time_mean": _safe_float(np.nanmean(cv_results["fit_time"])),
                "fit_time_std": _safe_float(np.nanstd(cv_results["fit_time"])),
                "score_time_mean": _safe_float(np.nanmean(cv_results["score_time"])),
                "score_time_std": _safe_float(np.nanstd(cv_results["score_time"])),
                "error": None,
            }

        except Exception as error:
            result = {
                "model_id": candidate.model_id,
                "display_name": candidate.display_name,
                "family": candidate.family,
                "complexity": candidate.complexity,
                "interpretability": candidate.interpretability,
                "is_dummy_baseline": candidate.is_dummy_baseline,
                "status": "failed",
                "metrics": {},
                "classification_diagnostics": {},
                "primary_metric": primary_metric,
                "primary_score": None,
                "primary_metric_direction": _get_metric_direction(primary_metric),
                "fit_time_mean": None,
                "fit_time_std": None,
                "score_time_mean": None,
                "score_time_std": None,
                "error": str(error),
            }

        model_results.append(result)

    training_summary = {
        "task_type": task_type,
        "base_task": base_task,
        "models_attempted": len(candidate_models),
        "models_completed": sum(
            1 for result in model_results if result["status"] == "completed"
        ),
        "models_failed": sum(
            1 for result in model_results if result["status"] == "failed"
        ),
        "primary_metric": primary_metric,
        "primary_metric_direction": _get_metric_direction(primary_metric),
        "validation_method": preprocessing_config["validation_strategy"]["method"],
        "cv_folds": getattr(cv, "n_splits", None),
        "implemented_metrics": list(scoring.keys()),
        "unsupported_metrics_in_first_trainer": [],
        "leakage_control": (
            "Each model was evaluated as a full scikit-learn Pipeline, so preprocessing "
            "was fitted inside each training fold only."
        ),
    }

    return model_results, training_summary