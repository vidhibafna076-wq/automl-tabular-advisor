from dataclasses import dataclass
from typing import Any

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge


@dataclass
class CandidateModel:
    """
    Represents one approved model candidate.

    The estimator is the actual scikit-learn model object.
    The remaining fields are metadata used for explanation and reporting.
    """

    model_id: str
    display_name: str
    base_task: str
    family: str
    estimator: Any
    strengths: list[str]
    limitations: list[str]
    recommended_for: list[str]
    complexity: str
    interpretability: str
    supports_predict_proba: bool
    is_dummy_baseline: bool = False


def _classification_models(random_state: int = 42) -> list[CandidateModel]:
    """
    Return approved baseline classification models.
    """

    return [
        CandidateModel(
            model_id="dummy_most_frequent",
            display_name="Dummy Classifier",
            base_task="classification",
            family="baseline",
            estimator=DummyClassifier(strategy="most_frequent"),
            strengths=[
                "Provides a naive baseline for comparison.",
                "Helps check whether real models improve beyond simple guessing.",
            ],
            limitations=[
                "Does not learn real patterns from the data.",
                "Should never be selected as the final useful model unless all real models fail.",
            ],
            recommended_for=[
                "Baseline comparison",
                "Reliability checking",
            ],
            complexity="very_low",
            interpretability="high",
            supports_predict_proba=True,
            is_dummy_baseline=True,
        ),
        CandidateModel(
            model_id="logistic_regression",
            display_name="Logistic Regression",
            base_task="classification",
            family="linear",
            estimator=LogisticRegression(
                max_iter=1000,
                random_state=random_state,
            ),
            strengths=[
                "Fast to train.",
                "Good interpretable baseline for classification.",
                "Works well when relationships are approximately linear.",
            ],
            limitations=[
                "May underperform when relationships are highly nonlinear.",
                "Can be affected by outliers and feature scaling.",
            ],
            recommended_for=[
                "Interpretable classification baseline",
                "Small to medium tabular datasets",
            ],
            complexity="low",
            interpretability="high",
            supports_predict_proba=True,
        ),
        CandidateModel(
            model_id="random_forest_classifier",
            display_name="Random Forest Classifier",
            base_task="classification",
            family="tree_ensemble",
            estimator=RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1,
            ),
            strengths=[
                "Handles nonlinear relationships well.",
                "Works well on many tabular datasets.",
                "Less sensitive to feature scaling than linear models.",
            ],
            limitations=[
                "Less interpretable than logistic regression.",
                "Can overfit on small datasets if not controlled.",
                "Can be slower than simple linear models.",
            ],
            recommended_for=[
                "General-purpose tabular classification",
                "Mixed numerical and categorical datasets",
            ],
            complexity="medium",
            interpretability="medium",
            supports_predict_proba=True,
        ),
        CandidateModel(
            model_id="hist_gradient_boosting_classifier",
            display_name="Histogram Gradient Boosting Classifier",
            base_task="classification",
            family="boosting",
            estimator=HistGradientBoostingClassifier(
                random_state=random_state,
            ),
            strengths=[
                "Strong performance on many structured datasets.",
                "Can model complex nonlinear relationships.",
                "Often competitive with advanced boosting approaches.",
            ],
            limitations=[
                "Less interpretable than linear models.",
                "May need tuning for best performance.",
                "Can be unreliable on extremely small datasets.",
            ],
            recommended_for=[
                "Higher-performing tabular classification baseline",
                "Nonlinear structured data problems",
            ],
            complexity="medium_high",
            interpretability="low",
            supports_predict_proba=True,
        ),
    ]


def _regression_models(random_state: int = 42) -> list[CandidateModel]:
    """
    Return approved baseline regression models.
    """

    return [
        CandidateModel(
            model_id="dummy_mean",
            display_name="Dummy Regressor",
            base_task="regression",
            family="baseline",
            estimator=DummyRegressor(strategy="mean"),
            strengths=[
                "Provides a naive baseline for comparison.",
                "Helps check whether real models improve beyond predicting the average.",
            ],
            limitations=[
                "Does not learn real patterns from the data.",
                "Should never be selected as the final useful model unless all real models fail.",
            ],
            recommended_for=[
                "Baseline comparison",
                "Reliability checking",
            ],
            complexity="very_low",
            interpretability="high",
            supports_predict_proba=False,
            is_dummy_baseline=True,
        ),
        CandidateModel(
            model_id="ridge_regression",
            display_name="Ridge Regression",
            base_task="regression",
            family="linear",
            estimator=Ridge(),
            strengths=[
                "Fast to train.",
                "Good regularised linear baseline.",
                "More stable than ordinary linear regression when features are correlated.",
            ],
            limitations=[
                "May underperform on highly nonlinear problems.",
                "Can be affected by outliers.",
            ],
            recommended_for=[
                "Interpretable regression baseline",
                "Small to medium tabular datasets",
            ],
            complexity="low",
            interpretability="high",
            supports_predict_proba=False,
        ),
        CandidateModel(
            model_id="random_forest_regressor",
            display_name="Random Forest Regressor",
            base_task="regression",
            family="tree_ensemble",
            estimator=RandomForestRegressor(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1,
            ),
            strengths=[
                "Handles nonlinear relationships well.",
                "Works well on many tabular regression problems.",
                "Less sensitive to feature scaling than linear models.",
            ],
            limitations=[
                "Less interpretable than linear regression.",
                "Can overfit on small datasets.",
                "Can be slower than simple models.",
            ],
            recommended_for=[
                "General-purpose tabular regression",
                "Mixed numerical and categorical datasets",
            ],
            complexity="medium",
            interpretability="medium",
            supports_predict_proba=False,
        ),
        CandidateModel(
            model_id="hist_gradient_boosting_regressor",
            display_name="Histogram Gradient Boosting Regressor",
            base_task="regression",
            family="boosting",
            estimator=HistGradientBoostingRegressor(
                random_state=random_state,
            ),
            strengths=[
                "Strong performance on many structured regression datasets.",
                "Can model complex nonlinear relationships.",
                "Often performs well with tabular data.",
            ],
            limitations=[
                "Less interpretable than linear models.",
                "May need tuning for best performance.",
                "Can be unreliable on extremely small datasets.",
            ],
            recommended_for=[
                "Higher-performing tabular regression baseline",
                "Nonlinear structured data problems",
            ],
            complexity="medium_high",
            interpretability="low",
            supports_predict_proba=False,
        ),
    ]


def get_candidate_models(
    base_task: str,
    random_state: int = 42,
) -> list[CandidateModel]:
    """
    Return approved candidate models for the detected task family.
    """

    if base_task == "classification":
        return _classification_models(random_state=random_state)

    if base_task == "regression":
        return _regression_models(random_state=random_state)

    raise ValueError(
        f"Unsupported base task: {base_task}. Expected 'classification' or 'regression'."
    )


def candidate_model_to_summary(model: CandidateModel) -> dict[str, Any]:
    """
    Convert a CandidateModel into a JSON-safe summary.

    The estimator object is intentionally excluded because it cannot be saved
    directly to experiment_state.json.
    """

    return {
        "model_id": model.model_id,
        "display_name": model.display_name,
        "base_task": model.base_task,
        "family": model.family,
        "strengths": model.strengths,
        "limitations": model.limitations,
        "recommended_for": model.recommended_for,
        "complexity": model.complexity,
        "interpretability": model.interpretability,
        "supports_predict_proba": model.supports_predict_proba,
        "is_dummy_baseline": model.is_dummy_baseline,
    }


def get_model_registry_summary(
    base_task: str,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """
    Return a JSON-safe summary of the approved model registry.
    """

    models = get_candidate_models(
        base_task=base_task,
        random_state=random_state,
    )

    return [
        candidate_model_to_summary(model)
        for model in models
    ]


def get_model_by_id(
    base_task: str,
    model_id: str,
    random_state: int = 42,
) -> CandidateModel:
    """
    Retrieve one model from the approved registry by model_id.

    This will be useful later when the training tool needs to build pipelines.
    """

    models = get_candidate_models(
        base_task=base_task,
        random_state=random_state,
    )

    for model in models:
        if model.model_id == model_id:
            return model

    raise ValueError(
        f"Model '{model_id}' not found in the approved {base_task} registry."
    )