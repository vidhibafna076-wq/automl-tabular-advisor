from datetime import datetime
from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src.data_loader import load_dataset
from src.model_registry import get_model_by_id
from src.preprocessing import build_preprocessor, prepare_features_and_target
from src.state import ExperimentState


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
    Convert values to JSON-safe floats.
    """

    try:
        value = float(value)

        if np.isnan(value) or np.isinf(value):
            return None

        return value

    except Exception:
        return None


def _extract_feature_names(fitted_pipeline: Pipeline) -> list[str]:
    """
    Extract feature names from the fitted preprocessing step where possible.
    """

    try:
        preprocessor = fitted_pipeline.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()
        return [str(name) for name in feature_names]

    except Exception:
        return []


def _extract_feature_importance(
    fitted_pipeline: Pipeline,
    max_features: int = 20,
) -> dict[str, Any]:
    """
    Extract simple feature importance when the final estimator supports it.

    Supported cases:
    - tree-based models with feature_importances_
    - linear models with coef_

    If feature importance is unavailable, return an honest unavailable summary.
    """

    model = fitted_pipeline.named_steps.get("model")
    feature_names = _extract_feature_names(fitted_pipeline)

    if model is None:
        return {
            "status": "unavailable",
            "reason": "No model step was found in the fitted pipeline.",
            "top_features": [],
        }

    raw_importance = None
    importance_type = None

    if hasattr(model, "feature_importances_"):
        raw_importance = model.feature_importances_
        importance_type = "tree_feature_importance"

    elif hasattr(model, "coef_"):
        coefficients = model.coef_

        if len(coefficients.shape) == 1:
            raw_importance = np.abs(coefficients)

        else:
            raw_importance = np.mean(np.abs(coefficients), axis=0)

        importance_type = "linear_absolute_coefficient"

    else:
        return {
            "status": "unavailable",
            "reason": (
                "The selected model does not expose feature_importances_ "
                "or coef_."
            ),
            "top_features": [],
        }

    if raw_importance is None:
        return {
            "status": "unavailable",
            "reason": "Feature importance values could not be extracted.",
            "top_features": [],
        }

    raw_importance = np.array(raw_importance).ravel()

    if not feature_names or len(feature_names) != len(raw_importance):
        feature_names = [
            f"feature_{index}"
            for index in range(len(raw_importance))
        ]

    ranked_indices = np.argsort(raw_importance)[::-1][:max_features]

    top_features = []

    for rank, index in enumerate(ranked_indices, start=1):
        top_features.append(
            {
                "rank": rank,
                "feature": feature_names[index],
                "importance": _safe_float(raw_importance[index]),
            }
        )

    return {
        "status": "available",
        "importance_type": importance_type,
        "top_features": top_features,
    }


def _save_json(data: dict[str, Any], output_path: Path) -> str:
    """
    Save a dictionary as JSON.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    return str(output_path)


def model_persistence_tool(
    state: ExperimentState,
    output_dir: str = "outputs/models",
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Save a final fitted model pipeline only when the critic approves.

    Conservative rule:
    - If critic decision is do_not_recommend_model, do not save a deployable model.
    - If critic decision is recommend_with_caution, do not save a deployable model yet.
    - Only save when critic decision is recommend_candidate_model.
    """

    tool_name = "model_persistence_tool"
    state.iteration += 1

    _add_tool_event(
        state=state,
        tool_name=tool_name,
        status="started",
        message="Started model persistence check.",
    )

    try:
        if state.status != "reliability_critique_completed":
            raise ValueError(
                "Cannot check model persistence before reliability critique is completed."
            )

        critic_report = state.critic_report or {}
        decision = critic_report.get("recommendation_decision")
        selected_candidate = critic_report.get("selected_candidate")

        if decision != "recommend_candidate_model":
            summary = {
                "status": "skipped",
                "reason": (
                    "No model was saved because the critic did not approve "
                    "a deployable model recommendation."
                ),
                "critic_decision": decision,
                "artifact_path": None,
                "metadata_path": None,
            }

            state.model_artifact_summary = summary
            state.feature_importance_summary = {
                "status": "skipped",
                "reason": "Feature importance was skipped because no final model was saved.",
                "top_features": [],
            }
            state.status = "model_persistence_checked"
            state.completed_steps.append("checked_model_persistence")

            _add_tool_event(
                state=state,
                tool_name=tool_name,
                status="completed",
                message=summary["reason"],
            )

            return {
                "success": True,
                "state": state,
                "model_artifact_summary": summary,
                "feature_importance_summary": state.feature_importance_summary,
                "error": None,
            }
        
        validation_summary = state.validation_summary or {}
        holdout_result = state.holdout_result or {}

        if (
            validation_summary.get("holdout_used")
            and holdout_result.get("status") != "completed"
        ):
            summary = {
                "status": "skipped",
                "reason": (
                    "No model was saved because a final holdout was reserved but has "
                    "not yet been evaluated."
                ),
                "critic_decision": decision,
                "artifact_path": None,
                "metadata_path": None,
            }

            state.model_artifact_summary = summary
            state.feature_importance_summary = {
                "status": "skipped",
                "reason": (
                    "Feature importance was skipped because no final model was saved."
                ),
                "top_features": [],
            }
            state.status = "model_persistence_checked"
            state.completed_steps.append("checked_model_persistence")

            _add_tool_event(
                state=state,
                tool_name=tool_name,
                status="completed",
                message=summary["reason"],
            )

            return {
                "success": True,
                "state": state,
                "model_artifact_summary": summary,
                "feature_importance_summary": state.feature_importance_summary,
                "error": None,
            }

        if not selected_candidate:
            raise ValueError(
                "Critic approved model persistence, but no selected candidate was found."
            )

        model_id = selected_candidate["model_id"]

        dataset = load_dataset(
            file_path=state.dataset_path,
            target_column=state.target_column,
        )

        X, y = prepare_features_and_target(
            df=dataset.df,
            preprocessing_config=state.preprocessing_config,
        )

        preprocessor = build_preprocessor(state.preprocessing_config)

        candidate_model = get_model_by_id(
            base_task=state.base_task,
            model_id=model_id,
            random_state=random_state,
        )

        fitted_pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", clone(candidate_model.estimator)),
            ]
        )

        fitted_pipeline.fit(X, y)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        artifact_path = output_path / f"{timestamp}_{model_id}_pipeline.joblib"
        metadata_path = output_path / f"{timestamp}_{model_id}_metadata.json"

        joblib.dump(fitted_pipeline, artifact_path)

        feature_importance_summary = _extract_feature_importance(
            fitted_pipeline=fitted_pipeline,
        )

        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_path": state.dataset_path,
            "target_column": state.target_column,
            "task_type": state.task_type,
            "base_task": state.base_task,
            "model_id": model_id,
            "display_name": selected_candidate.get("display_name"),
            "primary_metric": selected_candidate.get("primary_metric"),
            "primary_score": selected_candidate.get("primary_score"),
            "critic_decision": decision,
            "preprocessing_config": state.preprocessing_config,
            "feature_importance_summary": feature_importance_summary,
        }

        saved_metadata_path = _save_json(
            data=metadata,
            output_path=metadata_path,
        )

        summary = {
            "status": "saved",
            "reason": "Final fitted pipeline was saved because the critic approved the candidate model.",
            "critic_decision": decision,
            "model_id": model_id,
            "display_name": selected_candidate.get("display_name"),
            "artifact_path": str(artifact_path),
            "metadata_path": saved_metadata_path,
        }

        state.model_artifact_summary = summary
        state.feature_importance_summary = feature_importance_summary
        state.status = "model_persistence_checked"
        state.completed_steps.append("checked_model_persistence")

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="completed",
            message=f"Saved final fitted pipeline to {artifact_path}.",
        )

        return {
            "success": True,
            "state": state,
            "model_artifact_summary": summary,
            "feature_importance_summary": feature_importance_summary,
            "error": None,
        }

    except Exception as error:
        state.status = "model_persistence_failed"
        state.warnings.append(str(error))
        state.completed_steps.append("failed_model_persistence")

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="failed",
            message=str(error),
        )

        return {
            "success": False,
            "state": state,
            "model_artifact_summary": {},
            "feature_importance_summary": {},
            "error": str(error),
        }