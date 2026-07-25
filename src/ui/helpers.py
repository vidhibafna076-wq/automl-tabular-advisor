"""Formatting, state inspection, and CSV helpers for the Streamlit UI."""

from __future__ import annotations

from hashlib import sha256
from html import escape
from pathlib import Path
from typing import Any, Iterable, Mapping
import io
import json
import re

import pandas as pd
import streamlit as st

from .constants import PREVIEW_ROWS


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def safe_text(value: Any, fallback: str = "N/A") -> str:
    """Escape text before placing it inside custom HTML."""

    if value is None or value == "":
        return escape(fallback)
    return escape(str(value))


def format_metric(value: Any, digits: int = 4) -> float | None:
    """Return a rounded finite-looking number or None."""

    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return round(number, digits)


def display_value(value: Any, fallback: str = "N/A") -> str:
    formatted = format_metric(value)
    if formatted is not None:
        return f"{formatted:g}"
    if value is None or value == "":
        return fallback
    return str(value)


def humanise(value: Any, fallback: str = "N/A") -> str:
    if value is None or value == "":
        return fallback
    return str(value).replace("_", " ").strip().title()


def normalise_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "applied", "completed"}:
            return True
        if lowered in {"false", "no", "0", "skipped", "not_applied"}:
            return False
    return None


def safe_filename(filename: str, fallback: str = "dataset.csv") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name)
    return cleaned.strip("._") or fallback


def fingerprint_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def state_attr(state: Any, *names: str, default: Any = None) -> Any:
    """Read the first populated attribute/key across backend state versions."""

    for name in names:
        if isinstance(state, Mapping) and name in state:
            value = state[name]
        else:
            value = getattr(state, name, None)
        if value not in (None, "", [], {}):
            return value
    return default


def first_mapping(source: Any, *names: str) -> dict[str, Any]:
    value = state_attr(source, *names, default={})
    return value if isinstance(value, dict) else {}


def first_value(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = mapping.get(name)
        if value not in (None, ""):
            return value
    return default


def nested_metric(
    mapping: Mapping[str, Any],
    metric_name: str | None = None,
) -> tuple[str | None, float | None]:
    """Extract a primary metric and score from several common artifact shapes."""

    resolved_name = first_value(
        mapping,
        "primary_metric",
        "metric",
        "metric_name",
        default=metric_name,
    )
    direct = first_value(
        mapping,
        "primary_score",
        "holdout_score",
        "test_score",
        "score",
        "value",
    )
    direct_score = format_metric(direct)
    if direct_score is not None:
        return str(resolved_name) if resolved_name else None, direct_score

    metrics = mapping.get("metrics")
    if isinstance(metrics, Mapping):
        if resolved_name and resolved_name in metrics:
            candidate = metrics[resolved_name]
            if isinstance(candidate, Mapping):
                candidate = first_value(
                    candidate,
                    "score",
                    "value",
                    "holdout",
                    "test",
                    "mean",
                )
            score = format_metric(candidate)
            if score is not None:
                return str(resolved_name), score

        for name, candidate in metrics.items():
            if isinstance(candidate, Mapping):
                candidate = first_value(
                    candidate,
                    "score",
                    "value",
                    "holdout",
                    "test",
                    "mean",
                )
            score = format_metric(candidate)
            if score is not None:
                return str(name), score

    return str(resolved_name) if resolved_name else None, None


def tuning_evidence(state: Any) -> dict[str, Any]:
    artifact = first_mapping(
        state,
        "tuning_result",
        "tuning_summary",
        "hyperparameter_tuning",
        "model_tuning_result",
    )
    persistence = first_mapping(
        state,
        "model_persistence",
        "persistence_summary",
        "model_artifact_summary",
    )

    applied = normalise_bool(
        first_value(
            artifact,
            "tuning_applied",
            "applied",
            "was_tuned",
            default=first_value(persistence, "tuning_applied", "was_tuned"),
        )
    )
    params = first_value(
        artifact,
        "tuned_parameters",
        "best_params",
        "best_parameters",
        default=first_value(persistence, "tuned_parameters", "best_params", default={}),
    )
    baseline = format_metric(
        first_value(
            artifact,
            "baseline_score",
            "untuned_score",
            "score_before_tuning",
        )
    )
    tuned = format_metric(
        first_value(
            artifact,
            "tuned_score",
            "best_score",
            "score_after_tuning",
            "best_cv_score",
        )
    )
    improvement = format_metric(
        first_value(artifact, "improvement", "score_improvement", "absolute_improvement")
    )
    if improvement is None and baseline is not None and tuned is not None:
        improvement = round(tuned - baseline, 4)

    return {
        "available": bool(artifact or persistence),
        "applied": applied,
        "parameters": params if isinstance(params, Mapping) else {},
        "baseline_score": baseline,
        "tuned_score": tuned,
        "improvement": improvement,
        "reason": first_value(
            artifact,
            "reason",
            "skip_reason",
            "message",
            "status_message",
        ),
        "raw": artifact,
    }


def holdout_evidence(state: Any) -> dict[str, Any]:
    artifact = first_mapping(
        state,
        "holdout_evaluation",
        "holdout_evaluation_summary",
        "final_holdout_evaluation",
        "final_holdout_results",
        "holdout_result",
    )
    comparison = first_mapping(state, "comparison_summary")
    primary_metric = first_value(comparison, "primary_metric")
    metric, score = nested_metric(artifact, primary_metric)
    evaluated = normalise_bool(
        first_value(
            artifact,
            "evaluated",
            "holdout_evaluated",
            "completed",
            default=None,
        )
    )
    status = str(first_value(artifact, "status", "evaluation_status", default="")).lower()
    if evaluated is None:
        if score is not None:
            evaluated = True
        elif status in {"skipped", "not_available", "not_reserved", "not_evaluated"}:
            evaluated = False

    return {
        "available": bool(artifact),
        "evaluated": evaluated,
        "metric": metric,
        "score": score,
        "reason": first_value(
            artifact,
            "skip_reason",
            "reason",
            "message",
            "status_message",
        ),
        "rows": first_value(
            artifact,
            "rows",
            "holdout_rows",
            "test_rows",
            "sample_count",
        ),
        "raw": artifact,
    }


def best_leaderboard_item(state: Any) -> dict[str, Any] | None:
    comparison = first_mapping(state, "comparison_summary")
    leaderboard = state_attr(state, "leaderboard", default=[]) or []
    model_id = first_value(
        comparison,
        "best_useful_model_id",
        "selected_model_id",
        "best_model_id",
    )
    if model_id:
        match = next(
            (item for item in leaderboard if item.get("model_id") == model_id),
            None,
        )
        if match:
            return match
    return leaderboard[0] if leaderboard else None


def expected_feature_columns(state: Any) -> list[str]:
    """Infer the raw prediction schema from the saved experiment record."""

    profile = first_mapping(state, "profile")
    config = first_mapping(state, "preprocessing_config")
    target = state_attr(state, "target_column")
    dropped = set(config.get("columns_to_drop", []) or [])

    candidates: list[str] = []
    for key in (
        "feature_columns",
        "input_columns",
        "raw_feature_columns",
        "numerical_columns",
        "categorical_columns",
        "possible_date_columns",
        "text_columns",
    ):
        values = profile.get(key, [])
        if isinstance(values, Iterable) and not isinstance(values, (str, bytes, Mapping)):
            candidates.extend(str(value) for value in values)

    direct = state_attr(
        state,
        "feature_columns",
        "expected_feature_columns",
        default=[],
    )
    if isinstance(direct, Iterable) and not isinstance(direct, (str, bytes, Mapping)):
        candidates = [str(value) for value in direct] + candidates

    unique: list[str] = []
    seen: set[str] = set()
    for column in candidates:
        if column == target or column in dropped or column in seen:
            continue
        seen.add(column)
        unique.append(column)
    return unique


def resolve_artifact_path(value: str | Path | None, run_dir: Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if run_dir:
        candidate = run_dir / path
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / path


@st.cache_data(show_spinner=False, max_entries=8)
def parse_csv_preview(
    file_bytes: bytes,
    max_rows: int = PREVIEW_ROWS,
) -> tuple[pd.DataFrame, bool]:
    """Parse at most max_rows + 1 so setup reruns do not reread the full CSV."""

    preview = pd.read_csv(io.BytesIO(file_bytes), nrows=max_rows + 1)
    truncated = len(preview) > max_rows
    if truncated:
        preview = preview.iloc[:max_rows].copy()
    return preview, truncated


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def friendly_failure(context: str, reference: str | None = None) -> str:
    suffix = f" Reference: {reference}." if reference else ""
    if context == "csv":
        return (
            "The CSV could not be read. Check its delimiter, encoding, header row, "
            f"and quoting, then try again.{suffix}"
        )
    if context == "prediction":
        return (
            "Prediction could not be completed. Check the selected model and input "
            f"schema, then review the local error log if needed.{suffix}"
        )
    return (
        "The experiment could not be completed. Review the local run log and "
        f"try again with a smaller or corrected dataset.{suffix}"
    )
