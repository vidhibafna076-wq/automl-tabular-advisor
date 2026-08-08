"""Tests for safe saved-model environment compatibility checks."""

from typing import Any

import pytest

import src.inference as inference_module


CURRENT_ENVIRONMENT: dict[str, Any] = {
    "schema_version": 1,
    "python_version": "3.13.3",
    "python_implementation": "CPython",
    "platform": "Windows-test",
    "package_versions": {
        "scikit-learn": "1.9.0",
        "numpy": "2.5.0",
        "joblib": "1.5.3",
    },
}


def test_legacy_artifact_is_allowed_but_marked_unverified(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        inference_module,
        "runtime_environment",
        lambda: CURRENT_ENVIRONMENT,
    )

    result = inference_module._validate_artifact_environment({})

    assert result["status"] == "legacy_unverified"
    assert result["compatible"] is None
    assert result["warnings"]


def test_matching_artifact_environment_is_compatible(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        inference_module,
        "runtime_environment",
        lambda: CURRENT_ENVIRONMENT,
    )

    result = inference_module._validate_artifact_environment(
        {"artifact_environment": CURRENT_ENVIRONMENT}
    )

    assert result["status"] == "compatible"
    assert result["compatible"] is True
    assert result["warnings"] == []


def test_different_sklearn_version_is_rejected_before_loading(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        inference_module,
        "runtime_environment",
        lambda: CURRENT_ENVIRONMENT,
    )
    saved = {
        **CURRENT_ENVIRONMENT,
        "package_versions": {
            **CURRENT_ENVIRONMENT["package_versions"],
            "scikit-learn": "1.8.2",
        },
    }

    with pytest.raises(
        ValueError,
        match=r"incompatible.*scikit-learn.*saved with 1\.8\.2.*running with 1\.9\.0",
    ):
        inference_module._validate_artifact_environment(
            {"artifact_environment": saved}
        )


def test_different_python_major_version_is_rejected(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        inference_module,
        "runtime_environment",
        lambda: CURRENT_ENVIRONMENT,
    )
    saved = {
        **CURRENT_ENVIRONMENT,
        "python_version": "2.7.18",
    }

    with pytest.raises(ValueError, match=r"incompatible.*Python"):
        inference_module._validate_artifact_environment(
            {"artifact_environment": saved}
        )
