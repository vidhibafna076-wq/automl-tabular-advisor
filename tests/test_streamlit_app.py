"""Browser-free smoke tests for the Streamlit application entry point."""

from streamlit.testing.v1 import AppTest


def _app() -> AppTest:
    return AppTest.from_file("app.py", default_timeout=15).run()


def test_app_renders_training_workspace_without_exceptions() -> None:
    app = _app()

    assert not app.exception
    assert app.radio[0].label == "Workspace"
    assert app.radio[0].value == "Train and evaluate"
    assert app.radio[1].label == "Dataset source"
    assert app.radio[1].value == "Upload CSV"

    assert app.selectbox[0].label == "Prediction target"
    assert app.selectbox[0].disabled is True
    assert app.text_area[0].label == "Experiment objective"
    assert app.text_area[0].disabled is True
    assert app.checkbox[0].label == "Pre-approve dropping likely identifier columns"
    assert app.checkbox[0].value is False
    assert app.button[0].label == "Start evaluation"
    assert app.button[0].disabled is True


def test_sample_dataset_populates_an_executable_experiment() -> None:
    app = _app()
    app.radio[1].set_value("Use sample dataset").run()

    assert not app.exception
    assert any(message.value == "loan_sample.csv selected" for message in app.success)

    target = next(item for item in app.selectbox if item.label == "Prediction target")
    task_type = next(item for item in app.selectbox if item.label == "Task type")
    runtime_limit = next(
        item for item in app.selectbox if item.label == "Automatic run limit"
    )
    objective = next(
        item for item in app.text_area if item.label == "Experiment objective"
    )
    start = next(item for item in app.button if item.label == "Start evaluation")

    assert target.value == "Loan_Status"
    assert target.disabled is False
    assert task_type.value == "Auto-detect"
    assert runtime_limit.value == "20 minutes"
    assert objective.value == "Predict whether a loan application will be approved"
    assert objective.disabled is False
    assert app.checkbox[0].label == "Pre-approve dropping likely identifier columns"
    assert app.checkbox[0].value is True
    assert start.disabled is False


def test_approval_form_can_submit_checkbox_and_decision_together() -> None:
    script = """
from types import SimpleNamespace
import streamlit as st
from src.ui.results import render_approval_panel

state = SimpleNamespace(
    preprocessing_config={
        "pending_approvals": [
            {
                "title": "Drop ID",
                "reason": "The column looks like an identifier.",
                "columns": ["Loan_ID"],
            }
        ]
    }
)

def resume(decisions):
    st.session_state["captured_decisions"] = decisions

render_approval_panel(state, {"ui_run_id": "test-run"}, resume)
"""
    app = AppTest.from_string(script, default_timeout=15).run()

    assert not app.exception
    assert app.button[0].label == "Apply decisions and resume"
    assert app.button[0].disabled is False

    app.radio[0].set_value("Approve the proposed change")
    app.checkbox[0].check()
    app.button[0].click().run()

    assert not app.exception
    assert not app.error
    assert app.session_state["captured_decisions"] == [{"action": "approve"}]
