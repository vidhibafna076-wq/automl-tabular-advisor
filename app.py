from datetime import datetime
from pathlib import Path
from typing import Any
import io

import pandas as pd
import streamlit as st

from src.agents.orchestrator_agent import run_orchestrator
from src.state import ExperimentState, save_state


st.set_page_config(
    page_title="Agentic AutoML Advisor",
    page_icon="🤖",
    layout="wide",
)


def _safe_filename(filename: str) -> str:
    """
    Convert an uploaded filename into a safer local filename.
    """

    allowed_chars = []

    for char in filename:
        if char.isalnum() or char in [".", "_", "-"]:
            allowed_chars.append(char)
        else:
            allowed_chars.append("_")

    return "".join(allowed_chars)


def _save_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded CSV file into data/raw and return its local path.
    """

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = _safe_filename(uploaded_file.name)

    output_path = raw_dir / f"{timestamp}_{safe_name}"
    output_path.write_bytes(uploaded_file.getvalue())

    return str(output_path)


def _format_metric(value: Any) -> Any:
    """
    Format metric values for display.
    """

    if value is None:
        return None

    try:
        return round(float(value), 4)
    except Exception:
        return value


def make_quality_dataframe(quality_issues: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert quality issues into a display dataframe.
    """

    if not quality_issues:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "Severity": issue["severity"],
                "Issue": issue["issue"],
                "Finding": issue["finding"],
                "Recommendation": issue["recommendation"],
            }
            for issue in quality_issues
        ]
    )


def make_leaderboard_dataframe(leaderboard: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert leaderboard into a display dataframe.
    """

    if not leaderboard:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "Rank": item["rank"],
                "Model": item["display_name"],
                "Family": item["family"],
                "Primary Metric": item["primary_metric"],
                "Primary Score": item["primary_score"],
                "CV Std": item["primary_cv_std"],
                "Train Score": item["primary_train_score"],
                "Overfit Gap": item["overfit_gap"],
                "Beats Dummy": item["improvement_over_dummy"]["beats_dummy"],
                "Reliability": item["reliability_level"],
            }
            for item in leaderboard
        ]
    )


def make_training_metrics_dataframe(model_results: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert nested model metrics into a long dataframe.
    """

    rows = []

    for result in model_results:
        for metric_name, metric_values in result.get("metrics", {}).items():
            rows.append(
                {
                    "Model": result["display_name"],
                    "Metric": metric_name,
                    "CV Mean": _format_metric(metric_values.get("cv_mean")),
                    "CV Std": _format_metric(metric_values.get("cv_std")),
                    "Train Mean": _format_metric(metric_values.get("train_mean")),
                }
            )

    return pd.DataFrame(rows)


def make_timeline_dataframe(tool_history: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert tool history into a display dataframe.
    """

    if not tool_history:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "Timestamp": event["timestamp"],
                "Tool": event["tool_name"],
                "Status": event["status"],
                "Message": event["message"],
            }
            for event in tool_history
        ]
    )


def show_status_cards(state: ExperimentState) -> None:
    """
    Show high-level status cards.
    """

    critic_report = state.critic_report or {}
    training_summary = state.training_summary or {}
    comparison_summary = state.comparison_summary or {}

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Current Status", state.status)

    with col2:
        st.metric(
            "Reliability",
            critic_report.get("overall_reliability", "N/A"),
        )

    with col3:
        st.metric(
            "Models Completed",
            training_summary.get("models_completed", 0),
        )

    with col4:
        st.metric(
            "Best Useful Model",
            comparison_summary.get("best_useful_display_name") or "None",
        )


def show_dataset_tab(state: ExperimentState) -> None:
    """
    Display dataset profile and quality information.
    """

    st.subheader("Dataset Overview")

    profile = state.profile or {}

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", profile.get("rows", "N/A"))

    with col2:
        st.metric("Columns", profile.get("columns", "N/A"))

    with col3:
        st.metric("Features", profile.get("feature_count", "N/A"))

    with col4:
        st.metric("Task Type", state.task_type or "N/A")

    st.markdown("### Numerical Features")
    st.write(profile.get("numerical_columns", []))

    st.markdown("### Categorical Features")
    st.write(profile.get("categorical_columns", []))

    st.markdown("### Data Quality Issues")
    quality_df = make_quality_dataframe(state.quality_issues)

    if quality_df.empty:
        st.info("No quality issues recorded.")
    else:
        st.dataframe(quality_df, use_container_width=True)


def show_preprocessing_tab(state: ExperimentState) -> None:
    """
    Display preprocessing configuration.
    """

    st.subheader("Preprocessing Strategy")

    config = state.preprocessing_config or {}

    if not config:
        st.info("No preprocessing configuration available.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Columns Dropped")
        st.write(config.get("columns_to_drop", []))

        st.markdown("### Numerical Features")
        st.write(config.get("numerical_features", []))

    with col2:
        st.markdown("### Categorical Features")
        st.write(config.get("categorical_features", []))

        st.markdown("### Pending Approvals")
        st.write(config.get("pending_approvals", []))

    st.markdown("### Missing Value Strategy")
    st.json(config.get("missing_value_strategy", {}))

    st.markdown("### Validation Strategy")
    st.json(config.get("validation_strategy", {}))

    st.markdown("### Metric Strategy")
    st.json(config.get("metric_strategy", {}))


def show_results_tab(state: ExperimentState) -> None:
    """
    Display model results and leaderboard.
    """

    st.subheader("Model Leaderboard")

    leaderboard_df = make_leaderboard_dataframe(state.leaderboard)

    if leaderboard_df.empty:
        st.info("No leaderboard available.")
    else:
        st.dataframe(leaderboard_df, use_container_width=True)

    st.subheader("Training Metrics")

    metrics_df = make_training_metrics_dataframe(state.model_results)

    if metrics_df.empty:
        st.info("No training metrics available.")
    else:
        st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Model Diagnostics")

    for result in state.model_results:
        diagnostics = result.get("classification_diagnostics", {})

        with st.expander(result["display_name"]):
            st.markdown(f"**Model ID:** `{result['model_id']}`")
            st.markdown(f"**Primary score:** `{result['primary_score']}`")

            if diagnostics:
                st.markdown("#### Classification Diagnostics")
                st.write("Positive label:", diagnostics.get("positive_label"))
                st.write("Out-of-fold ROC-AUC:", diagnostics.get("out_of_fold_roc_auc"))
                st.write("Out-of-fold PR-AUC:", diagnostics.get("out_of_fold_pr_auc"))

                st.markdown("#### Confusion Matrix")
                matrix = diagnostics.get("confusion_matrix", [])
                labels = diagnostics.get("labels", [])

                if matrix:
                    matrix_df = pd.DataFrame(
                        matrix,
                        index=[f"Actual {label}" for label in labels],
                        columns=[f"Predicted {label}" for label in labels],
                    )
                    st.dataframe(matrix_df, use_container_width=True)
                else:
                    st.info("No confusion matrix available.")

                st.markdown("#### Class-Level Metrics")
                class_metrics = diagnostics.get("class_level_metrics", {})

                if class_metrics:
                    class_df = pd.DataFrame.from_dict(
                        class_metrics,
                        orient="index",
                    )
                    st.dataframe(class_df, use_container_width=True)
                else:
                    st.info("No class-level metrics available.")


def show_critic_tab(state: ExperimentState) -> None:
    """
    Display critic report.
    """

    st.subheader("Reliability Critic Report")

    critic_report = state.critic_report or {}

    if not critic_report:
        st.info("No critic report available.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Overall Reliability",
            critic_report.get("overall_reliability", "N/A"),
        )

    with col2:
        st.metric(
            "Decision",
            critic_report.get("recommendation_decision", "N/A"),
        )

    with col3:
        st.metric(
            "Can Tune?",
            str(critic_report.get("can_proceed_to_tuning", "N/A")),
        )

    st.markdown("### Findings")

    findings = critic_report.get("findings", [])

    if not findings:
        st.success("No critic findings.")
    else:
        for finding in findings:
            severity = finding["severity"].upper()
            st.markdown(f"#### [{severity}] {finding['issue']}")
            st.write(f"**Evidence:** {finding['evidence']}")
            st.write(f"**Recommendation:** {finding['recommendation']}")

    st.markdown("### Next Actions")
    for action in critic_report.get("next_actions", []):
        st.write(f"- {action}")


def show_report_tab(state: ExperimentState) -> None:
    """
    Display final report and download button.
    """

    st.subheader("Final Report")

    report_path = state.final_report_path

    if not report_path:
        st.info("No final report path available.")
        return

    path = Path(report_path)

    if not path.exists():
        st.warning(f"Report path was recorded, but file was not found: {report_path}")
        return

    report_text = path.read_text(encoding="utf-8")

    st.download_button(
        label="Download final_report.md",
        data=report_text,
        file_name="final_report.md",
        mime="text/markdown",
    )

    with st.expander("Preview final report", expanded=False):
        st.markdown(report_text)


def show_timeline_tab(state: ExperimentState) -> None:
    """
    Display tool execution timeline.
    """

    st.subheader("Tool Execution Timeline")

    timeline_df = make_timeline_dataframe(state.tool_history)

    if timeline_df.empty:
        st.info("No tool history available.")
    else:
        st.dataframe(timeline_df, use_container_width=True)


st.title("🤖 Agentic AutoML Advisor")
st.caption("A controlled agentic workflow for tabular AutoML experiments.")

st.markdown(
    """
This app runs the full AutoML advisor pipeline:

**Inspect → Plan → Preprocess → Train → Compare → Critique → Report**
"""
)

with st.sidebar:
    st.header("Experiment Setup")

    use_sample_dataset = st.checkbox(
        "Use built-in sample loan dataset",
        value=True,
    )

    uploaded_file = None
    df_preview = None
    dataset_path_for_run = None

    if use_sample_dataset:
        sample_path = Path("data/sample/loan_sample.csv")

        if sample_path.exists():
            dataset_path_for_run = str(sample_path)
            df_preview = pd.read_csv(sample_path)
            st.success("Sample dataset loaded.")
        else:
            st.error("Sample dataset not found at data/sample/loan_sample.csv")
    else:
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
        )

        if uploaded_file is not None:
            try:
                df_preview = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
                st.success("Uploaded dataset loaded.")
            except Exception as error:
                st.error(f"Could not read uploaded CSV: {error}")

    if df_preview is not None:
        target_column = st.selectbox(
            "Select target column",
            options=list(df_preview.columns),
        )

        user_objective = st.text_area(
            "User objective",
            value="Predict whether a loan application will be approved",
        )

        approve_drop_id_columns = st.checkbox(
            "Approve dropping possible ID columns",
            value=True,
            help=(
                "If checked, the workflow may drop columns that look like IDs, "
                "such as Loan_ID or Customer_ID."
            ),
        )

        run_button = st.button(
            "Run AutoML Advisor",
            type="primary",
        )

    else:
        target_column = None
        user_objective = None
        approve_drop_id_columns = False
        run_button = False


if df_preview is not None:
    st.subheader("Dataset Preview")
    st.dataframe(df_preview.head(20), use_container_width=True)


if run_button:
    with st.spinner("Running orchestrated AutoML workflow..."):
        try:
            if use_sample_dataset:
                final_dataset_path = dataset_path_for_run
            else:
                final_dataset_path = _save_uploaded_file(uploaded_file)

            state = ExperimentState(
                dataset_path=final_dataset_path,
                target_column=target_column,
                user_objective=user_objective,
            )

            state.completed_steps.append("created_experiment_state")

            orchestrator_result = run_orchestrator(
                state=state,
                approve_drop_id_columns=approve_drop_id_columns,
            )

            state = orchestrator_result["state"]

            save_state(state, "outputs/reports/experiment_state.json")

            st.session_state["last_state"] = state
            st.session_state["last_orchestrator_result"] = orchestrator_result

            st.success(f"Workflow completed with status: {state.status}")

        except Exception as error:
            st.error(f"Workflow failed: {error}")


if "last_state" in st.session_state:
    state = st.session_state["last_state"]
    orchestrator_result = st.session_state["last_orchestrator_result"]

    st.divider()

    show_status_cards(state)

    st.markdown("### Orchestrator Summary")
    st.write(f"**Success:** {orchestrator_result['success']}")
    st.write(f"**Stop reason:** {orchestrator_result['stop_reason']}")

    with st.expander("Actions taken"):
        st.dataframe(
            pd.DataFrame(orchestrator_result["actions_taken"]),
            use_container_width=True,
        )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Dataset",
            "Preprocessing",
            "Results",
            "Critic",
            "Final Report",
            "Timeline",
        ]
    )

    with tab1:
        show_dataset_tab(state)

    with tab2:
        show_preprocessing_tab(state)

    with tab3:
        show_results_tab(state)

    with tab4:
        show_critic_tab(state)

    with tab5:
        show_report_tab(state)

    with tab6:
        show_timeline_tab(state)

else:
    st.info("Configure the experiment in the sidebar and click **Run AutoML Advisor**.")