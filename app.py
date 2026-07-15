from datetime import datetime
from pathlib import Path
from typing import Any
from html import escape
import io

import pandas as pd
import streamlit as st

from src.agents.orchestrator_agent import run_orchestrator
from src.state import ExperimentState, save_state


st.set_page_config(
    page_title="Agentic AutoML Advisor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -----------------------------------------------------------------------------
# CSS / VISUAL DESIGN
# -----------------------------------------------------------------------------

def inject_css() -> None:
    """
    Apply a professional product-style UI.
    """

    st.markdown(
        """
        <style>
            :root {
                --bg: #f6f8fb;
                --surface: #ffffff;
                --surface-soft: #f8fafc;
                --text: #0f172a;
                --muted: #64748b;
                --border: #e2e8f0;
                --primary: #2563eb;
                --primary-soft: #dbeafe;
                --green: #16a34a;
                --green-soft: #dcfce7;
                --amber: #d97706;
                --amber-soft: #fef3c7;
                --red: #dc2626;
                --red-soft: #fee2e2;
                --shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
            }

            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(circle at top left, rgba(37, 99, 235, 0.12), transparent 34rem),
                    radial-gradient(circle at top right, rgba(124, 58, 237, 0.10), transparent 32rem),
                    var(--bg);
            }

            [data-testid="stHeader"] {
                background: rgba(246, 248, 251, 0.72);
                backdrop-filter: blur(14px);
            }

            .block-container {
                max-width: 100vw !important;
                width: 100% !important;
                padding-top: 1.4rem !important;
                padding-left: clamp(1rem, 3vw, 3rem) !important;
                padding-right: clamp(1rem, 3vw, 3rem) !important;
                padding-bottom: 4rem !important;
            }

            h1, h2, h3 {
                letter-spacing: -0.035em;
                color: var(--text);
            }

            p, li, label, span, div {
                color: var(--text);
            }

            .hero {
                background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 48%, #312e81 100%);
                border-radius: 30px;
                padding: 2rem 2.2rem;
                color: #ffffff;
                box-shadow: var(--shadow);
                position: relative;
                overflow: hidden;
                min-height: 260px;
                margin-bottom: 1.4rem;
            }

            .hero:after {
                content: "";
                position: absolute;
                top: -120px;
                right: -90px;
                width: 300px;
                height: 300px;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.12);
            }

            .hero h1 {
                color: #ffffff;
                font-size: clamp(2.2rem, 4vw, 3.6rem);
                line-height: 1.02;
                margin: 0 0 1rem 0;
                max-width: 850px;
            }

            .hero p {
                color: #dbeafe;
                max-width: 800px;
                font-size: 1.08rem;
                line-height: 1.65;
                margin-bottom: 1.2rem;
            }

            .hero-badges {
                display: flex;
                flex-wrap: wrap;
                gap: 0.55rem;
                margin-top: 1.1rem;
            }

            .hero-badge {
                color: #eff6ff !important;
                background: rgba(255, 255, 255, 0.12);
                border: 1px solid rgba(255, 255, 255, 0.18);
                padding: 0.45rem 0.75rem;
                border-radius: 999px;
                font-size: 0.88rem;
                font-weight: 650;
            }

            .section-title {
                margin-top: 1.4rem;
                margin-bottom: 0.7rem;
            }

            .section-title h2 {
                margin-bottom: 0.25rem;
            }

            .section-title p {
                margin-top: 0;
                color: var(--muted);
                font-size: 1rem;
            }

            .card-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 1rem;
                margin: 1rem 0 1.5rem 0;
            }

            .agent-card {
                background: #ffffff;
                border: 1px solid var(--border);
                border-radius: 20px;
                padding: 1.1rem;
                min-height: 150px;
                box-shadow: 0 8px 26px rgba(15, 23, 42, 0.05);
            }

            .agent-icon {
                width: 42px;
                height: 42px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 14px;
                background: var(--primary-soft);
                margin-bottom: 0.75rem;
                font-size: 1.35rem;
            }

            .agent-card h3 {
                margin: 0 0 0.25rem 0;
                font-size: 1.05rem;
            }

            .agent-card p {
                margin: 0;
                color: var(--muted);
                line-height: 1.55;
                font-size: 0.94rem;
            }

            .upload-shell {
                background: #ffffff;
                border: 1px solid var(--border);
                border-radius: 26px;
                padding: 1.5rem;
                box-shadow: var(--shadow);
                margin-top: 1rem;
                margin-bottom: 1.2rem;
            }

            .upload-title {
                text-align: center;
                margin-bottom: 1rem;
            }

            .upload-title h2 {
                margin-bottom: 0.3rem;
            }

            .upload-title p {
                color: var(--muted);
                margin-top: 0;
            }

            .metric-card {
                background: #ffffff;
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 1rem;
                min-height: 118px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            }

            .metric-label {
                color: var(--muted);
                font-size: 0.82rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 0.45rem;
            }

            .metric-value {
                font-size: 1.45rem;
                line-height: 1.15;
                font-weight: 850;
                color: var(--text);
                word-break: break-word;
            }

            .metric-subtext {
                color: var(--muted);
                font-size: 0.85rem;
                margin-top: 0.4rem;
                line-height: 1.35;
            }

            .badge-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                margin: 0.4rem 0 1rem 0;
            }

            .tag {
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                padding: 0.35rem 0.6rem;
                background: #eef2ff;
                color: #3730a3 !important;
                border: 1px solid #c7d2fe;
                font-size: 0.82rem;
                font-weight: 650;
            }

            .pill {
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                padding: 0.35rem 0.65rem;
                font-size: 0.8rem;
                font-weight: 750;
                border: 1px solid var(--border);
                background: var(--surface-soft);
            }

            .pill.green {
                background: var(--green-soft);
                color: #166534 !important;
                border-color: #bbf7d0;
            }

            .pill.amber {
                background: var(--amber-soft);
                color: #92400e !important;
                border-color: #fde68a;
            }

            .pill.red {
                background: var(--red-soft);
                color: #991b1b !important;
                border-color: #fecaca;
            }

            .pill.blue {
                background: var(--primary-soft);
                color: #1e40af !important;
                border-color: #bfdbfe;
            }

            .finding-card {
                border-radius: 18px;
                padding: 1rem;
                border: 1px solid var(--border);
                background: #ffffff;
                margin-bottom: 0.8rem;
            }

            .finding-card.high,
            .finding-card.critical,
            .finding-card.error {
                border-color: #fecaca;
                background: #fff7f7;
            }

            .finding-card.medium,
            .finding-card.warning {
                border-color: #fde68a;
                background: #fffbeb;
            }

            .finding-card.low,
            .finding-card.info {
                border-color: #bfdbfe;
                background: #eff6ff;
            }

            .finding-card h4 {
                margin: 0 0 0.5rem 0;
            }

            .finding-card p {
                color: var(--muted);
                margin: 0.25rem 0;
                line-height: 1.55;
            }

            .callout {
                border-radius: 18px;
                padding: 1rem;
                background: #eff6ff;
                border: 1px solid #bfdbfe;
                margin: 1rem 0;
            }

            .callout p {
                margin: 0;
                color: #1e3a8a;
                line-height: 1.55;
            }

            .agent-workflow-shell {
                background: #ffffff;
                border: 1px solid var(--border);
                border-radius: 24px;
                padding: 1.2rem;
                box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
                margin-top: 1rem;
                margin-bottom: 1rem;
            }

            .agent-workflow-title {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
            }

            .agent-workflow-title h3 {
                margin: 0;
                font-size: 1.05rem;
            }

            .agent-workflow-title p {
                margin: 0.2rem 0 0 0;
                color: var(--muted);
                font-size: 0.9rem;
            }

            .agent-status-pill {
                border-radius: 999px;
                background: var(--primary-soft);
                color: #1e40af !important;
                border: 1px solid #bfdbfe;
                padding: 0.4rem 0.75rem;
                font-size: 0.82rem;
                font-weight: 750;
                white-space: nowrap;
            }

            div[data-testid="stFileUploader"] section {
                border: 2px dashed #93c5fd !important;
                background: #eff6ff !important;
                border-radius: 22px !important;
                padding: 1.1rem !important;
            }

            .stButton > button {
                border-radius: 999px;
                padding: 0.72rem 1.25rem;
                font-weight: 800;
                border: none;
                box-shadow: 0 12px 24px rgba(37, 99, 235, 0.18);
            }

            .stDownloadButton > button {
                border-radius: 999px;
                padding: 0.7rem 1.15rem;
                font-weight: 800;
            }

            [data-testid="stDataFrame"] {
                border-radius: 16px;
                overflow: hidden;
            }

            @media (max-width: 900px) {
                .hero h1 {
                    font-size: 2.25rem;
                }

                .card-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

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


def _safe_text(value: Any, fallback: str = "N/A") -> str:
    """
    Return escaped display text for HTML snippets.
    """

    if value is None or value == "":
        return fallback

    return escape(str(value))


def _shorten(value: Any, max_chars: int = 42) -> str:
    """
    Shorten long values for cards.
    """

    text = str(value) if value is not None else "N/A"

    if len(text) <= max_chars:
        return text

    return text[: max_chars - 1] + "…"


def make_quality_dataframe(quality_issues: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert quality issues into a display dataframe.
    """

    if not quality_issues:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "Severity": issue.get("severity", "N/A"),
                "Issue": issue.get("issue", "N/A"),
                "Finding": issue.get("finding", "N/A"),
                "Recommendation": issue.get("recommendation", "N/A"),
            }
            for issue in quality_issues
        ]
    )


def make_leaderboard_dataframe(leaderboard: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert leaderboard into a readable display dataframe.
    """

    if not leaderboard:
        return pd.DataFrame()

    rows = []

    for item in leaderboard:
        improvement = item.get("improvement_over_dummy", {}) or {}

        rows.append(
            {
                "Rank": item.get("rank"),
                "Model": item.get("display_name"),
                "Family": item.get("family"),
                "Metric": item.get("primary_metric"),
                "CV Score": _format_metric(item.get("primary_score")),
                "CV Std": _format_metric(item.get("primary_cv_std")),
                "Train Score": _format_metric(item.get("primary_train_score")),
                "Overfit Gap": _format_metric(item.get("overfit_gap")),
                "Beats Dummy": improvement.get("beats_dummy"),
                "Reliability": item.get("reliability_level"),
            }
        )

    return pd.DataFrame(rows)


def make_training_metrics_dataframe(model_results: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert nested model metrics into a long dataframe.
    """

    rows = []

    for result in model_results:
        for metric_name, metric_values in result.get("metrics", {}).items():
            rows.append(
                {
                    "Model": result.get("display_name"),
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
                "Timestamp": event.get("timestamp"),
                "Agent / Tool": event.get("tool_name"),
                "Status": event.get("status"),
                "Message": event.get("message"),
            }
            for event in tool_history
        ]
    )


def render_section_title(title: str, subtitle: str | None = None) -> None:
    """
    Render a consistent section heading.
    """

    subtitle_html = f"<p>{_safe_text(subtitle)}</p>" if subtitle else ""

    st.markdown(
        f"""
        <div class="section-title">
            <h2>{_safe_text(title)}</h2>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: Any, subtext: str = "") -> None:
    """
    Render a custom metric card.
    """

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{_safe_text(label)}</div>
            <div class="metric-value">{_safe_text(_shorten(value))}</div>
            <div class="metric-subtext">{_safe_text(subtext, "")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pill(text: Any, tone: str = "blue") -> str:
    """
    Return a pill HTML string.
    """

    return f'<span class="pill {tone}">{_safe_text(text)}</span>'


def reliability_tone(value: Any) -> str:
    """
    Map reliability or severity words to visual tones.
    """

    text = str(value or "").lower()

    if any(word in text for word in ["high", "good", "strong", "reliable", "pass", "done"]):
        return "green"

    if any(word in text for word in ["low", "poor", "weak", "fail", "risk", "critical", "error"]):
        return "red"

    return "amber"


def display_tags(
    values: list[Any],
    empty_message: str = "None detected.",
    max_items: int = 28,
) -> None:
    """
    Display a list as readable tags instead of raw Python output.
    """

    if not values:
        st.caption(empty_message)
        return

    shown = values[:max_items]
    hidden_count = max(0, len(values) - len(shown))

    tags = "".join(
        f'<span class="tag">{_safe_text(value)}</span>'
        for value in shown
    )

    if hidden_count:
        tags += f'<span class="tag">+{hidden_count} more</span>'

    st.markdown(
        f'<div class="badge-row">{tags}</div>',
        unsafe_allow_html=True,
    )


def safe_container_with_border():
    """
    Use bordered containers when the installed Streamlit version supports it.
    """

    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


# -----------------------------------------------------------------------------
# TOP UI SECTIONS
# -----------------------------------------------------------------------------

def show_hero() -> None:
    """
    Render the main product hero.
    """

    st.markdown(
        """
        <div class="hero">
            <h1>Agentic AutoML Advisor</h1>
            <p>
                Upload a tabular CSV dataset and let a guided ML advisor inspect the data,
                plan the experiment, train baseline models, compare them against a dummy baseline,
                critique the reliability of the results, and generate a clear final report.
            </p>
            <div class="hero-badges">
                <span class="hero-badge">CSV datasets</span>
                <span class="hero-badge">Classification & regression</span>
                <span class="hero-badge">Baseline comparison</span>
                <span class="hero-badge">Reliability critic</span>
                <span class="hero-badge">Final report</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_agent_cards() -> None:
    """
    Explain the agentic workflow.
    """

    render_section_title(
        "How the advisor works",
        "The system behaves like a small ML team. Each agent has a clear responsibility.",
    )

    st.markdown(
        """
        <div class="card-grid">
            <div class="agent-card">
                <div class="agent-icon">🧭</div>
                <h3>Orchestrator Agent</h3>
                <p>Controls the full workflow and decides which step should run next.</p>
            </div>
            <div class="agent-card">
                <div class="agent-icon">🧪</div>
                <h3>Planner Agent</h3>
                <p>Understands the dataset, detects the ML task, and creates the experiment strategy.</p>
            </div>
            <div class="agent-card">
                <div class="agent-icon">🛡️</div>
                <h3>Critic Agent</h3>
                <p>Checks whether model results are trustworthy, useful, and better than a dummy baseline.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_workflow_stepper(done: bool = False, active_index: int = 1) -> None:
    """
    Show a clean native Streamlit agent workflow panel.
    This avoids the raw HTML issue from the previous version.
    """

    steps = [
        {
            "name": "Inspect",
            "agent": "Data Scanner",
            "icon": "🔎",
            "desc": "Reads the CSV and profiles the dataset.",
        },
        {
            "name": "Plan",
            "agent": "Planner Agent",
            "icon": "🧪",
            "desc": "Detects task type and experiment strategy.",
        },
        {
            "name": "Preprocess",
            "agent": "Data Prep Agent",
            "icon": "🧹",
            "desc": "Handles missing values and feature encoding.",
        },
        {
            "name": "Train",
            "agent": "Model Agent",
            "icon": "🏗️",
            "desc": "Trains baseline ML models.",
        },
        {
            "name": "Compare",
            "agent": "Evaluator Agent",
            "icon": "🏆",
            "desc": "Ranks models against the dummy baseline.",
        },
        {
            "name": "Critique",
            "agent": "Critic Agent",
            "icon": "🛡️",
            "desc": "Checks reliability and usefulness.",
        },
        {
            "name": "Report",
            "agent": "Report Agent",
            "icon": "📄",
            "desc": "Creates the final advisor report.",
        },
    ]

    progress_value = 1.0 if done else max(0.05, min(active_index / len(steps), 1.0))
    run_status = "Completed" if done else f"Step {active_index} of {len(steps)}"

    st.markdown(
        f"""
        <div class="agent-workflow-shell">
            <div class="agent-workflow-title">
                <div>
                    <h3>🤖 Agent workflow</h3>
                    <p>The advisor moves from dataset inspection to model critique and final reporting.</p>
                </div>
                <span class="agent-status-pill">{run_status}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(progress_value)

    cols = st.columns(len(steps), gap="small")

    for index, step in enumerate(steps, start=1):
        with cols[index - 1]:
            if done or index < active_index:
                status_text = "Done"
                status_method = st.success
                status_icon = "✅"
            elif index == active_index:
                status_text = "Active"
                status_method = st.info
                status_icon = "🤖"
            else:
                status_text = "Waiting"
                status_method = st.caption
                status_icon = "○"

            with safe_container_with_border():
                st.markdown(f"### {status_icon}")
                st.markdown(f"**{step['name']}**")
                st.caption(step["agent"])
                st.caption(step["desc"])
                status_method(status_text)


def show_sidebar_guide() -> None:
    """
    Keep sidebar useful but not dominant.
    """

    with st.sidebar:
        st.markdown("## 🤖 AutoML Advisor")

        st.caption(
            "A guided first-pass machine learning advisor for tabular CSV datasets."
        )

        st.markdown("### Best for")
        st.markdown(
            """
            - Loan approval prediction
            - Customer churn
            - Student performance
            - Sales or price prediction
            - Small-to-medium tabular ML experiments
            """
        )

        st.markdown("### You need")
        st.markdown(
            """
            - A CSV file
            - One target column
            - A short objective
            """
        )

        st.markdown("### You get")
        st.markdown(
            """
            - Dataset profile
            - Experiment plan
            - Model leaderboard
            - Reliability critique
            - Downloadable report
            """
        )


# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------

def load_sample_dataset() -> tuple[str | None, pd.DataFrame | None, str | None]:
    """
    Load the built-in sample dataset if available.
    """

    sample_path = Path("data/sample/loan_sample.csv")

    if not sample_path.exists():
        return None, None, "Sample dataset not found at data/sample/loan_sample.csv"

    try:
        df = pd.read_csv(sample_path)
        return str(sample_path), df, None
    except Exception as error:
        return None, None, f"Could not read sample dataset: {error}"


def load_uploaded_dataset(uploaded_file) -> tuple[pd.DataFrame | None, str | None]:
    """
    Read uploaded CSV bytes into a dataframe preview.
    """

    if uploaded_file is None:
        return None, None

    try:
        df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
        return df, None
    except Exception as error:
        return None, f"Could not read uploaded CSV: {error}"


def show_setup_panel() -> tuple[
    pd.DataFrame | None,
    str | None,
    Any,
    str | None,
    str | None,
    bool,
    bool,
]:
    """
    Render the centered experiment setup panel.
    """

    render_section_title(
        "Start a new AutoML run",
        "Use the sample dataset for a quick demo, or upload your own CSV file.",
    )

    st.markdown(
        """
        <div class="upload-shell">
            <div class="upload-title">
                <h2>Upload your dataset</h2>
                <p>Choose a CSV, select the target column, and let the agents run the ML workflow.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_space, main_col, right_space = st.columns([0.15, 0.70, 0.15])

    with main_col:
        dataset_mode = st.radio(
            "Choose dataset source",
            options=["Use sample loan dataset", "Upload my own CSV"],
            horizontal=True,
            label_visibility="collapsed",
        )

        df_preview = None
        dataset_path_for_run = None
        uploaded_file = None

        if dataset_mode == "Use sample loan dataset":
            dataset_path_for_run, df_preview, error = load_sample_dataset()

            if error:
                st.error(error)
            elif df_preview is not None:
                st.success("Sample loan dataset loaded. You can run the advisor immediately.")

        else:
            uploaded_file = st.file_uploader(
                "Drop your CSV here or click to browse",
                type=["csv"],
                help=(
                    "Upload a tabular CSV file. "
                    "The advisor will ask you to select the target column next."
                ),
            )

            df_preview, error = load_uploaded_dataset(uploaded_file)

            if error:
                st.error(error)
            elif df_preview is not None:
                st.success("CSV uploaded successfully. Select your target column below.")

        target_column = None
        user_objective = None
        approve_drop_id_columns = True
        run_button = False

        if df_preview is not None:
            st.markdown("---")

            c1, c2, c3 = st.columns(3)

            with c1:
                render_metric_card(
                    "Rows",
                    f"{df_preview.shape[0]:,}",
                    "Total records detected",
                )

            with c2:
                render_metric_card(
                    "Columns",
                    f"{df_preview.shape[1]:,}",
                    "Including the target column",
                )

            with c3:
                render_metric_card(
                    "Missing cells",
                    f"{int(df_preview.isna().sum().sum()):,}",
                    "Before preprocessing",
                )

            target_column = st.selectbox(
                "What should the advisor predict?",
                options=list(df_preview.columns),
                help=(
                    "Select the target/output column. The remaining columns will be "
                    "treated as features unless removed by preprocessing."
                ),
            )

            user_objective = st.text_area(
                "Describe your ML objective",
                value="Predict whether a loan application will be approved",
                help=(
                    "Write a short business or analytics goal. "
                    "This will be included in the final report."
                ),
            )

            approve_drop_id_columns = st.checkbox(
                "Allow the advisor to drop possible ID columns",
                value=True,
                help=(
                    "Recommended. Columns like Loan_ID or Customer_ID usually do not "
                    "help generalisable model learning."
                ),
            )

            st.markdown(
                """
                <div class="callout">
                    <p>
                        <strong>What happens after you click run?</strong><br>
                        The orchestrator will profile your dataset, create a plan, preprocess features,
                        train baseline models, compare results, run a reliability critique, and generate a report.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            run_button = st.button(
                "Run Agentic AutoML Advisor",
                type="primary",
                width="stretch",
            )

    return (
        df_preview,
        dataset_path_for_run,
        uploaded_file,
        target_column,
        user_objective,
        approve_drop_id_columns,
        run_button,
    )


# -----------------------------------------------------------------------------
# RESULT SECTIONS
# -----------------------------------------------------------------------------

def show_status_cards(state: ExperimentState) -> None:
    """
    Show high-level status cards.
    """

    critic_report = state.critic_report or {}
    training_summary = state.training_summary or {}
    comparison_summary = state.comparison_summary or {}

    reliability = critic_report.get("overall_reliability", "N/A")
    best_model = comparison_summary.get("best_useful_display_name") or "None"

    cols = st.columns(4)

    with cols[0]:
        render_metric_card(
            "Current Status",
            state.status,
            "Workflow execution state",
        )

    with cols[1]:
        render_metric_card(
            "Reliability",
            reliability,
            "Critic agent judgement",
        )

    with cols[2]:
        render_metric_card(
            "Models Completed",
            training_summary.get("models_completed", 0),
            "Finished training runs",
        )

    with cols[3]:
        render_metric_card(
            "Best Useful Model",
            best_model,
            "After dummy baseline check",
        )


def show_dataset_tab(state: ExperimentState) -> None:
    """
    Display dataset profile and quality information.
    """

    profile = state.profile or {}

    render_section_title(
        "Dataset understanding",
        "A readable summary of what the advisor detected before training models.",
    )

    cols = st.columns(4)

    with cols[0]:
        render_metric_card("Rows", profile.get("rows", "N/A"), "Records used")

    with cols[1]:
        render_metric_card("Columns", profile.get("columns", "N/A"), "Raw input columns")

    with cols[2]:
        render_metric_card("Features", profile.get("feature_count", "N/A"), "After selecting target")

    with cols[3]:
        render_metric_card("Task Type", state.task_type or "N/A", "Detected ML problem")

    left, right = st.columns(2)

    with left:
        st.markdown("### Numerical features")
        display_tags(
            profile.get("numerical_columns", []),
            "No numerical features detected.",
        )

    with right:
        st.markdown("### Categorical features")
        display_tags(
            profile.get("categorical_columns", []),
            "No categorical features detected.",
        )

    st.markdown("### Data quality issues")

    quality_issues = state.quality_issues or []
    quality_df = make_quality_dataframe(quality_issues)

    if quality_df.empty:
        st.success("No major quality issues were recorded by the advisor.")
    else:
        for issue in quality_issues[:5]:
            severity = str(issue.get("severity", "info")).lower()
            tone = reliability_tone(severity)

            st.markdown(
                f"""
                <div class="finding-card {escape(severity)}">
                    <h4>
                        {render_pill(issue.get("severity", "Info"), tone)}
                        {_safe_text(issue.get("issue", "Data quality issue"))}
                    </h4>
                    <p><strong>Finding:</strong> {_safe_text(issue.get("finding", "N/A"))}</p>
                    <p><strong>Recommendation:</strong> {_safe_text(issue.get("recommendation", "N/A"))}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("View all quality issues as a table"):
            st.dataframe(
                quality_df,
                width="stretch",
                hide_index=True,
            )


def show_preprocessing_tab(state: ExperimentState) -> None:
    """
    Display preprocessing configuration.
    """

    render_section_title(
        "Preprocessing strategy",
        "How the advisor prepared raw tabular data before model training.",
    )

    config = state.preprocessing_config or {}

    if not config:
        st.info("No preprocessing configuration available.")
        return

    left, right = st.columns(2)

    with left:
        st.markdown("### Columns dropped")
        display_tags(
            config.get("columns_to_drop", []),
            "No columns were dropped.",
        )

        st.markdown("### Numerical features")
        display_tags(
            config.get("numerical_features", []),
            "No numerical features recorded.",
        )

    with right:
        st.markdown("### Categorical features")
        display_tags(
            config.get("categorical_features", []),
            "No categorical features recorded.",
        )

        st.markdown("### Pending approvals")
        display_tags(
            config.get("pending_approvals", []),
            "No approvals pending.",
        )

    with st.expander("Missing value strategy", expanded=True):
        st.json(config.get("missing_value_strategy", {}))

    with st.expander("Validation and metric strategy", expanded=False):
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Validation")
            st.json(config.get("validation_strategy", {}))

        with c2:
            st.markdown("#### Metrics")
            st.json(config.get("metric_strategy", {}))


def show_results_tab(state: ExperimentState) -> None:
    """
    Display model results and leaderboard.
    """

    render_section_title(
        "Model results",
        "The advisor compares candidate models and checks whether they beat a simple dummy baseline.",
    )

    leaderboard_df = make_leaderboard_dataframe(state.leaderboard)

    if leaderboard_df.empty:
        st.info("No leaderboard available.")
    else:
        best_row = leaderboard_df.iloc[0]

        c1, c2, c3 = st.columns(3)

        with c1:
            render_metric_card(
                "Top ranked model",
                best_row.get("Model"),
                "Highest ranked candidate",
            )

        with c2:
            render_metric_card(
                "Primary score",
                best_row.get("CV Score"),
                best_row.get("Metric", "Primary metric"),
            )

        with c3:
            render_metric_card(
                "Reliability",
                best_row.get("Reliability"),
                "Model-level judgement",
            )

        st.markdown("### Leaderboard")
        st.dataframe(
            leaderboard_df,
            width="stretch",
            hide_index=True,
        )

    metrics_df = make_training_metrics_dataframe(state.model_results)

    st.markdown("### Training metrics")

    if metrics_df.empty:
        st.info("No training metrics available.")
    else:
        st.dataframe(
            metrics_df,
            width="stretch",
            hide_index=True,
        )

    st.markdown("### Model diagnostics")

    if not state.model_results:
        st.info("No model diagnostics available.")
        return

    for result in state.model_results:
        diagnostics = result.get("classification_diagnostics", {}) or {}
        model_name = result.get("display_name", "Model")

        with st.expander(f"Diagnostics: {model_name}", expanded=False):
            c1, c2 = st.columns(2)

            with c1:
                st.markdown(f"**Model ID:** `{result.get('model_id', 'N/A')}`")

            with c2:
                st.markdown(f"**Primary score:** `{_format_metric(result.get('primary_score'))}`")

            if diagnostics:
                c1, c2, c3 = st.columns(3)

                with c1:
                    render_metric_card(
                        "Positive label",
                        diagnostics.get("positive_label"),
                        "Classification target",
                    )

                with c2:
                    render_metric_card(
                        "OOF ROC-AUC",
                        _format_metric(diagnostics.get("out_of_fold_roc_auc")),
                        "Out-of-fold score",
                    )

                with c3:
                    render_metric_card(
                        "OOF PR-AUC",
                        _format_metric(diagnostics.get("out_of_fold_pr_auc")),
                        "Useful for imbalance",
                    )

                matrix = diagnostics.get("confusion_matrix", [])
                labels = diagnostics.get("labels", [])

                if matrix:
                    st.markdown("#### Confusion matrix")

                    matrix_df = pd.DataFrame(
                        matrix,
                        index=[f"Actual {label}" for label in labels],
                        columns=[f"Predicted {label}" for label in labels],
                    )

                    st.dataframe(matrix_df, width="stretch")

                class_metrics = diagnostics.get("class_level_metrics", {})

                if class_metrics:
                    st.markdown("#### Class-level metrics")

                    class_df = pd.DataFrame.from_dict(
                        class_metrics,
                        orient="index",
                    )

                    st.dataframe(class_df, width="stretch")

            else:
                st.info("No classification diagnostics available for this model.")


def show_critic_tab(state: ExperimentState) -> None:
    """
    Display critic report.
    """

    render_section_title(
        "Reliability critic",
        "The critic agent reviews whether the result is trustworthy enough to continue.",
    )

    critic_report = state.critic_report or {}

    if not critic_report:
        st.info("No critic report available.")
        return

    reliability = critic_report.get("overall_reliability", "N/A")
    decision = critic_report.get("recommendation_decision", "N/A")
    can_tune = critic_report.get("can_proceed_to_tuning", "N/A")

    c1, c2, c3 = st.columns(3)

    with c1:
        render_metric_card(
            "Overall Reliability",
            reliability,
            "Final critic judgement",
        )

    with c2:
        render_metric_card(
            "Decision",
            decision,
            "Recommended next step",
        )

    with c3:
        render_metric_card(
            "Can Tune?",
            can_tune,
            "Whether tuning is sensible",
        )

    st.markdown("### Critic findings")

    findings = critic_report.get("findings", [])

    if not findings:
        st.success("No critic findings were reported.")
    else:
        for finding in findings:
            severity = str(finding.get("severity", "info")).lower()
            tone = reliability_tone(severity)

            st.markdown(
                f"""
                <div class="finding-card {escape(severity)}">
                    <h4>
                        {render_pill(finding.get("severity", "Info"), tone)}
                        {_safe_text(finding.get("issue", "Finding"))}
                    </h4>
                    <p><strong>Evidence:</strong> {_safe_text(finding.get("evidence", "N/A"))}</p>
                    <p><strong>Recommendation:</strong> {_safe_text(finding.get("recommendation", "N/A"))}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Next actions")

    next_actions = critic_report.get("next_actions", [])

    if not next_actions:
        st.info("No next actions were provided.")
    else:
        for action in next_actions:
            st.markdown(f"- {action}")


def show_report_tab(state: ExperimentState) -> None:
    """
    Display final report and download button.
    """

    render_section_title(
        "Final advisor report",
        "Download or preview the Markdown report generated from the completed workflow.",
    )

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
        width="stretch",
    )

    with st.expander("Preview final report", expanded=True):
        st.markdown(report_text)


def show_timeline_tab(state: ExperimentState) -> None:
    """
    Display tool execution timeline.
    """

    render_section_title(
        "Agent execution timeline",
        "A transparent trace of the tools and agents that ran during this workflow.",
    )

    timeline_df = make_timeline_dataframe(state.tool_history)

    if timeline_df.empty:
        st.info("No tool history available.")
    else:
        st.dataframe(
            timeline_df,
            width="stretch",
            hide_index=True,
        )


def show_results_dashboard(
    state: ExperimentState,
    orchestrator_result: dict[str, Any],
) -> None:
    """
    Render the completed experiment dashboard.
    """

    st.divider()

    render_section_title(
        "Advisor run completed",
        "Review the experiment summary, model performance, critic judgement, and final report.",
    )

    show_workflow_stepper(done=True)

    show_status_cards(state)

    success = orchestrator_result.get("success", False)
    stop_reason = orchestrator_result.get("stop_reason", "N/A")
    tone = "green" if success else "red"

    st.markdown(
        f"""
        <div class="callout">
            <p>
                {render_pill("Success" if success else "Not successful", tone)}
                &nbsp; <strong>Stop reason:</strong> {_safe_text(stop_reason)}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Actions taken by the orchestrator", expanded=False):
        actions_taken = orchestrator_result.get("actions_taken", [])

        if actions_taken:
            st.dataframe(
                pd.DataFrame(actions_taken),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No orchestrator actions were recorded.")

    tabs = st.tabs(
        [
            "📊 Dataset",
            "🧹 Preprocessing",
            "🏆 Results",
            "🛡️ Critic",
            "📄 Final Report",
            "🧭 Timeline",
        ]
    )

    with tabs[0]:
        show_dataset_tab(state)

    with tabs[1]:
        show_preprocessing_tab(state)

    with tabs[2]:
        show_results_tab(state)

    with tabs[3]:
        show_critic_tab(state)

    with tabs[4]:
        show_report_tab(state)

    with tabs[5]:
        show_timeline_tab(state)


# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------

def main() -> None:
    inject_css()
    show_sidebar_guide()
    show_hero()
    show_agent_cards()
    show_workflow_stepper(done=False, active_index=1)

    (
        df_preview,
        dataset_path_for_run,
        uploaded_file,
        target_column,
        user_objective,
        approve_drop_id_columns,
        run_button,
    ) = show_setup_panel()

    if df_preview is not None:
        with st.expander("Preview first 20 rows", expanded=False):
            st.dataframe(
                df_preview.head(20),
                width="stretch",
                hide_index=True,
            )

    if run_button:
        if df_preview is None or target_column is None:
            st.error("Please load a dataset and select a target column before running the advisor.")
            return

        with st.spinner(
            "The agents are inspecting, planning, training, critiquing, and reporting..."
        ):
            try:
                if dataset_path_for_run:
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
        show_results_dashboard(
            st.session_state["last_state"],
            st.session_state["last_orchestrator_result"],
        )
    else:
        st.markdown(
            """
            <div class="callout">
                <p>
                    <strong>Ready when you are.</strong> Load a dataset above, choose the target column,
                    and click <strong>Run Agentic AutoML Advisor</strong> to start the workflow.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()