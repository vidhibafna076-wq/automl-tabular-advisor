from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any
import io

import pandas as pd
import streamlit as st

from src.agents.orchestrator_agent import run_orchestrator
from src.state import ExperimentState, save_state


st.set_page_config(
    page_title="Agentic AutoML Advisor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


WORKFLOW_STEPS: list[tuple[str, str]] = [
    ("Inspect", "profiled_dataset"),
    ("Plan", "created_experiment_plan"),
    ("Prepare", "created_preprocessing_config"),
    ("Pipeline", "created_preprocessing_pipeline"),
    ("Registry", "created_model_registry"),
    ("Train", "trained_baseline_models"),
    ("Compare", "created_model_leaderboard"),
    ("Critique", "created_critic_report"),
    ("Persist", "checked_model_persistence"),
    ("Report", "created_final_report"),
]

ACTION_LABELS = {
    "inspect_dataset": "Inspecting dataset",
    "create_experiment_plan": "Creating experiment plan",
    "create_preprocessing_config": "Planning preprocessing",
    "create_preprocessing_pipeline": "Building preprocessing pipeline",
    "create_model_registry": "Preparing model registry",
    "train_baseline_models": "Training baseline models",
    "compare_models": "Comparing model evidence",
    "run_reliability_critic": "Reviewing reliability",
    "check_model_persistence": "Checking model persistence",
    "create_final_report": "Creating final report",
}


def inject_css() -> None:
    """Apply a restrained workbench-style visual system."""

    st.markdown(
        """
        <style>
            :root {
                --app-bg: #f5f7fa;
                --surface: #ffffff;
                --surface-muted: #f8fafc;
                --text: #172033;
                --muted: #667085;
                --border: #d9e0e8;
                --primary: #175cd3;
                --primary-soft: #eff6ff;
                --success: #067647;
                --success-soft: #ecfdf3;
                --warning: #b54708;
                --warning-soft: #fffaeb;
                --danger: #b42318;
                --danger-soft: #fef3f2;
            }

            [data-testid="stAppViewContainer"] {
                background: var(--app-bg);
            }

            [data-testid="stHeader"] {
                background: rgba(245, 247, 250, 0.94);
            }

            .block-container {
                max-width: 1440px;
                padding-top: 1.15rem;
                padding-bottom: 3rem;
            }

            h1, h2, h3, h4 {
                color: var(--text);
                letter-spacing: 0;
            }

            p, li, label, span, div {
                color: var(--text);
            }

            .app-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                padding: 0.35rem 0 1rem 0;
                margin-bottom: 1rem;
                border-bottom: 1px solid var(--border);
            }

            .app-header h1 {
                margin: 0;
                font-size: 1.65rem;
                line-height: 1.2;
            }

            .app-header p {
                margin: 0.2rem 0 0 0;
                color: var(--muted);
                font-size: 0.95rem;
            }

            .header-meta {
                display: flex;
                gap: 0.45rem;
                flex-wrap: wrap;
                justify-content: flex-end;
            }

            .meta-label, .tag, .severity-label {
                display: inline-flex;
                align-items: center;
                border: 1px solid var(--border);
                border-radius: 6px;
                background: var(--surface-muted);
                padding: 0.28rem 0.5rem;
                color: #344054 !important;
                font-size: 0.78rem;
                font-weight: 650;
            }

            .section-heading {
                margin: 0 0 0.75rem 0;
            }

            .section-heading h2 {
                margin: 0;
                font-size: 1.25rem;
            }

            .section-heading p {
                margin: 0.25rem 0 0 0;
                color: var(--muted);
                font-size: 0.9rem;
            }

            div[data-testid="stVerticalBlockBorderWrapper"] {
                background: var(--surface);
                border-color: var(--border) !important;
                border-radius: 8px !important;
            }

            [data-testid="stMetric"] {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 0.8rem 0.9rem;
                min-height: 96px;
            }

            [data-testid="stMetricLabel"] {
                color: var(--muted);
            }

            [data-testid="stMetricValue"] {
                font-size: 1.45rem;
            }

            .decision-banner {
                border: 1px solid var(--border);
                border-left-width: 5px;
                border-radius: 8px;
                padding: 1rem 1.1rem;
                margin: 0.75rem 0 1rem 0;
                background: var(--surface);
            }

            .decision-banner h2 {
                margin: 0;
                font-size: 1.2rem;
            }

            .decision-banner p {
                margin: 0.35rem 0 0 0;
                color: #475467;
            }

            .decision-banner.success {
                border-color: #6ce9a6;
                border-left-color: var(--success);
                background: var(--success-soft);
            }

            .decision-banner.warning {
                border-color: #fec84b;
                border-left-color: var(--warning);
                background: var(--warning-soft);
            }

            .decision-banner.danger {
                border-color: #fda29b;
                border-left-color: var(--danger);
                background: var(--danger-soft);
            }

            .finding {
                border: 1px solid var(--border);
                border-left: 4px solid #98a2b3;
                border-radius: 6px;
                padding: 0.8rem 0.9rem;
                margin-bottom: 0.6rem;
                background: var(--surface);
            }

            .finding.high, .finding.critical, .finding.error {
                border-left-color: var(--danger);
            }

            .finding.medium, .finding.warning {
                border-left-color: var(--warning);
            }

            .finding.low, .finding.info {
                border-left-color: var(--primary);
            }

            .finding h4 {
                margin: 0 0 0.35rem 0;
                font-size: 0.98rem;
            }

            .finding p {
                margin: 0.2rem 0;
                color: #475467;
                font-size: 0.9rem;
                line-height: 1.45;
            }

            .tag-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.35rem;
                margin: 0.35rem 0 0.8rem 0;
            }

            .empty-preview {
                min-height: 310px;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
                color: var(--muted);
                border: 1px dashed #b8c2cf;
                border-radius: 8px;
                background: var(--surface-muted);
                padding: 2rem;
            }

            .stButton > button,
            .stDownloadButton > button {
                border-radius: 6px;
                font-weight: 700;
                min-height: 2.55rem;
            }

            div[data-testid="stFileUploader"] section {
                border-color: #98a2b3 !important;
                border-radius: 8px !important;
                background: var(--surface-muted) !important;
            }

            [data-testid="stDataFrame"] {
                border: 1px solid var(--border);
                border-radius: 6px;
                overflow: hidden;
            }

            button[data-baseweb="tab"] {
                font-weight: 650;
            }

            @media (max-width: 760px) {
                .app-header {
                    align-items: flex-start;
                    flex-direction: column;
                }

                .header-meta {
                    justify-content: flex-start;
                }

                [data-testid="stMetric"] {
                    min-height: 82px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Command-centre overrides. These intentionally sit after the original
    # workbench rules so the application's presentation can evolve without
    # changing any modelling or orchestration behaviour.
    st.markdown(
        """
        <style>
            :root {
                --app-bg: #060a12;
                --surface: #0a111d;
                --surface-muted: #0d1725;
                --surface-raised: #111d2c;
                --text: #f4f7fb;
                --muted: #8494a8;
                --border: #1c2b3d;
                --border-bright: #2b3c51;
                --primary: #8b7cff;
                --primary-soft: rgba(139, 124, 255, 0.12);
                --cyan: #3ddbec;
                --cyan-soft: rgba(61, 219, 236, 0.08);
                --success: #45dfa0;
                --success-soft: rgba(69, 223, 160, 0.08);
                --warning: #ffba69;
                --warning-soft: rgba(255, 186, 105, 0.08);
                --danger: #ff7184;
                --danger-soft: rgba(255, 113, 132, 0.08);
            }

            html, body, [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(circle at 65% -20%, rgba(61, 219, 236, 0.07), transparent 34%),
                    var(--app-bg) !important;
                color: var(--text) !important;
            }

            [data-testid="stHeader"] {
                height: 0;
                background: transparent !important;
            }

            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            #MainMenu,
            footer {
                display: none !important;
            }

            .block-container {
                max-width: 1480px;
                padding: 0.85rem 2.25rem 3rem;
            }

            h1, h2, h3, h4, p, li, label, span, div {
                color: var(--text);
            }

            .top-shell {
                display: flex;
                align-items: center;
                min-height: 62px;
                margin: -0.85rem -2.25rem 1.8rem;
                padding: 0 2.25rem;
                border-bottom: 1px solid var(--border);
                background: rgba(6, 10, 18, 0.74);
                backdrop-filter: blur(18px);
            }

            .brand-symbol {
                display: grid;
                place-items: center;
                width: 38px;
                height: 38px;
                margin-right: 0.8rem;
                border: 1px solid var(--cyan);
                border-radius: 10px;
                color: var(--cyan) !important;
                font-size: 1.08rem;
                font-weight: 800;
                box-shadow: 0 0 24px rgba(61, 219, 236, 0.08);
            }

            .brand-name {
                font-size: 1.02rem;
                font-weight: 750;
                letter-spacing: -0.02em;
            }

            .brand-name span {
                margin-left: 0.32rem;
                color: var(--muted) !important;
                font-weight: 520;
            }

            .system-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                margin-left: auto;
                color: var(--muted) !important;
                font-size: 0.72rem;
                font-weight: 600;
            }

            .system-pill i,
            .reliability-pill i {
                display: inline-block;
                width: 7px;
                height: 7px;
                border-radius: 50%;
                background: var(--success);
                box-shadow: 0 0 12px var(--success);
            }

            .hero-row {
                display: flex;
                align-items: flex-end;
                justify-content: space-between;
                gap: 1rem;
                margin: 0 0 1.25rem;
            }

            .eyebrow {
                display: block;
                margin-bottom: 0.45rem;
                color: var(--cyan) !important;
                font-size: 0.64rem;
                font-weight: 750;
                letter-spacing: 0.16em;
                text-transform: uppercase;
            }

            .hero-row h1 {
                margin: 0;
                font-size: clamp(2rem, 3.4vw, 3.25rem);
                line-height: 0.98;
                letter-spacing: -0.055em;
                font-weight: 660;
            }

            .agent-ready {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                min-width: 160px;
                padding: 0.65rem 0.8rem;
                border: 1px solid var(--border);
                border-radius: 12px;
                background: rgba(13, 23, 37, 0.72);
            }

            .agent-orbit {
                position: relative;
                width: 33px;
                height: 33px;
                border: 1px solid var(--border-bright);
                border-radius: 50%;
            }

            .agent-orbit::before,
            .agent-orbit::after {
                content: "";
                position: absolute;
                width: 5px;
                height: 5px;
                border-radius: 50%;
                background: var(--cyan);
                box-shadow: 0 0 8px var(--cyan);
            }

            .agent-orbit::before { top: 2px; left: 13px; }
            .agent-orbit::after { right: 3px; bottom: 5px; background: var(--primary); }
            .agent-ready strong { display: block; font-size: 0.73rem; }
            .agent-ready small { display: block; margin-top: 0.12rem; color: var(--muted); font-size: 0.62rem; }

            .experiment-label {
                margin-bottom: 0.45rem;
                color: var(--muted) !important;
                font-size: 0.66rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }

            div[data-testid="stVerticalBlockBorderWrapper"] {
                border: 1px solid var(--border) !important;
                border-radius: 15px !important;
                background: linear-gradient(145deg, rgba(13, 23, 37, 0.96), rgba(8, 15, 26, 0.96)) !important;
                box-shadow: 0 22px 70px rgba(0, 0, 0, 0.16);
            }

            div[data-testid="stVerticalBlockBorderWrapper"] > div {
                background: transparent !important;
            }

            [data-testid="stRadio"] > label,
            [data-testid="stFileUploader"] > label,
            [data-testid="stTextArea"] > label,
            [data-testid="stSelectbox"] > label,
            [data-testid="stCheckbox"] > label {
                color: var(--muted) !important;
                font-size: 0.7rem !important;
                font-weight: 650 !important;
            }

            [data-baseweb="radio"] {
                padding: 0.34rem 0.58rem;
                border: 1px solid var(--border);
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.018);
            }

            div[data-testid="stFileUploader"] section {
                min-height: 220px;
                display: flex;
                align-items: center;
                justify-content: center;
                border: 1px dashed #2f7483 !important;
                border-radius: 12px !important;
                background:
                    repeating-linear-gradient(0deg, transparent 0 31px, rgba(255,255,255,.018) 32px),
                    repeating-linear-gradient(90deg, transparent 0 31px, rgba(255,255,255,.018) 32px),
                    rgba(61, 219, 236, 0.025) !important;
                transition: 0.18s ease;
            }

            div[data-testid="stFileUploader"] section:hover {
                border-color: var(--cyan) !important;
                background-color: rgba(61, 219, 236, 0.045) !important;
            }

            div[data-testid="stFileUploader"] section button {
                border: 1px solid var(--border-bright) !important;
                background: var(--surface-raised) !important;
                color: var(--text) !important;
            }

            .sample-ready,
            .upload-intro {
                min-height: 220px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: 0.4rem;
                border: 1px solid #24596a;
                border-radius: 12px;
                background: var(--cyan-soft);
                text-align: center;
            }

            .sample-icon {
                display: grid;
                place-items: center;
                width: 52px;
                height: 52px;
                margin-bottom: 0.35rem;
                border: 1px solid var(--cyan);
                border-radius: 13px;
                color: var(--cyan) !important;
                font-weight: 800;
            }

            .sample-ready strong { font-size: 0.92rem; }
            .sample-ready span { color: var(--muted) !important; font-size: 0.68rem; }

            [data-baseweb="textarea"] textarea,
            [data-baseweb="select"] > div,
            [data-baseweb="input"] > div {
                border-color: var(--border-bright) !important;
                background: #09121f !important;
                color: var(--text) !important;
            }

            [data-baseweb="textarea"] textarea:focus,
            [data-baseweb="select"] > div:focus-within {
                border-color: var(--cyan) !important;
                box-shadow: 0 0 0 1px var(--cyan) !important;
            }

            .stButton > button,
            .stDownloadButton > button {
                min-height: 2.75rem;
                border: 1px solid var(--border-bright);
                border-radius: 9px;
                background: rgba(255, 255, 255, 0.025);
                color: var(--text);
                font-size: 0.76rem;
                font-weight: 720;
                transition: 0.16s ease;
            }

            .stButton > button[kind="primary"],
            .stDownloadButton > button[kind="primary"] {
                border: 0 !important;
                background: linear-gradient(105deg, #654ce3, #8a75ff) !important;
                box-shadow: 0 12px 30px rgba(103, 77, 226, 0.2);
            }

            .stButton > button:hover:not(:disabled),
            .stDownloadButton > button:hover:not(:disabled) {
                border-color: var(--cyan);
                color: white;
                transform: translateY(-1px);
            }

            .stButton > button:disabled { opacity: 0.34; }

            .dataset-strip {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.6rem;
                margin: 0.9rem 0 0.5rem;
            }

            .dataset-stat {
                padding: 0.68rem 0.78rem;
                border: 1px solid var(--border);
                border-radius: 9px;
                background: rgba(255, 255, 255, 0.018);
            }

            .dataset-stat span { display: block; color: var(--muted) !important; font-size: 0.58rem; letter-spacing: 0.06em; text-transform: uppercase; }
            .dataset-stat strong { display: block; margin-top: 0.18rem; font-size: 0.94rem; font-variant-numeric: tabular-nums; }

            .workflow-shell {
                margin: 0.9rem 0 0;
                padding: 0.9rem 1.05rem 0.95rem;
                border: 1px solid var(--border);
                border-radius: 13px;
                background: linear-gradient(145deg, rgba(13, 23, 37, 0.9), rgba(8, 15, 26, 0.9));
            }

            .workflow-header {
                display: flex;
                align-items: baseline;
                justify-content: space-between;
                margin-bottom: 0.85rem;
            }

            .workflow-header strong { font-size: 0.72rem; }
            .workflow-header span { color: var(--muted) !important; font-size: 0.6rem; }
            .workflow-track { position: relative; display: grid; grid-template-columns: repeat(10, 1fr); min-width: 700px; }
            .workflow-track::before { content: ""; position: absolute; top: 12px; left: 4%; right: 4%; height: 1px; background: var(--border-bright); }
            .workflow-step { position: relative; z-index: 1; display: flex; flex-direction: column; align-items: center; gap: 0.42rem; color: #65758a !important; }
            .workflow-step b { display: grid; place-items: center; width: 25px; height: 25px; border: 1px solid var(--border-bright); border-radius: 50%; background: #0b1421; color: inherit !important; font-size: 0.58rem; font-weight: 650; }
            .workflow-step small { color: inherit !important; font-size: 0.56rem; white-space: nowrap; }
            .workflow-step.complete { color: var(--cyan) !important; }
            .workflow-step.complete b { border-color: var(--cyan); background: #0b2530; }
            .workflow-step.current { color: #b2a8ff !important; }
            .workflow-step.current b { border-color: var(--primary); background: #201b42; box-shadow: 0 0 0 4px rgba(139, 124, 255, 0.1); }

            .section-heading { margin: 1.4rem 0 0.7rem; }
            .section-heading h2 { margin: 0; font-size: 1.03rem; letter-spacing: -0.02em; }
            .section-heading p { margin: 0.18rem 0 0; color: var(--muted); font-size: 0.72rem; }

            [data-testid="stMetric"] {
                min-height: 84px;
                padding: 0.72rem 0.82rem;
                border: 1px solid var(--border);
                border-radius: 10px;
                background: rgba(13, 23, 37, 0.86);
            }

            [data-testid="stMetricLabel"] p { color: var(--muted) !important; font-size: 0.65rem; }
            [data-testid="stMetricValue"] { color: var(--text); font-size: 1.22rem; }

            .decision-banner {
                border: 1px solid var(--border);
                border-left-width: 4px;
                border-radius: 11px;
                padding: 0.88rem 1rem;
                margin: 0.6rem 0 0.8rem;
                background: var(--surface);
            }

            .decision-banner h2 { margin: 0; font-size: 0.96rem; }
            .decision-banner p { margin: 0.24rem 0 0; color: var(--muted); font-size: 0.7rem; }
            .decision-banner.success { border-color: rgba(69, 223, 160, 0.28); border-left-color: var(--success); background: var(--success-soft); }
            .decision-banner.warning { border-color: rgba(255, 186, 105, 0.28); border-left-color: var(--warning); background: var(--warning-soft); }
            .decision-banner.danger { border-color: rgba(255, 113, 132, 0.28); border-left-color: var(--danger); background: var(--danger-soft); }

            .finding { border-color: var(--border); background: rgba(13, 23, 37, 0.75); }
            .finding h4 { font-size: 0.82rem; }
            .finding p { color: var(--muted); font-size: 0.7rem; }
            .meta-label, .tag, .severity-label { border-color: var(--border); background: var(--surface-muted); color: #bac7d5 !important; font-size: 0.62rem; }

            [data-testid="stDataFrame"] { border-color: var(--border); border-radius: 9px; }
            [data-testid="stExpander"] { border-color: var(--border); border-radius: 9px; background: rgba(13, 23, 37, 0.55); }
            [data-testid="stExpander"] summary p { font-size: 0.7rem; }

            [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 0.2rem; border-bottom: 1px solid var(--border); }
            button[data-baseweb="tab"] { color: var(--muted); font-size: 0.72rem; font-weight: 650; }
            button[data-baseweb="tab"][aria-selected="true"] { color: var(--cyan); }

            [data-testid="stStatusWidget"] { border-color: var(--border); background: var(--surface); }
            [data-testid="stProgress"] > div > div { background: linear-gradient(90deg, var(--cyan), var(--primary)); }

            @media (max-width: 900px) {
                .block-container { padding: 0.75rem 1rem 2rem; }
                .top-shell { margin: -0.75rem -1rem 1.35rem; padding: 0 1rem; }
                .dataset-strip { grid-template-columns: repeat(2, 1fr); }
                .workflow-shell { overflow-x: auto; }
            }

            @media (max-width: 620px) {
                .hero-row { align-items: flex-start; }
                .agent-ready { min-width: auto; padding: 0.5rem; }
                .agent-ready > div:last-child { display: none; }
                .system-pill { font-size: 0; }
                .system-pill::after { content: "Ready"; font-size: 0.68rem; }
                .dataset-strip { grid-template-columns: 1fr 1fr; }
            }

            @media (prefers-reduced-motion: reduce) {
                *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; animation: none !important; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _safe_text(value: Any, fallback: str = "N/A") -> str:
    if value is None or value == "":
        return escape(fallback)
    return escape(str(value))


def _format_metric(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _display_value(value: Any, fallback: str = "N/A") -> str:
    formatted = _format_metric(value)
    return str(formatted) if formatted is not None else str(value or fallback)


def _safe_filename(filename: str) -> str:
    allowed = [char if char.isalnum() or char in ".-_" else "_" for char in filename]
    return "".join(allowed) or "dataset.csv"


def _save_uploaded_file(uploaded_file: Any) -> str:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = raw_dir / f"{timestamp}_{_safe_filename(uploaded_file.name)}"
    output_path.write_bytes(uploaded_file.getvalue())
    return str(output_path)


def safe_container_with_border():
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def render_header() -> None:
    st.markdown(
        """
        <div class="top-shell">
            <div class="brand-symbol">A</div>
            <div class="brand-name">Agentic AutoML <span>Advisor</span></div>
            <div class="system-pill"><i></i> System ready</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_agent_workflow(state: ExperimentState | None = None) -> None:
    """Render the controlled ten-step workflow as a compact visual rail."""

    completed = set(state.completed_steps if state else [])
    completed_count = sum(marker in completed for _, marker in WORKFLOW_STEPS)
    current_index = min(completed_count, len(WORKFLOW_STEPS) - 1)

    steps_html = []
    for index, (label, marker) in enumerate(WORKFLOW_STEPS):
        is_complete = marker in completed
        is_current = state is not None and not is_complete and index == current_index
        state_class = "complete" if is_complete else "current" if is_current else ""
        marker_text = "&#10003;" if is_complete else str(index + 1)
        steps_html.append(
            f'<div class="workflow-step {state_class}">'
            f'<b>{marker_text}</b><small>{_safe_text(label)}</small></div>'
        )

    status_text = f"{completed_count} of {len(WORKFLOW_STEPS)} complete" if state else "Ready"
    st.markdown(
        f"""
        <div class="workflow-shell">
            <div class="workflow-header">
                <strong>Agent workflow</strong>
                <span>{status_text}</span>
            </div>
            <div class="workflow-track">{''.join(steps_html)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dataset_strip(df: pd.DataFrame, target_column: str | None) -> None:
    """Show only the dataset facts needed before a run."""

    missing_cells = int(df.isna().sum().sum())
    numeric_columns = int(df.select_dtypes(include="number").shape[1])
    target_values = df[target_column].nunique(dropna=True) if target_column else "—"
    st.markdown(
        f"""
        <div class="dataset-strip">
            <div class="dataset-stat"><span>Rows</span><strong>{len(df):,}</strong></div>
            <div class="dataset-stat"><span>Columns</span><strong>{len(df.columns):,}</strong></div>
            <div class="dataset-stat"><span>Missing cells</span><strong>{missing_cells:,}</strong></div>
            <div class="dataset-stat"><span>{'Target values' if target_column else 'Numeric columns'}</span><strong>{target_values if target_column else numeric_columns}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f"<p>{_safe_text(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="section-heading">
            <h2>{_safe_text(title)}</h2>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_tags(
    values: list[Any],
    empty_message: str = "None detected.",
    max_items: int = 24,
) -> None:
    if not values:
        st.caption(empty_message)
        return

    shown = values[:max_items]
    tags = "".join(f'<span class="tag">{_safe_text(value)}</span>' for value in shown)
    hidden_count = len(values) - len(shown)

    if hidden_count > 0:
        tags += f'<span class="tag">+{hidden_count} more</span>'

    st.markdown(f'<div class="tag-row">{tags}</div>', unsafe_allow_html=True)


def render_finding(finding: dict[str, Any], evidence_key: str) -> None:
    severity = str(finding.get("severity", "info")).lower()
    st.markdown(
        f"""
        <div class="finding {escape(severity)}">
            <h4><span class="severity-label">{_safe_text(severity.upper())}</span>
                {_safe_text(finding.get("issue", "Finding"))}</h4>
            <p><strong>{_safe_text(evidence_key.replace('_', ' ').title())}:</strong>
                {_safe_text(finding.get(evidence_key, "Not available"))}</p>
            <p><strong>Recommendation:</strong>
                {_safe_text(finding.get("recommendation", "Not available"))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_quality_dataframe(quality_issues: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Severity": issue.get("severity"),
                "Issue": issue.get("issue"),
                "Finding": issue.get("finding"),
                "Recommendation": issue.get("recommendation"),
            }
            for issue in quality_issues
        ]
    )


def make_leaderboard_dataframe(leaderboard: list[dict[str, Any]]) -> pd.DataFrame:
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
                "Dummy Score": _format_metric(item.get("dummy_baseline_score")),
                "Improvement": _format_metric(improvement.get("raw_improvement")),
                "Beats Dummy": improvement.get("beats_dummy"),
                "Reliability": item.get("reliability_level"),
            }
        )

    return pd.DataFrame(rows)


def make_training_metrics_dataframe(model_results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for result in model_results:
        for metric_name, values in result.get("metrics", {}).items():
            rows.append(
                {
                    "Model": result.get("display_name"),
                    "Metric": metric_name,
                    "CV Mean": _format_metric(values.get("cv_mean")),
                    "CV Std": _format_metric(values.get("cv_std")),
                    "Train Mean": _format_metric(values.get("train_mean")),
                }
            )

    return pd.DataFrame(rows)


def make_timeline_dataframe(tool_history: list[dict[str, Any]]) -> pd.DataFrame:
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


def load_sample_dataset() -> tuple[str | None, pd.DataFrame | None, str | None]:
    sample_path = Path("data/sample/loan_sample.csv")

    try:
        return str(sample_path), pd.read_csv(sample_path), None
    except Exception as error:
        return None, None, f"Could not load sample dataset: {error}"


def load_uploaded_dataset(uploaded_file: Any) -> tuple[pd.DataFrame | None, str | None]:
    if uploaded_file is None:
        return None, None

    try:
        return pd.read_csv(io.BytesIO(uploaded_file.getvalue())), None
    except Exception as error:
        return None, f"Could not read uploaded CSV: {error}"


def show_sidebar_help() -> None:
    with st.sidebar:
        st.markdown("## Agentic AutoML")
        st.caption("A controlled modelling workflow for tabular CSV data.")
        st.markdown("**Scope**")
        st.caption("Classification · Regression · Baseline comparison · Reliability review")
        st.info("The critic can withhold a recommendation when the evidence is weak.")


def show_setup_workbench() -> tuple[
    pd.DataFrame | None,
    str | None,
    Any,
    str | None,
    str | None,
    bool,
    bool,
]:
    st.markdown(
        """
        <div class="hero-row">
            <div>
                <span class="eyebrow">New experiment</span>
                <h1>Build a trusted model.</h1>
            </div>
            <div class="agent-ready">
                <div class="agent-orbit"></div>
                <div><strong>10 agents ready</strong><small>Controlled workflow</small></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = None
    dataset_path_for_run = None
    df_preview = None
    error = None
    target_column = None
    user_objective = None
    approve_drop_id_columns = True

    with safe_container_with_border():
        st.markdown('<div class="experiment-label">Dataset source</div>', unsafe_allow_html=True)
        dataset_mode = st.radio(
            "Dataset source",
            options=["Upload CSV", "Use sample dataset"],
            horizontal=True,
            key="dataset_source",
            label_visibility="collapsed",
        )

        source_col, details_col = st.columns([0.58, 0.42], gap="large")

        with source_col:
            if dataset_mode == "Use sample dataset":
                dataset_path_for_run, df_preview, error = load_sample_dataset()
                st.markdown(
                    """
                    <div class="sample-ready">
                        <div class="sample-icon">CSV</div>
                        <strong>loan_sample.csv</strong>
                        <span>Sample dataset ready</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                uploaded_file = st.file_uploader(
                    "Upload CSV dataset",
                    type=["csv"],
                    help="Upload one structured CSV file.",
                    label_visibility="collapsed",
                    key="dataset_upload",
                )
                df_preview, error = load_uploaded_dataset(uploaded_file)

            if error:
                st.error(error)

        with details_col:
            st.markdown('<div class="experiment-label">Experiment details</div>', unsafe_allow_html=True)

            if df_preview is not None:
                target_options = list(df_preview.columns)
                default_index = None
                if dataset_mode == "Use sample dataset" and "Loan_Status" in target_options:
                    default_index = target_options.index("Loan_Status")

                target_column = st.selectbox(
                    "Prediction target",
                    options=target_options,
                    index=default_index,
                    placeholder="Select the column to predict",
                )
            else:
                st.selectbox(
                    "Prediction target",
                    options=[],
                    placeholder="Upload a dataset first",
                    disabled=True,
                )

            objective_default = (
                "Predict whether a loan application will be approved"
                if dataset_mode == "Use sample dataset"
                else ""
            )
            objective_key = "sample_objective" if dataset_mode == "Use sample dataset" else "upload_objective"
            user_objective = st.text_area(
                "Experiment objective",
                value=objective_default,
                placeholder="What should the model predict?",
                key=objective_key,
                disabled=df_preview is None,
                height=86,
            )

            with st.expander("Advanced options", expanded=False):
                approve_drop_id_columns = st.checkbox(
                    "Drop likely identifier columns",
                    value=True,
                    help="Columns such as Customer_ID usually do not generalise to new records.",
                )

            run_button = st.button(
                "Start experiment  →",
                type="primary",
                width="stretch",
                disabled=df_preview is None or target_column is None or not user_objective,
            )

    if df_preview is not None:
        render_dataset_strip(df_preview, target_column)
        with st.expander("Preview dataset", expanded=False):
            st.dataframe(
                df_preview.head(20),
                width="stretch",
                hide_index=True,
                height=280,
            )

    if "last_state" not in st.session_state:
        render_agent_workflow()

    return (
        df_preview,
        dataset_path_for_run,
        uploaded_file,
        target_column,
        user_objective,
        approve_drop_id_columns,
        run_button,
    )


def _decision_content(
    state: ExperimentState,
    orchestrator_result: dict[str, Any],
) -> tuple[str, str, str]:
    if state.status == "preprocessing_config_created_with_pending_approvals":
        return (
            "warning",
            "Action required before training",
            "The workflow paused because one or more preprocessing decisions need approval.",
        )

    if not orchestrator_result.get("success", False):
        return (
            "danger",
            "The run did not complete",
            str(orchestrator_result.get("error") or "Review the run details and try again."),
        )

    critic_report = state.critic_report or {}
    decision = critic_report.get("recommendation_decision")
    reliability = critic_report.get("overall_reliability", "unknown")

    if decision == "do_not_recommend_model":
        return (
            "danger",
            "Model recommendation withheld",
            f"Reliability is {reliability}. The current evidence is not strong enough to select or save a model.",
        )

    if decision == "recommend_with_caution":
        return (
            "warning",
            "Candidate found, further validation required",
            f"Reliability is {reliability}. Treat the selected model as provisional rather than deployable.",
        )

    if decision == "recommend_candidate_model":
        selected = critic_report.get("selected_candidate", {}) or {}
        return (
            "success",
            "Candidate model recommended",
            f"The critic approved {selected.get('display_name', 'the selected candidate')} with {reliability} reliability.",
        )

    return (
        "warning",
        "Run completed without a final model decision",
        "Review the workflow status and reliability details before continuing.",
    )


def render_decision_banner(
    state: ExperimentState,
    orchestrator_result: dict[str, Any],
) -> None:
    tone, title, message = _decision_content(state, orchestrator_result)
    st.markdown(
        f"""
        <div class="decision-banner {tone}">
            <h2>{_safe_text(title)}</h2>
            <p>{_safe_text(message)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _best_leaderboard_item(state: ExperimentState) -> dict[str, Any] | None:
    best_useful_id = (state.comparison_summary or {}).get("best_useful_model_id")

    if best_useful_id:
        return next(
            (item for item in state.leaderboard if item.get("model_id") == best_useful_id),
            None,
        )

    return state.leaderboard[0] if state.leaderboard else None


def show_status_metrics(state: ExperimentState) -> None:
    critic_report = state.critic_report or {}
    training_summary = state.training_summary or {}
    comparison_summary = state.comparison_summary or {}
    best_model = comparison_summary.get("best_useful_display_name") or "None"

    cols = st.columns(4)
    cols[0].metric("Task", str(state.task_type or "N/A").replace("_", " ").title())
    cols[1].metric("Reliability", str(critic_report.get("overall_reliability", "N/A")).title())
    cols[2].metric("Models completed", training_summary.get("models_completed", 0))
    cols[3].metric("Best useful model", best_model)


def show_leaderboard_chart(state: ExperimentState) -> None:
    chart_rows = []

    for item in state.leaderboard:
        score = _format_metric(item.get("primary_score"))
        dummy_score = _format_metric(item.get("dummy_baseline_score"))

        if score is None:
            continue

        chart_rows.append(
            {
                "Model": item.get("display_name"),
                "Model score": score,
                "Dummy baseline": dummy_score,
            }
        )

    if not chart_rows:
        st.info("No model scores are available to chart.")
        return

    chart_df = pd.DataFrame(chart_rows).set_index("Model")
    st.bar_chart(chart_df, horizontal=True, width="stretch")
    st.caption("The grey comparison series repeats the dummy score so each model can be judged against the same baseline.")


def show_overview_tab(state: ExperimentState) -> None:
    render_section_title(
        "Run overview",
        "The decision, performance evidence, and most important next actions.",
    )

    best_item = _best_leaderboard_item(state)
    comparison = state.comparison_summary or {}
    overview_cols = st.columns(4)
    overview_cols[0].metric("Primary metric", comparison.get("primary_metric", "N/A"))
    overview_cols[1].metric(
        "Candidate score",
        _display_value(best_item.get("primary_score") if best_item else None),
    )
    overview_cols[2].metric(
        "Dummy score",
        _display_value(comparison.get("dummy_baseline_score")),
    )
    overview_cols[3].metric(
        "CV variation",
        _display_value(best_item.get("primary_cv_std") if best_item else None),
    )

    chart_col, action_col = st.columns([0.62, 0.38], gap="large")

    with chart_col:
        st.markdown("### Model comparison")
        show_leaderboard_chart(state)

    with action_col:
        st.markdown("### Recommended next actions")
        next_actions = (state.critic_report or {}).get("next_actions", [])

        if not next_actions:
            st.info("No next actions were recorded.")
        else:
            for index, action in enumerate(next_actions, start=1):
                st.markdown(f"**{index}.** {action}")

    high_priority = [
        issue
        for issue in state.quality_issues
        if str(issue.get("severity", "")).lower() in {"high", "critical"}
    ]

    if high_priority:
        st.markdown("### High-priority data concerns")
        for issue in high_priority[:3]:
            render_finding(issue, "finding")


def show_data_tab(state: ExperimentState) -> None:
    profile = state.profile or {}
    config = state.preprocessing_config or {}

    render_section_title(
        "Data and preprocessing",
        "What the advisor detected and how those columns were prepared for modelling.",
    )

    cols = st.columns(4)
    cols[0].metric("Rows", profile.get("rows", "N/A"))
    cols[1].metric("Columns", profile.get("columns", "N/A"))
    cols[2].metric("Features", profile.get("feature_count", "N/A"))
    cols[3].metric("Duplicate rows", profile.get("duplicate_rows", 0))

    feature_col, prep_col = st.columns(2, gap="large")

    with feature_col:
        with safe_container_with_border():
            st.markdown("### Detected columns")
            st.markdown("**Numerical features**")
            display_tags(profile.get("numerical_columns", []), "No numerical features detected.")
            st.markdown("**Categorical features**")
            display_tags(profile.get("categorical_columns", []), "No categorical features detected.")
            st.markdown("**Possible identifiers**")
            display_tags(profile.get("possible_id_columns", []), "No likely identifiers detected.")
            st.markdown("**Possible date columns**")
            display_tags(profile.get("possible_date_columns", []), "No date-like columns detected.")

    with prep_col:
        with safe_container_with_border():
            st.markdown("### Preprocessing decisions")
            st.markdown("**Dropped columns**")
            display_tags(config.get("columns_to_drop", []), "No columns were dropped.")

            missing_strategy = config.get("missing_value_strategy", {})
            numerical_strategy = missing_strategy.get("numerical", {}).get("strategy", "N/A")
            categorical_strategy = missing_strategy.get("categorical", {}).get("strategy", "N/A")
            validation = config.get("validation_strategy", {})
            metrics = config.get("metric_strategy", {})

            st.markdown(f"**Numerical missing values:** {numerical_strategy} imputation")
            st.markdown(f"**Categorical missing values:** {categorical_strategy} imputation")
            st.markdown(f"**Validation:** {validation.get('method', 'N/A')}")
            st.markdown(f"**Primary metric:** {metrics.get('primary_metric', 'N/A')}")

            approvals = config.get("pending_approvals", [])
            if approvals:
                st.warning(f"{len(approvals)} preprocessing approval(s) are still pending.")
            else:
                st.success("No preprocessing approvals are pending.")

    st.markdown("### Data quality findings")
    quality_issues = state.quality_issues or []

    if not quality_issues:
        st.success("No data quality findings were recorded.")
    else:
        for issue in quality_issues[:6]:
            render_finding(issue, "finding")

        if len(quality_issues) > 6:
            with st.expander("View all data quality findings"):
                st.dataframe(
                    make_quality_dataframe(quality_issues),
                    width="stretch",
                    hide_index=True,
                )

    leakage_warnings = state.leakage_warnings or []
    if leakage_warnings:
        st.markdown("### Leakage warnings")
        for warning in leakage_warnings:
            render_finding(warning, "finding")

    with st.expander("Technical preprocessing configuration", expanded=False):
        st.json(config)


def show_models_tab(state: ExperimentState) -> None:
    render_section_title(
        "Model evaluation",
        "Cross-validation results, dummy-baseline comparison, and model-level diagnostics.",
    )

    leaderboard_df = make_leaderboard_dataframe(state.leaderboard)

    if leaderboard_df.empty:
        st.info("No leaderboard is available.")
        return

    show_leaderboard_chart(state)

    st.markdown("### Leaderboard")
    st.dataframe(leaderboard_df, width="stretch", hide_index=True)

    with st.expander("All training metrics", expanded=False):
        metrics_df = make_training_metrics_dataframe(state.model_results)
        if metrics_df.empty:
            st.info("No training metrics are available.")
        else:
            st.dataframe(metrics_df, width="stretch", hide_index=True)

    st.markdown("### Model diagnostics")

    for result in state.model_results:
        model_name = result.get("display_name", "Model")
        primary_score = _display_value(result.get("primary_score"))

        with st.expander(f"{model_name} | primary score: {primary_score}", expanded=False):
            summary_cols = st.columns(4)
            summary_cols[0].metric("Status", str(result.get("status", "N/A")).title())
            summary_cols[1].metric("Family", result.get("family", "N/A"))
            summary_cols[2].metric("Primary metric", result.get("primary_metric", "N/A"))
            summary_cols[3].metric("Primary score", primary_score)

            if result.get("error"):
                st.error(result["error"])

            diagnostics = result.get("classification_diagnostics", {}) or {}
            if not diagnostics:
                st.caption("No classification diagnostics are available for this model.")
                continue

            diagnostic_cols = st.columns(3)
            diagnostic_cols[0].metric("Positive label", diagnostics.get("positive_label", "N/A"))
            diagnostic_cols[1].metric(
                "OOF ROC-AUC",
                _display_value(diagnostics.get("out_of_fold_roc_auc")),
            )
            diagnostic_cols[2].metric(
                "OOF PR-AUC",
                _display_value(diagnostics.get("out_of_fold_pr_auc")),
            )

            matrix = diagnostics.get("confusion_matrix", [])
            labels = diagnostics.get("labels", [])

            if matrix:
                matrix_df = pd.DataFrame(
                    matrix,
                    index=[f"Actual {label}" for label in labels],
                    columns=[f"Predicted {label}" for label in labels],
                )
                st.markdown("#### Confusion matrix")
                st.dataframe(
                    matrix_df.style.background_gradient(cmap="Blues"),
                    width="stretch",
                )

            class_metrics = diagnostics.get("class_level_metrics", {})
            if class_metrics:
                st.markdown("#### Class-level metrics")
                st.dataframe(
                    pd.DataFrame.from_dict(class_metrics, orient="index"),
                    width="stretch",
                )


def show_reliability_tab(
    state: ExperimentState,
    orchestrator_result: dict[str, Any],
) -> None:
    render_section_title(
        "Reliability review",
        "Why the advisor accepted, cautioned against, or rejected the current model evidence.",
    )

    critic_report = state.critic_report or {}

    if not critic_report:
        st.info("The workflow did not produce a critic report.")
    else:
        cols = st.columns(3)
        cols[0].metric("Overall reliability", critic_report.get("overall_reliability", "N/A"))
        cols[1].metric("Decision", critic_report.get("recommendation_decision", "N/A"))
        cols[2].metric("Can proceed to tuning", critic_report.get("can_proceed_to_tuning", "N/A"))

        st.markdown("### Critic findings")
        findings = critic_report.get("findings", [])

        if not findings:
            st.success("No critic findings were recorded.")
        else:
            for finding in findings:
                render_finding(finding, "evidence")

        st.markdown("### Next actions")
        next_actions = critic_report.get("next_actions", [])
        for index, action in enumerate(next_actions, start=1):
            st.markdown(f"**{index}.** {action}")

    with st.expander("Run details and execution timeline", expanded=False):
        stop_reason = orchestrator_result.get("stop_reason", "N/A")
        st.markdown(f"**Final workflow status:** `{state.status}`")
        st.markdown(f"**Stop reason:** `{stop_reason}`")

        actions = orchestrator_result.get("actions_taken", [])
        if actions:
            st.markdown("#### Orchestrator actions")
            st.dataframe(pd.DataFrame(actions), width="stretch", hide_index=True)

        timeline_df = make_timeline_dataframe(state.tool_history)
        if not timeline_df.empty:
            st.markdown("#### Tool timeline")
            st.dataframe(timeline_df, width="stretch", hide_index=True)


def show_report_tab(state: ExperimentState) -> None:
    render_section_title(
        "Final report",
        "Download the experiment record or inspect the generated Markdown report.",
    )

    if not state.final_report_path:
        st.info("No final report was created for this run.")
        return

    path = Path(state.final_report_path)
    if not path.exists():
        st.warning(f"The report was recorded but could not be found at {state.final_report_path}.")
        return

    report_text = path.read_text(encoding="utf-8")
    st.download_button(
        "Download report",
        data=report_text,
        file_name="automl_advisor_report.md",
        mime="text/markdown",
        type="primary",
        key="download_final_automl_report",
    )

    with st.expander("Preview report", expanded=False):
        st.markdown(report_text)


def show_results_dashboard(
    state: ExperimentState,
    orchestrator_result: dict[str, Any],
) -> None:
    render_agent_workflow(state)
    render_section_title(
        "Experiment result",
        "Recommendation and supporting evidence.",
    )
    render_decision_banner(state, orchestrator_result)
    show_status_metrics(state)

    tabs = st.tabs(["Overview", "Data", "Models", "Reliability", "Report"])

    with tabs[0]:
        show_overview_tab(state)

    with tabs[1]:
        show_data_tab(state)

    with tabs[2]:
        show_models_tab(state)

    with tabs[3]:
        show_reliability_tab(state, orchestrator_result)

    with tabs[4]:
        show_report_tab(state)


def _show_run_message(state: ExperimentState, orchestrator_result: dict[str, Any]) -> None:
    if not orchestrator_result.get("success", False):
        st.error(f"Workflow failed: {orchestrator_result.get('error') or state.status}")
    elif state.status == "preprocessing_config_created_with_pending_approvals":
        st.warning("Workflow paused because preprocessing approval is required.")
    elif state.status == "final_report_created":
        st.success("Analysis completed. Review the recommendation and supporting evidence below.")
    else:
        st.info(f"Workflow stopped with status: {state.status}")


def main() -> None:
    inject_css()
    show_sidebar_help()
    render_header()

    if "last_state" in st.session_state:
        _, new_experiment_col = st.columns([0.82, 0.18])
        with new_experiment_col:
            if st.button("← New experiment", width="stretch"):
                for key in [
                    "last_state",
                    "last_orchestrator_result",
                    "dataset_upload",
                    "sample_objective",
                    "upload_objective",
                    "dataset_source",
                ]:
                    st.session_state.pop(key, None)
                st.rerun()

        show_results_dashboard(
            st.session_state["last_state"],
            st.session_state["last_orchestrator_result"],
        )
        return

    (
        df_preview,
        dataset_path_for_run,
        uploaded_file,
        target_column,
        user_objective,
        approve_drop_id_columns,
        run_button,
    ) = show_setup_workbench()

    if run_button:
        if df_preview is None or target_column is None:
            st.error("Load a dataset and select a target before running the advisor.")
            return

        run_status = st.status("Agents are preparing the experiment...", expanded=True)
        progress_bar = st.progress(0, text="Preparing controlled workflow")

        def update_run_progress(event: dict[str, Any]) -> None:
            action = str(event.get("action", ""))
            phase = str(event.get("phase", ""))
            step_number = min(int(event.get("step_number", 1)), len(ACTION_LABELS))
            label = ACTION_LABELS.get(action, action.replace("_", " ").title())

            if phase == "started" and action in ACTION_LABELS:
                progress_bar.progress(
                    max(0, step_number - 1) / len(ACTION_LABELS),
                    text=f"Step {step_number} of {len(ACTION_LABELS)} · {label}",
                )
                run_status.update(label=label, state="running", expanded=True)
            elif phase == "completed" and action in ACTION_LABELS:
                progress_bar.progress(
                    step_number / len(ACTION_LABELS),
                    text=f"Step {step_number} of {len(ACTION_LABELS)} complete",
                )
                run_status.write(f"✓ {label}")

        try:
            final_dataset_path = (
                dataset_path_for_run
                if dataset_path_for_run
                else _save_uploaded_file(uploaded_file)
            )

            state = ExperimentState(
                dataset_path=final_dataset_path,
                target_column=target_column,
                user_objective=user_objective,
            )
            state.completed_steps.append("created_experiment_state")

            orchestrator_result = run_orchestrator(
                state=state,
                approve_drop_id_columns=approve_drop_id_columns,
                progress_callback=update_run_progress,
            )
            state = orchestrator_result["state"]
            save_state(state, "outputs/reports/experiment_state.json")

            if state.status == "final_report_created":
                progress_bar.progress(1.0, text="All ten agent steps complete")
                run_status.update(label="Analysis completed", state="complete", expanded=False)
            elif state.status == "preprocessing_config_created_with_pending_approvals":
                run_status.update(label="Analysis paused for approval", state="error", expanded=True)
            else:
                run_status.update(label="Analysis stopped", state="error", expanded=True)

            st.session_state["last_state"] = state
            st.session_state["last_orchestrator_result"] = orchestrator_result
            st.rerun()

        except Exception as error:
            run_status.update(label="Analysis failed", state="error", expanded=True)
            st.error(f"Workflow failed: {error}")

if __name__ == "__main__":
    main()