"""Reusable visual components for the AutoML workbench."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import streamlit as st

from .constants import WORKFLOW_STEPS
from .helpers import safe_text


def inject_css() -> None:
    """Apply one accessible visual system.

    Essential interface text is kept at 14 px or above. Selectors are limited
    to a small set of stable Streamlit test IDs and our own CSS classes.
    """

    st.markdown(
        """
        <style>
            :root {
                --app-bg: #070b13;
                --surface: #0d1522;
                --surface-raised: #121e2e;
                --surface-soft: #101a29;
                --text: #f4f7fb;
                --muted: #a4b1c1;
                --border: #26364a;
                --border-strong: #38506a;
                --primary: #9a8cff;
                --cyan: #49d7e8;
                --success: #54dfa4;
                --warning: #ffc074;
                --danger: #ff7c8d;
            }

            html {
                font-size: 16px;
            }

            [data-testid="stAppViewContainer"] {
                color: var(--text);
                background:
                    radial-gradient(circle at 66% -18%, rgba(73, 215, 232, .08), transparent 34%),
                    var(--app-bg);
            }

            [data-testid="stHeader"] {
                background: rgba(7, 11, 19, .88);
            }

            [data-testid="stSidebar"] {
                border-right: 1px solid var(--border);
                background: #09111c;
            }

            .block-container {
                width: min(100%, 1480px);
                max-width: 1480px;
                padding-top: 1.2rem;
                padding-bottom: 3rem;
            }

            h1, h2, h3, h4, p, li, label, span, div {
                color: var(--text);
            }

            p, li {
                line-height: 1.55;
            }

            .top-shell {
                display: flex;
                align-items: center;
                gap: .85rem;
                min-height: 58px;
                margin-bottom: 1.6rem;
                padding-bottom: 1rem;
                border-bottom: 1px solid var(--border);
            }

            .brand-symbol {
                display: grid;
                place-items: center;
                width: 42px;
                height: 42px;
                border: 1px solid var(--cyan);
                border-radius: 11px;
                color: var(--cyan) !important;
                font-size: 1rem;
                font-weight: 800;
            }

            .brand-name {
                font-size: 1.05rem;
                font-weight: 760;
            }

            .brand-name span {
                margin-left: .3rem;
                color: var(--muted) !important;
                font-weight: 520;
            }

            .system-pill {
                display: inline-flex;
                align-items: center;
                gap: .5rem;
                margin-left: auto;
                color: var(--muted) !important;
                font-size: .875rem;
                font-weight: 650;
            }

            .system-pill i {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: var(--success);
                box-shadow: 0 0 12px rgba(84, 223, 164, .8);
            }

            .system-pill.warning i {
                background: var(--warning);
                box-shadow: 0 0 12px rgba(255, 192, 116, .7);
            }

            .hero-row {
                display: flex;
                align-items: flex-end;
                justify-content: space-between;
                gap: 1.25rem;
                margin: 0 0 1.2rem;
            }

            .eyebrow {
                display: block;
                margin-bottom: .45rem;
                color: var(--cyan) !important;
                font-size: .8rem;
                font-weight: 750;
                letter-spacing: .12em;
                text-transform: uppercase;
            }

            .hero-row h1 {
                max-width: 800px;
                margin: 0;
                font-size: clamp(2rem, 3.4vw, 3.3rem);
                line-height: 1.02;
                letter-spacing: -.045em;
                font-weight: 680;
            }

            .workflow-badge {
                min-width: 220px;
                padding: .8rem .95rem;
                border: 1px solid var(--border);
                border-radius: 12px;
                background: rgba(16, 26, 41, .86);
            }

            .workflow-badge strong {
                display: block;
                font-size: .95rem;
            }

            .workflow-badge small {
                display: block;
                margin-top: .2rem;
                color: var(--muted) !important;
                font-size: .875rem;
            }

            .section-heading {
                margin: 1.5rem 0 .8rem;
            }

            .section-heading h2 {
                margin: 0;
                font-size: 1.25rem;
                letter-spacing: -.02em;
            }

            .section-heading p {
                margin: .25rem 0 0;
                color: var(--muted);
                font-size: .95rem;
            }

            div[data-testid="stVerticalBlockBorderWrapper"] {
                border: 1px solid var(--border) !important;
                border-radius: 14px !important;
                background: linear-gradient(145deg, rgba(16, 26, 41, .96), rgba(10, 17, 29, .96));
            }

            [data-testid="stMetric"] {
                min-height: 98px;
                padding: .85rem .95rem;
                border: 1px solid var(--border);
                border-radius: 11px;
                background: rgba(13, 21, 34, .92);
            }

            [data-testid="stMetricLabel"] p {
                color: var(--muted) !important;
                font-size: .875rem;
            }

            [data-testid="stMetricValue"] {
                font-size: 1.45rem;
            }

            .dataset-strip {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: .7rem;
                margin: .9rem 0 .55rem;
            }

            .dataset-stat {
                padding: .85rem .9rem;
                border: 1px solid var(--border);
                border-radius: 10px;
                background: rgba(255, 255, 255, .018);
            }

            .dataset-stat span {
                display: block;
                color: var(--muted) !important;
                font-size: .875rem;
            }

            .dataset-stat strong {
                display: block;
                margin-top: .18rem;
                font-size: 1.05rem;
                font-variant-numeric: tabular-nums;
            }

            .workflow-shell {
                margin: 1rem 0;
                padding: 1rem 1.05rem 1.1rem;
                overflow-x: auto;
                border: 1px solid var(--border);
                border-radius: 13px;
                background: rgba(13, 21, 34, .92);
            }

            .workflow-header {
                display: flex;
                align-items: baseline;
                justify-content: space-between;
                gap: 1rem;
                margin-bottom: 1rem;
            }

            .workflow-header strong,
            .workflow-header span {
                font-size: .875rem;
            }

            .workflow-header span {
                color: var(--muted) !important;
            }

            .workflow-track {
                position: relative;
                display: grid;
                grid-template-columns: repeat(12, minmax(76px, 1fr));
                min-width: 1040px;
            }

            .workflow-track::before {
                content: "";
                position: absolute;
                top: 14px;
                left: 3.5%;
                right: 3.5%;
                height: 1px;
                background: var(--border-strong);
            }

            .workflow-step {
                position: relative;
                z-index: 1;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: .45rem;
                color: #8190a4 !important;
            }

            .workflow-step b {
                display: grid;
                place-items: center;
                width: 29px;
                height: 29px;
                border: 1px solid var(--border-strong);
                border-radius: 50%;
                background: #0b1421;
                color: inherit !important;
                font-size: .875rem;
            }

            .workflow-step small {
                color: inherit !important;
                font-size: .875rem;
                white-space: nowrap;
            }

            .workflow-step.complete {
                color: var(--cyan) !important;
            }

            .workflow-step.complete b {
                border-color: var(--cyan);
                background: #0b2530;
            }

            .workflow-step.current {
                color: #c1b9ff !important;
            }

            .workflow-step.current b {
                border-color: var(--primary);
                background: #211d42;
                box-shadow: 0 0 0 4px rgba(154, 140, 255, .12);
            }

            .decision-banner {
                border: 1px solid var(--border);
                border-left-width: 5px;
                border-radius: 11px;
                padding: 1rem 1.1rem;
                margin: .7rem 0 1rem;
                background: var(--surface);
            }

            .decision-banner h2 {
                margin: 0;
                font-size: 1.15rem;
            }

            .decision-banner p {
                margin: .35rem 0 0;
                color: var(--muted);
                font-size: .95rem;
            }

            .decision-banner.success {
                border-color: rgba(84, 223, 164, .32);
                border-left-color: var(--success);
                background: rgba(84, 223, 164, .075);
            }

            .decision-banner.warning {
                border-color: rgba(255, 192, 116, .34);
                border-left-color: var(--warning);
                background: rgba(255, 192, 116, .075);
            }

            .decision-banner.danger {
                border-color: rgba(255, 124, 141, .34);
                border-left-color: var(--danger);
                background: rgba(255, 124, 141, .075);
            }

            .finding {
                margin-bottom: .7rem;
                padding: .9rem 1rem;
                border: 1px solid var(--border);
                border-left: 4px solid var(--cyan);
                border-radius: 8px;
                background: rgba(13, 21, 34, .8);
            }

            .finding.high,
            .finding.critical,
            .finding.error {
                border-left-color: var(--danger);
            }

            .finding.medium,
            .finding.warning {
                border-left-color: var(--warning);
            }

            .finding h4 {
                margin: 0 0 .4rem;
                font-size: 1rem;
            }

            .finding p {
                margin: .25rem 0;
                color: var(--muted);
                font-size: .9rem;
            }

            .severity-label,
            .tag {
                display: inline-flex;
                align-items: center;
                padding: .28rem .52rem;
                border: 1px solid var(--border);
                border-radius: 7px;
                background: var(--surface-soft);
                color: #c9d3df !important;
                font-size: .875rem;
                font-weight: 650;
            }

            .tag-row {
                display: flex;
                flex-wrap: wrap;
                gap: .4rem;
                margin: .4rem 0 .9rem;
            }

            .evidence-card {
                height: 100%;
                padding: 1rem;
                border: 1px solid var(--border);
                border-radius: 11px;
                background: rgba(13, 21, 34, .75);
            }

            .evidence-card h3 {
                margin: 0 0 .4rem;
                font-size: 1rem;
            }

            .evidence-card p {
                margin: .25rem 0;
                color: var(--muted);
                font-size: .9rem;
            }

            .empty-preview {
                min-height: 280px;
                display: grid;
                place-items: center;
                padding: 2rem;
                border: 1px dashed var(--border-strong);
                border-radius: 11px;
                color: var(--muted) !important;
                text-align: center;
            }

            .stButton > button,
            .stDownloadButton > button {
                min-height: 2.8rem;
                border-radius: 9px;
                font-size: .95rem;
                font-weight: 720;
            }

            div[data-testid="stFileUploader"] section {
                min-height: 180px;
                border-color: #397a88 !important;
                border-radius: 12px !important;
                background: rgba(73, 215, 232, .035) !important;
            }

            [data-testid="stDataFrame"] {
                overflow: hidden;
                border: 1px solid var(--border);
                border-radius: 9px;
            }

            button[data-baseweb="tab"] {
                font-size: .9rem;
                font-weight: 680;
            }

            @media (max-width: 900px) {
                .block-container {
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
                .dataset-strip {
                    grid-template-columns: repeat(2, 1fr);
                }
            }

            @media (max-width: 640px) {
                .hero-row {
                    align-items: flex-start;
                    flex-direction: column;
                }
                .workflow-badge {
                    width: 100%;
                }
                .dataset-strip {
                    grid-template-columns: 1fr;
                }
            }

            @media (prefers-reduced-motion: reduce) {
                *, *::before, *::after {
                    transition: none !important;
                    animation: none !important;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_container_with_border():
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def render_header(workflow_available: bool = True) -> None:
    status_class = "" if workflow_available else "warning"
    status = "Workflow available" if workflow_available else "Setup needs attention"
    st.markdown(
        f"""
        <div class="top-shell">
            <div class="brand-symbol">A</div>
            <div class="brand-name">Agentic AutoML <span>Advisor</span></div>
            <div class="system-pill {status_class}"><i></i>{safe_text(status)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(eyebrow: str, title: str, badge: str, badge_detail: str) -> None:
    st.markdown(
        f"""
        <div class="hero-row">
            <div>
                <span class="eyebrow">{safe_text(eyebrow)}</span>
                <h1>{safe_text(title)}</h1>
            </div>
            <div class="workflow-badge">
                <strong>{safe_text(badge)}</strong>
                <small>{safe_text(badge_detail)}</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f"<p>{safe_text(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="section-heading">
            <h2>{safe_text(title)}</h2>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_banner(tone: str, title: str, message: str) -> None:
    tone = tone if tone in {"success", "warning", "danger"} else "warning"
    st.markdown(
        f"""
        <div class="decision-banner {tone}">
            <h2>{safe_text(title)}</h2>
            <p>{safe_text(message)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_tags(
    values: list[Any],
    empty_message: str = "None detected.",
    max_items: int = 30,
) -> None:
    if not values:
        st.caption(empty_message)
        return

    shown = values[:max_items]
    tags = "".join(f'<span class="tag">{safe_text(value)}</span>' for value in shown)
    hidden = len(values) - len(shown)
    if hidden:
        tags += f'<span class="tag">+{hidden} more</span>'
    st.markdown(f'<div class="tag-row">{tags}</div>', unsafe_allow_html=True)


def render_finding(finding: dict[str, Any], evidence_key: str = "evidence") -> None:
    severity = str(finding.get("severity", "info")).lower()
    recommendation = finding.get("recommendation", "Not available")
    evidence = finding.get(evidence_key, finding.get("finding", "Not available"))
    st.markdown(
        f"""
        <div class="finding {safe_text(severity)}">
            <h4><span class="severity-label">{safe_text(severity.upper())}</span>
                {safe_text(finding.get("issue", "Finding"))}</h4>
            <p><strong>{safe_text(evidence_key.replace("_", " ").title())}:</strong>
                {safe_text(evidence)}</p>
            <p><strong>Recommendation:</strong> {safe_text(recommendation)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dataset_strip(
    dataframe: pd.DataFrame,
    target_column: str | None,
    truncated: bool = False,
) -> None:
    missing = int(dataframe.isna().sum().sum())
    numeric = int(dataframe.select_dtypes(include="number").shape[1])
    target_values = (
        dataframe[target_column].nunique(dropna=True)
        if target_column and target_column in dataframe
        else numeric
    )
    rows_label = "Preview rows" if truncated else "Rows"
    fourth_label = "Target values" if target_column else "Numeric columns"
    st.markdown(
        f"""
        <div class="dataset-strip">
            <div class="dataset-stat"><span>{rows_label}</span><strong>{len(dataframe):,}</strong></div>
            <div class="dataset-stat"><span>Columns</span><strong>{len(dataframe.columns):,}</strong></div>
            <div class="dataset-stat"><span>Missing in preview</span><strong>{missing:,}</strong></div>
            <div class="dataset-stat"><span>{fourth_label}</span><strong>{target_values}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def step_is_complete(completed: set[str], markers: tuple[str, ...]) -> bool:
    return any(marker in completed for marker in markers)


def _artifact_stage_is_complete(
    state: Any,
    *attribute_names: str,
) -> bool:
    """Return whether a stage artifact records a terminal successful outcome.

    Older runs may not contain the newer tuning and holdout markers in
    ``completed_steps`` even though their structured result artifacts were
    written successfully. Treat completed and policy-skipped artifacts as
    finished stages while retaining marker-based behaviour for older states.
    """

    terminal_statuses = {
        "completed",
        "skipped",
    }

    for attribute_name in attribute_names:
        artifact = getattr(state, attribute_name, None)

        if not isinstance(artifact, Mapping):
            continue

        status = str(artifact.get("status", "")).strip().lower()

        if status in terminal_statuses:
            return True

    return False


def workflow_completion_flags(state: Any | None = None) -> list[bool]:
    """Build workflow completion flags from markers and result artifacts."""

    completed = set(getattr(state, "completed_steps", []) if state else [])
    completion_flags = [
        step_is_complete(completed, step.markers) for step in WORKFLOW_STEPS
    ]

    if state is None:
        return completion_flags

    artifact_checks = {
        "tune": _artifact_stage_is_complete(
            state,
            "tuning_result",
            "tuning_summary",
        ),
        "holdout": _artifact_stage_is_complete(
            state,
            "holdout_result",
        ),
    }

    for index, step in enumerate(WORKFLOW_STEPS):
        label = str(step.label).strip().lower()

        if artifact_checks.get(label, False):
            completion_flags[index] = True

    return completion_flags


def render_workflow(state: Any | None = None) -> None:
    completion_flags = workflow_completion_flags(state)
    completed_count = sum(completion_flags)
    current_index = next(
        (
            index
            for index, is_complete in enumerate(completion_flags)
            if not is_complete
        ),
        len(WORKFLOW_STEPS) - 1,
    )

    steps_html: list[str] = []
    for index, (step, is_complete) in enumerate(
        zip(WORKFLOW_STEPS, completion_flags)
    ):
        is_current = state is not None and not is_complete and index == current_index
        state_class = "complete" if is_complete else "current" if is_current else ""
        marker = "&#10003;" if is_complete else str(index + 1)
        steps_html.append(
            f'<div class="workflow-step {state_class}">'
            f"<b>{marker}</b><small>{safe_text(step.label)}</small></div>"
        )

    status = (
        f"{completed_count} of {len(WORKFLOW_STEPS)} stages complete"
        if state
        else "Ready to evaluate"
    )
    st.markdown(
        f"""
        <div class="workflow-shell">
            <div class="workflow-header">
                <strong>Controlled evaluation workflow</strong>
                <span>{safe_text(status)}</span>
            </div>
            <div class="workflow-track">{''.join(steps_html)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )