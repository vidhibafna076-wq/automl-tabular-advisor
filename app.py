"""Streamlit entry point for the Agentic AutoML Advisor UI."""

import streamlit as st


st.set_page_config(
    page_title="Agentic AutoML Advisor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


from src.ui.app_shell import main  # noqa: E402


if __name__ == "__main__":
    main()
