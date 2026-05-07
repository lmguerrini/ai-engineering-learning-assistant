"""Streamlit UI page renderers for each app section."""

import streamlit as st


# Re-export symbols from shared and learn_page for backward compatibility
from src.ui.shared import (  # noqa: F401
    LEARN_TOPICS,
    _DEPTH_TO_STYLE,
    _LEARN_PATH_DISPLAY_NAMES,
    _LEARN_PATH_LABELS,
    _LEARN_PATH_TO_ENUM,
    _LEARN_PATH_TOPIC_MAP,
    _LEARNING_DEPTH_LABELS,
    _LEARNING_MODE_LABELS,
    _display_debug_trace,
    _display_feedback_widget,
    _display_memory_section,
    _display_sources_section,
    _show_friendly_error,
)
from src.ui.learn_page import render_learn  # noqa: F401
from src.ui.quiz_page import (  # noqa: F401
    _display_hitl_save,
    _display_quiz_questions,
    _display_quiz_results,
    render_quiz,
)
from src.ui.dashboard_page import (  # noqa: F401
    _accumulate_usage_records,
    _display_session_cost_summary,
    render_advanced,
)
from src.ui.progress_page import render_progress  # noqa: F401

# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

def render_intro() -> None:
    """Render the Home section."""
    st.header("Welcome to the AI Engineering Learning App")
    st.markdown(
        """
        This app helps you study AI Engineering concepts through guided workflows:

        1. **Learn** — Generate a structured Learn Path on a chosen topic.
        2. **Quiz** — Test your understanding with generated questions.
        3. **Progress** — Review your scores, weak areas, and preferences.
        4. **Dashboard** — Inspect retrieval data, graph traces, and settings.

        Select a section from the sidebar to get started.
        """
    )

    try:
        from src.config import get_settings

        settings = get_settings()
        if not settings.openai_api_key:
            _show_friendly_error("no_api_key")
        else:
            st.success("OpenAI API key configured. You're ready to learn.")
    except Exception:
        st.info("This is an early version. Configure your .env file to get started.")

