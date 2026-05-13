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
from src.ui.help_page import render_help_assistant  # noqa: F401
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

def _set_active_section(section: str) -> None:
    """Switch sidebar navigation to the requested section."""
    st.session_state["active_section"] = section
    st.rerun()


def _render_home_feature_card(title: str, body: str) -> None:
    """Render one compact home-page feature card."""
    st.markdown(
        f"""
        <div class="home-feature-card">
          <div class="home-feature-title">{title}</div>
          <div class="home-feature-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_intro() -> None:
    """Render the Home section."""
    st.markdown(
        """
        <style>
        .home-hero {
            padding: 1.35rem 1.5rem 1.2rem 1.5rem;
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 8px;
            background: rgba(21, 101, 192, 0.08);
        }
        .home-title {
            font-size: 2rem;
            font-weight: 700;
            line-height: 1.15;
            margin: 0 0 0.8rem 0;
        }
        .home-copy {
            font-size: 1rem;
            line-height: 1.55;
            margin: 0;
        }
        .home-feature-card {
            height: 100%;
            padding: 1rem 1.05rem;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 8px;
            background: rgba(15, 23, 42, 0.18);
        }
        .home-feature-title {
            margin: 0 0 0.35rem 0;
            font-size: 1rem;
            font-weight: 600;
        }
        .home-feature-body {
            font-size: 0.94rem;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="home-hero">
          <div class="home-title">AI Engineering Learning App</div>
          <p class="home-copy">
            Study AI engineering with grounded lessons, quizzes, progress tracking, and scoped help.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cta_cols = st.columns(3, gap="small")
    with cta_cols[0]:
        if st.button("Start Learning", use_container_width=True, key="home_open_learn"):
            _set_active_section("Learn")
    with cta_cols[1]:
        if st.button("Take a Quiz", use_container_width=True, key="home_open_quiz"):
            _set_active_section("Quiz")
    with cta_cols[2]:
        if st.button("Open Dashboard", use_container_width=True, key="home_open_dashboard"):
            _set_active_section("Dashboard")

    st.markdown("#### Explore")

    top_row = st.columns(3, gap="large")
    with top_row[0]:
        _render_home_feature_card(
            "Learn",
            "Generate grounded topic deep-dives and guided Learn Paths from curated notes and official snapshots.",
        )
    with top_row[1]:
        _render_home_feature_card(
            "Quiz",
            "Create knowledge-grounded quizzes with answer explanations tied to the material you studied.",
        )
    with top_row[2]:
        _render_home_feature_card(
            "Progress",
            "Track saved quiz attempts, weak areas, and learning signals across sessions.",
        )

    bottom_row = st.columns(2, gap="large")
    with bottom_row[0]:
        _render_home_feature_card(
            "Help Assistant",
            "Ask scoped AI engineering and app-workflow questions with grounded provenance and approved live docs when needed.",
        )
    with bottom_row[1]:
        _render_home_feature_card(
            "Dashboard",
            "Inspect traces, token usage, KB health, and evaluation readiness from one operational view.",
        )

    try:
        from src.config import get_settings

        settings = get_settings()
        st.divider()
        if not settings.openai_api_key:
            st.caption("Runtime status: OpenAI API key missing. Configure `.env` to enable generation.")
        else:
            st.caption("Runtime status: OpenAI API key configured.")
    except Exception:
        st.divider()
        st.caption("Runtime status: Configure `.env` to enable generation.")
