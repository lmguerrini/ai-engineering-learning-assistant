"""Streamlit UI page renderers for each app section."""

import streamlit as st


def render_intro() -> None:
    """Render the Intro / Help section."""
    st.header("👋 Welcome to the AI Engineering Learning Assistant")
    st.markdown(
        """
        This app helps you study AI Engineering concepts through a guided workflow:

        1. **Learn** — Get a structured study guide on a chosen topic.
        2. **Quiz** — Test your understanding with generated questions.
        3. **Progress / Feedback** — Review your scores, weak areas, and preferences.
        4. **Advanced / Debug** — Inspect retrieval data, graph traces, and settings.

        Select a section from the sidebar to get started.
        """
    )
    st.info("🚧 This is an early version. LLM and retrieval features are coming soon.")


def render_learn() -> None:
    """Render the Learn section placeholder."""
    st.header("📖 Learn")
    st.markdown(
        "Select a topic and generate a structured study guide powered by "
        "Agentic RAG. *(Coming in Phase 3)*"
    )
    st.selectbox(
        "Topic",
        [
            "AI Agents",
            "ReAct Pattern",
            "Tool Calling",
            "LangGraph",
            "State Management",
            "Agentic RAG",
            "Long-Term Memory",
            "Human-in-the-Loop",
            "Agent Harness",
            "Observability",
        ],
        key="learn_topic",
    )
    st.selectbox("Difficulty", ["beginner", "intermediate", "advanced"], index=1, key="learn_difficulty")
    st.button("Generate Study Guide", disabled=True, key="btn_learn")


def render_quiz() -> None:
    """Render the Quiz section placeholder."""
    st.header("🧠 Quiz")
    st.markdown(
        "Answer AI Engineering questions and get instant feedback. "
        "*(Coming in Phase 4)*"
    )
    st.button("Start Quiz", disabled=True, key="btn_quiz")


def render_progress() -> None:
    """Render the Progress / Feedback section placeholder."""
    st.header("📊 Progress / Feedback")
    st.markdown(
        "Track your studied topics, quiz scores, and weak areas. "
        "*(Coming in Phase 5)*"
    )
    st.info("No progress data yet. Complete a quiz to see your results here.")


def render_advanced() -> None:
    """Render the Advanced / Debug section placeholder."""
    st.header("⚙️ Advanced / Debug")
    st.markdown(
        "Inspect retrieval chunks, graph execution traces, token usage, "
        "and application settings. *(Coming in later phases)*"
    )
    with st.expander("Application Settings"):
        st.json({
            "default_model": "gpt-4o-mini",
            "log_level": "DEBUG",
            "langsmith_tracing": False,
        })
