"""Streamlit UI page renderers for each app section."""

import streamlit as st

from src.schemas import DifficultyLevel, ResponseStyle, StudyGuide


# ---------------------------------------------------------------------------
# Available topics for the Learn section
# ---------------------------------------------------------------------------

LEARN_TOPICS = [
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
]


# ---------------------------------------------------------------------------
# Intro / Help
# ---------------------------------------------------------------------------

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
    st.info("🚧 This is an early version. Quiz and memory features are coming soon.")


# ---------------------------------------------------------------------------
# Learn
# ---------------------------------------------------------------------------

def _display_study_guide(guide: StudyGuide) -> None:
    """Render a structured study guide in the Streamlit UI."""
    st.subheader(f"📘 {guide.topic}")
    st.caption(f"Difficulty: {guide.difficulty.value}")

    st.markdown("### Summary")
    st.markdown(guide.summary)

    if guide.key_concepts:
        st.markdown("### Key Concepts")
        for concept in guide.key_concepts:
            st.markdown(f"- {concept}")

    if guide.detailed_notes:
        st.markdown("### Detailed Notes")
        st.markdown(guide.detailed_notes)

    if guide.sources:
        st.markdown("### Sources Used")
        for src in guide.sources:
            with st.expander(f"📄 {src.title} (relevance: {src.relevance_score:.1f})"):
                st.markdown(src.content_snippet if src.content_snippet else "_No snippet available._")


def render_learn() -> None:
    """Render the Learn section with topic input and study guide generation."""
    st.header("📖 Learn")
    st.markdown(
        "Select a topic and generate a structured study guide powered by "
        "Agentic RAG."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        topic = st.selectbox("Topic", LEARN_TOPICS, key="learn_topic")
    with col2:
        difficulty = st.selectbox(
            "Difficulty",
            [d.value for d in DifficultyLevel],
            index=1,
            key="learn_difficulty",
        )
    with col3:
        style = st.selectbox(
            "Response Style",
            [s.value for s in ResponseStyle],
            index=1,
            key="learn_style",
        )

    generate = st.button("🚀 Generate Study Guide", key="btn_learn")

    if generate:
        with st.spinner("Running Learn workflow…"):
            from src.graphs.learn_graph import run_learn_workflow

            result = run_learn_workflow(
                topic=topic,
                difficulty=DifficultyLevel(difficulty),
                style=ResponseStyle(style),
            )

        error = result.get("error")
        if error:
            st.error(f"⚠️ {error}")
        else:
            guide = result.get("study_guide")
            if guide:
                _display_study_guide(guide)
            else:
                st.warning("No study guide was generated. Try a different topic.")

        # Store trace and token usage for debug view
        st.session_state["last_learn_trace"] = result.get("trace", [])
        st.session_state["last_learn_tokens"] = result.get("token_usage", {})

        # Show debug trace inline when requested
        with st.expander("🔍 Debug Trace"):
            for entry in result.get("trace", []):
                st.text(entry)
            tokens = result.get("token_usage", {})
            if tokens:
                st.json(tokens)


# ---------------------------------------------------------------------------
# Quiz (placeholder)
# ---------------------------------------------------------------------------

def render_quiz() -> None:
    """Render the Quiz section placeholder."""
    st.header("🧠 Quiz")
    st.markdown(
        "Answer AI Engineering questions and get instant feedback. "
        "*(Coming in Phase 4)*"
    )
    st.button("Start Quiz", disabled=True, key="btn_quiz")


# ---------------------------------------------------------------------------
# Progress / Feedback (placeholder)
# ---------------------------------------------------------------------------

def render_progress() -> None:
    """Render the Progress / Feedback section placeholder."""
    st.header("📊 Progress / Feedback")
    st.markdown(
        "Track your studied topics, quiz scores, and weak areas. "
        "*(Coming in Phase 5)*"
    )
    st.info("No progress data yet. Complete a quiz to see your results here.")


# ---------------------------------------------------------------------------
# Advanced / Debug
# ---------------------------------------------------------------------------

def render_advanced() -> None:
    """Render the Advanced / Debug section."""
    st.header("⚙️ Advanced / Debug")
    st.markdown(
        "Inspect retrieval chunks, graph execution traces, token usage, "
        "and application settings."
    )

    with st.expander("Application Settings"):
        from src.config import get_settings

        settings = get_settings()
        st.json({
            "default_model": settings.app_default_model,
            "log_level": settings.app_log_level,
            "langsmith_tracing": settings.langchain_tracing_v2,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        })

    with st.expander("Last Learn Workflow Trace"):
        trace = st.session_state.get("last_learn_trace", [])
        if trace:
            for entry in trace:
                st.text(entry)
        else:
            st.info("No trace available yet. Generate a study guide first.")

    with st.expander("Last Learn Token Usage"):
        tokens = st.session_state.get("last_learn_tokens", {})
        if tokens:
            st.json(tokens)
        else:
            st.info("No token usage data yet.")
