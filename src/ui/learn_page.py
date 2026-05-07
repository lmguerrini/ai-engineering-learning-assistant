"""Streamlit Learn page renderer."""

import streamlit as st

from src.schemas import DifficultyLevel, StudyGuide
from src.ui.display_helpers import downgrade_headings
from src.ui.shared import (
    LEARN_TOPICS,
    _DEPTH_TO_STYLE,
    _LEARN_PATH_LABELS,
    _LEARN_PATH_TO_ENUM,
    _LEARN_PATH_TOPIC_MAP,
    _LEARNING_DEPTH_LABELS,
    _LEARNING_MODE_LABELS,
    _accumulate_usage_records,
    _display_debug_trace,
    _display_feedback_widget,
    _display_memory_section,
    _display_sources_section,
)


# ---------------------------------------------------------------------------
# Learn helpers
# ---------------------------------------------------------------------------

def _display_study_guide(guide: StudyGuide, depth: str = "Deep Study",
                         mode: str = "Topic") -> None:
    """Render a structured Learn Path in the Streamlit UI."""
    if mode == "Learn Path":
        level_label = guide.difficulty.value.capitalize()
        st.subheader(f"{level_label} Learn Path")
        st.caption(f"Learn Path Mode · {depth}")
    else:
        st.subheader(guide.topic)
        st.caption(f"Topic Mode · {depth}")

    st.markdown("#### Overview")
    st.markdown(guide.summary)

    # Show Topics section when concepts are available
    if guide.key_concepts:
        st.markdown("#### Topics")
        for i, concept in enumerate(guide.key_concepts, 1):
            if ":" in concept:
                name, desc = concept.split(":", 1)
                st.markdown(f"{i}. **{name.strip()}** — {desc.strip()}")
            else:
                st.markdown(f"{i}. **{concept}**")

    if guide.detailed_notes:
        notes = downgrade_headings(guide.detailed_notes)
        # Skip redundant "Content" heading when notes already start with a section
        if not notes.lstrip().startswith("#"):
            st.markdown("#### Content")
        st.markdown(notes)

    _display_sources_section(guide)


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_learn() -> None:
    """Render the Learn section with Learning Mode selection."""
    st.header("Learn")
    st.markdown(
        "Generate a focused **Topic** study or a guided multi-topic "
        "**Learn Path** powered by Agentic RAG."
    )

    # Learning Mode selector
    learning_mode = st.selectbox(
        "Learning Mode", _LEARNING_MODE_LABELS,
        index=0, key="learn_mode",
    )

    if learning_mode == "Learn Path":
        col1, col2 = st.columns(2)
        with col1:
            learn_path = st.selectbox(
                "Learn Path",
                _LEARN_PATH_LABELS,
                index=1,
                key="learn_path",
            )
        with col2:
            depth_label = st.selectbox(
                "Learning Depth",
                _LEARNING_DEPTH_LABELS,
                index=1,
                key="learn_depth",
            )

        topic = _LEARN_PATH_TOPIC_MAP[learn_path]
        difficulty = _LEARN_PATH_TO_ENUM[learn_path]
        style = _DEPTH_TO_STYLE[depth_label]

    else:  # Topic mode
        col1, col2 = st.columns(2)
        with col1:
            topic = st.selectbox("Topic", LEARN_TOPICS, key="learn_topic")
        with col2:
            depth_label = st.selectbox(
                "Learning Depth",
                _LEARNING_DEPTH_LABELS,
                index=1,
                key="learn_depth_topic",
            )

        difficulty = DifficultyLevel.INTERMEDIATE
        style = _DEPTH_TO_STYLE[depth_label]

    btn_label = "Generate Learn Path" if learning_mode == "Learn Path" else "Generate Topic"
    generate = st.button(btn_label, key="btn_learn")

    if generate:
        with st.spinner("Running Learn workflow..."):
            from src.graphs.learn_graph import run_learn_workflow

            result = run_learn_workflow(
                topic=topic,
                difficulty=difficulty,
                style=style,
            )

        st.session_state["last_learn_result"] = result
        st.session_state["last_learn_depth"] = depth_label
        st.session_state["last_learn_mode"] = learning_mode

        error = result.get("error")
        if error:
            st.error(error)
        else:
            guide = result.get("study_guide")
            if guide:
                _display_study_guide(guide, depth=depth_label, mode=learning_mode)
                st.session_state["last_study_guide"] = guide
            else:
                st.warning("No Learn Path was generated. Try a different topic.")

            st.session_state["last_learn_topic"] = topic

        st.session_state["last_learn_trace"] = result.get("trace", [])
        st.session_state["last_learn_tokens"] = result.get("token_usage", {})
        _accumulate_usage_records(result.get("usage_records", []))

        _display_memory_section(result)
        _display_debug_trace(result, "Learn Workflow Trace")

    _display_feedback_widget("learn", st.session_state.get("last_learn_topic", ""))
