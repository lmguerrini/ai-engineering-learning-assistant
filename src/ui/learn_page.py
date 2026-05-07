"""Streamlit Learn page renderer."""

import re

import streamlit as st

from src.schemas import DifficultyLevel, StudyGuide
from src.ui.display_helpers import downgrade_headings
from src.ui.shared import (
    LEARN_TOPICS,
    _DEPTH_TO_STYLE,
    _LEARN_PATH_DISPLAY_NAMES,
    _LEARN_PATH_LABELS,
    _LEARN_PATH_STABLE_TOPICS,
    _LEARN_PATH_TO_ENUM,
    _LEARN_PATH_TOPIC_DESCRIPTIONS,
    _LEARN_PATH_TOPIC_EFFORT,
    _LEARN_PATH_TOPIC_MAP,
    _LEARNING_DEPTH_LABELS,
    _LEARNING_MODE_LABELS,
    _accumulate_usage_records,
    _display_debug_trace,
    _display_feedback_widget,
    _display_memory_section,
    _display_sources_section,
)


def _show_cached_learn_result() -> None:
    """Display a previously generated learn result stored in session state."""
    result = st.session_state.get("last_learn_result")
    if not result:
        return
    error = result.get("error")
    if error:
        st.error(error)
        return
    guide = result.get("study_guide")
    if guide:
        depth = st.session_state.get("last_learn_depth", "Summary")
        mode = st.session_state.get("last_learn_mode", "Topic")
        _display_study_guide(guide, depth=depth, mode=mode)
    _display_memory_section(result)
    _display_debug_trace(result, "Learn Workflow Trace")


# ---------------------------------------------------------------------------
# Learn helpers
# ---------------------------------------------------------------------------

def _display_study_guide(guide: StudyGuide, depth: str = "Deep Study",
                         mode: str = "Topic") -> None:
    """Render a structured Learn Path in the Streamlit UI."""
    if mode == "Learn Path":
        level_label = guide.difficulty.value.capitalize()
        display_name = _LEARN_PATH_DISPLAY_NAMES.get(level_label, level_label)
        st.subheader(display_name)
        st.caption(f"Learn Path Mode · {depth}")
    else:
        st.subheader(guide.topic)
        st.caption(f"Topic Mode · {depth}")

    st.markdown("#### Overview")
    st.markdown(guide.summary)

    # --- Topics section ---
    if mode == "Learn Path" and depth == "Summary":
        # Deterministic Topics table for Summary
        level_label = guide.difficulty.value.capitalize()
        topics = _LEARN_PATH_STABLE_TOPICS.get(level_label, [])
        descs = _LEARN_PATH_TOPIC_DESCRIPTIONS.get(level_label, {})
        if topics:
            st.markdown("#### Topics")
            rows = ["| # | Topic | Key Focus | Estimated Effort |"]
            rows.append("|---|-------|-----------|------------------|")
            efforts = _LEARN_PATH_TOPIC_EFFORT.get(level_label, {})
            for i, t in enumerate(topics, 1):
                effort = efforts.get(t, "~2–4 hours")
                rows.append(f"| {i} | {t} | {descs.get(t, '')} | {effort} |")
            st.markdown("\n".join(rows))
    elif guide.key_concepts:
        st.markdown("#### Topics")
        if mode == "Learn Path" and depth == "Deep Study":
            level_label = guide.difficulty.value.capitalize()
            descs = _LEARN_PATH_TOPIC_DESCRIPTIONS.get(level_label, {})
            for i, concept in enumerate(guide.key_concepts, 1):
                name = concept.split(":", 1)[0].strip() if ":" in concept else concept
                desc_text = descs.get(name, "")
                if desc_text:
                    st.markdown(f"{i}. **{name}** — {desc_text}")
                else:
                    st.markdown(f"{i}. **{name}**")
        else:
            for i, concept in enumerate(guide.key_concepts, 1):
                if ":" in concept:
                    name, desc = concept.split(":", 1)
                    st.markdown(f"{i}. **{name.strip()}** — {desc.strip()}")
                else:
                    st.markdown(f"{i}. **{concept}**")

    if guide.detailed_notes:
        notes = downgrade_headings(guide.detailed_notes)
        notes = _clean_generated_markdown(notes, guide, depth, mode)
        st.markdown(notes)

    _display_sources_section(guide)


# ---------------------------------------------------------------------------
# Deterministic markdown cleanup
# ---------------------------------------------------------------------------

def _clean_generated_markdown(text: str, guide: StudyGuide,
                              depth: str, mode: str) -> str:
    """Post-process generated markdown to fix common LLM rendering issues."""
    # 1. Remove duplicate title that matches the app-level title (Deep Study)
    if mode == "Learn Path" and depth == "Deep Study":
        level_label = guide.difficulty.value.capitalize()
        display_name = _LEARN_PATH_DISPLAY_NAMES.get(level_label, "")
        if display_name:
            # Remove leading heading that duplicates the app title
            text = re.sub(
                r"^\s*#{1,3}\s*" + re.escape(display_name) + r"\s*\n+",
                "", text, count=1,
            )

    # 2. Convert fake checkbox lines (- [ ] or - [x]) to plain bullets
    text = re.sub(r"^(\s*)- \[[ xX]\] ", r"\1- ", text, flags=re.MULTILINE)

    # 3. Remove duplicate/empty "Learn Path Overview" headings
    text = re.sub(
        r"^#{1,4}\s+Learn Path Overview\s*\n(?=\s*#{1,4}\s+Learn Path Overview)",
        "", text, flags=re.MULTILINE,
    )
    # Remove empty "Learn Path Overview" heading with no content before next heading
    text = re.sub(
        r"^(#{1,4}\s+Learn Path Overview)\s*\n+(?=#{1,4}\s)",
        "", text, flags=re.MULTILINE,
    )

    # 4. Summary: strip any Topics block from generated notes (rendered deterministically)
    if mode == "Learn Path" and depth == "Summary":
        # Remove Topics heading + table
        text = re.sub(
            r"^#{1,4}\s+Topics\s*\n(?:(?:\|.+\n)+)",
            "", text, flags=re.MULTILINE,
        )
        # Remove Topics heading + bullet/numbered list
        text = re.sub(
            r"^#{1,4}\s+Topics\s*\n(?:(?:\s*[-*\d]+\.?\s+.+\n)+)",
            "", text, flags=re.MULTILINE,
        )
        # Remove duplicate Overview heading (already rendered by app)
        text = re.sub(
            r"^#{1,4}\s+Overview\s*\n",
            "", text, count=1, flags=re.MULTILINE,
        )
        # Remove standalone "Content" heading
        text = re.sub(
            r"^#{1,4}\s+Content\s*\n",
            "", text, flags=re.MULTILINE,
        )
        # Remove paragraph immediately after Topics table that repeats the Overview
        if guide.summary:
            overview_start = guide.summary.strip()[:80]
            if overview_start:
                text = re.sub(
                    r"^" + re.escape(overview_start[:40]) + r"[^\n]*\n*",
                    "", text, count=1,
                )

    # 5. Collapse excessive blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_learn() -> None:
    """Render the Learn section with Learning Mode selection."""
    # Scoped CSS: only code blocks, no global layout override
    st.markdown(
        "<style>"
        "pre { white-space: pre-wrap !important; word-wrap: break-word !important; "
        "overflow-x: auto !important; max-width: 100% !important; "
        "box-sizing: border-box !important; }"
        "</style>",
        unsafe_allow_html=True,
    )
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
                format_func=lambda x: f"{x} — {_LEARN_PATH_DISPLAY_NAMES[x]}",
                index=0,
                key="learn_path",
            )
        with col2:
            depth_label = st.selectbox(
                "Learning Depth",
                _LEARNING_DEPTH_LABELS,
                index=0,
                key="learn_depth",
            )

        if depth_label == "Deep Study":
            st.info(
                "🔬 **Deep Study — Intensive Mode**\n\n"
                "This mode generates a comprehensive professional handbook "
                "with one major section per topic. "
                "It uses significantly more tokens and produces much longer output "
                "than Summary mode. Estimated cost and generation time are higher."
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
                index=0,
                key="learn_depth_topic",
            )

        difficulty = DifficultyLevel.INTERMEDIATE
        style = _DEPTH_TO_STYLE[depth_label]

    btn_label = "Generate Learn Path" if learning_mode == "Learn Path" else "Generate Topic"
    generate = st.button(btn_label, key="btn_learn")

    if generate:
        spinner_msg = (
            "Generating Learning Path (Deep Study) — this may take a moment…"
            if depth_label == "Deep Study"
            else "Generating Learn Path Summary…"
        )
        with st.spinner(spinner_msg):
            from src.graphs.learn_graph import run_learn_workflow

            result = run_learn_workflow(
                topic=topic,
                difficulty=difficulty,
                style=style,
            )

        st.session_state["last_learn_result"] = result
        st.session_state["last_learn_depth"] = depth_label
        st.session_state["last_learn_mode"] = learning_mode

        guide = result.get("study_guide")
        if guide:
            st.session_state["last_study_guide"] = guide
        st.session_state["last_learn_topic"] = topic
        st.session_state["last_learn_trace"] = result.get("trace", [])
        st.session_state["last_learn_tokens"] = result.get("token_usage", {})
        _accumulate_usage_records(result.get("usage_records", []))

        # Rerun so the sidebar token counter refreshes immediately
        st.rerun()

    # Display cached result (shown after rerun or on revisit)
    if st.session_state.get("last_learn_result"):
        _show_cached_learn_result()

    _display_feedback_widget("learn", st.session_state.get("last_learn_topic", ""))
