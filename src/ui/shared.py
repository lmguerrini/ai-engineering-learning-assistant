"""Shared UI helpers and constants used across page modules."""

import streamlit as st

from src.schemas import DifficultyLevel, ResponseStyle, StudyGuide
from src.ui.display_helpers import (
    deduplicate_sources,
    downgrade_headings,
    format_error_message,
    format_graph_state_summary,
    format_memory_transparency,
    format_source_display,
    format_sources_summary,
)

# ---------------------------------------------------------------------------
# User-facing label mappings
# ---------------------------------------------------------------------------

_LEARN_PATH_LABELS = ["Beginner", "Intermediate", "Advanced"]
_LEARN_PATH_TO_ENUM = {
    "Beginner": DifficultyLevel.BEGINNER,
    "Intermediate": DifficultyLevel.INTERMEDIATE,
    "Advanced": DifficultyLevel.ADVANCED,
}
_LEARNING_DEPTH_LABELS = ["Summary", "Deep Study"]
_DEPTH_TO_STYLE = {
    "Summary": ResponseStyle.CONCISE,
    "Deep Study": ResponseStyle.DETAILED,
}
_LEARNING_MODE_LABELS = ["Learn Path", "Topic"]

# Learn Path mode: maps path level to a guided topic string
_LEARN_PATH_TOPIC_MAP = {
    "Beginner": (
        "Foundations of AI Engineering: LLM basics, prompt engineering, "
        "development environment, and API usage"
    ),
    "Intermediate": (
        "Chains, RAG, and tools: LangChain chains, retrieval-augmented generation, "
        "function calling, tool integration, and evaluation"
    ),
    "Advanced": (
        "Agents and orchestration: LangGraph state management, agentic RAG, "
        "long-term memory, human-in-the-loop, observability, and production deployment"
    ),
}


# ---------------------------------------------------------------------------
# Available topics for the Learn and Quiz sections
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
# Shared UI helpers
# ---------------------------------------------------------------------------

def _show_friendly_error(error_type: str) -> None:
    """Display a user-friendly error block."""
    err = format_error_message(error_type)
    st.warning(f"**{err['title']}** — {err['message']}")
    st.caption(err["suggestion"])


def _display_sources_section(guide: StudyGuide) -> None:
    """Render source transparency for a study guide."""
    sources = deduplicate_sources(guide.sources) if guide else []
    st.markdown(f"#### Sources — {format_sources_summary(sources)}")

    if not sources:
        st.info(
            "No sources found. If the KB has not been ingested yet, "
            "run the ingestion pipeline first, then try again."
        )
        return

    for src in sources:
        info = format_source_display(src)
        meta_parts = [f"{k}: {v}" for k, v in info["metadata_items"]]
        meta_str = " · ".join(meta_parts) if meta_parts else ""
        label = info["title"]
        if meta_str:
            label += f" ({meta_str})"
        with st.expander(label):
            st.caption(f"Relevance: {info['relevance_label']}")
            st.text(info["snippet"][:300])


def _display_memory_section(result: dict) -> None:
    """Render memory transparency for a workflow result."""
    profile = result.get("memory_profile")
    mem = format_memory_transparency(profile)

    with st.expander("Memory Profile"):
        if not mem["loaded"]:
            st.info(
                "Memory profile will be built automatically as you study "
                "and save quiz results."
            )
            return

        if mem.get("recent_topics"):
            st.markdown("**Recent topics:** " + ", ".join(mem["recent_topics"]))
        if mem.get("weak_areas"):
            st.markdown("**Recurring weak areas:** " + ", ".join(mem["weak_areas"]))
        if mem.get("average_score") is not None:
            st.markdown(f"**Average score:** {mem['average_score']:.0f}%")
        if mem.get("suggested_focus"):
            st.markdown("**Suggested focus:** " + ", ".join(mem["suggested_focus"]))
        if mem.get("preferred_style"):
            st.markdown(f"**Preferred style:** {mem['preferred_style']}")


def _display_debug_trace(result: dict, label: str = "Workflow Trace") -> None:
    """Render workflow trace grouped into logical sections."""
    with st.expander(label):
        fields = format_graph_state_summary(result)
        if fields:
            request_fields = []
            retrieval_fields = []
            memory_fields = []
            token_fields = []
            for f in fields:
                lbl = f["label"]
                if lbl in ("Topic", "Learn Path", "Learning Depth", "Learning Mode"):
                    request_fields.append(f)
                elif lbl in ("Sources Retrieved", "Retrieval Attempts", "Query Refined"):
                    retrieval_fields.append(f)
                elif lbl in ("Memory Profile",):
                    memory_fields.append(f)
                elif lbl in ("Total Tokens",):
                    token_fields.append(f)

            if request_fields:
                st.markdown("**Request**")
                for f in request_fields:
                    st.markdown(f"- {f['label']}: {f['value']}")
            if retrieval_fields:
                st.markdown("**Retrieval**")
                for f in retrieval_fields:
                    st.markdown(f"- {f['label']}: {f['value']}")
            if memory_fields:
                st.markdown("**Memory**")
                for f in memory_fields:
                    st.markdown(f"- {f['label']}: {f['value']}")
            if token_fields:
                st.markdown("**Token Usage**")
                for f in token_fields:
                    st.markdown(f"- {f['label']}: {f['value']}")
            st.markdown("---")

        trace = result.get("trace", [])
        with st.expander("Raw trace"):
            if trace:
                for entry in trace:
                    st.text(entry)
            else:
                st.info("No trace entries recorded.")

        tokens = result.get("token_usage", {})
        if tokens and tokens.get("total_tokens"):
            with st.expander("Raw token details"):
                st.json(tokens)


def _display_feedback_widget(context_type: str, topic: str) -> None:
    """Display a rating + comment feedback form for learn or quiz."""
    if not topic:
        return

    key_prefix = f"fb_{context_type}"
    saved_key = f"{key_prefix}_saved"

    if st.session_state.get(saved_key):
        st.success("Feedback saved. Thank you!")
        return

    with st.expander(f"Rate this {context_type} experience"):
        rating = st.slider(
            "Rating", min_value=1, max_value=5, value=4, key=f"{key_prefix}_rating",
        )
        comment = st.text_input(
            "Comment (optional)", key=f"{key_prefix}_comment",
        )
        if st.button("Submit Feedback", key=f"{key_prefix}_btn"):
            from src.memory.feedback_service import save_feedback

            save_feedback(
                context_type=context_type,
                topic=topic,
                rating=rating,
                comment=comment,
            )
            st.session_state[saved_key] = True


def _accumulate_usage_records(records: list[dict]) -> None:
    """Append new usage records to session-level accumulator."""
    existing = st.session_state.get("session_usage_records", [])
    st.session_state["session_usage_records"] = existing + records
