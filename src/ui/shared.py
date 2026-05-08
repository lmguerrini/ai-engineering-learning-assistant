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
_LEARN_PATH_DISPLAY_NAMES = {
    "Beginner": "Foundations of LLM Application Development",
    "Intermediate": "Building Applications with LangChain, RAGs, and Streamlit",
    "Advanced": "AI Agents and Orchestration",
}
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

# Stable, professionally-capitalised topic lists per Learn Path level
_LEARN_PATH_STABLE_TOPICS: dict[str, list[str]] = {
    "Beginner": [
        "LLM Basics",
        "Prompt Engineering",
        "Development Environment",
        "API Usage",
        "First Working Application",
    ],
    "Intermediate": [
        "LangChain Chains",
        "Retrieval-Augmented Generation",
        "Function Calling",
        "Tool Integration",
        "Streamlit UI",
        "Evaluation",
    ],
    "Advanced": [
        "LangGraph State Management",
        "Agentic RAG",
        "Long-Term Memory",
        "Human-in-the-Loop",
        "Checkpointers",
        "Observability",
        "Production Deployment",
    ],
}

# Concise descriptions for each stable topic (used in Deep Study Topics list)
_LEARN_PATH_TOPIC_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "Beginner": {
        "LLM Basics": "Understand large language model fundamentals, tokenization, and inference.",
        "Prompt Engineering": "Design effective prompts using techniques like few-shot, chain-of-thought, and role prompting.",
        "Development Environment": "Set up Python tooling, virtual environments, and API keys for LLM development.",
        "API Usage": "Interact with OpenAI and other LLM provider APIs programmatically.",
        "First Working Application": "Build and deploy a simple end-to-end LLM-powered application.",
    },
    "Intermediate": {
        "LangChain Chains": "Build reusable chains and workflows with language model components.",
        "Retrieval-Augmented Generation": "Combine vector search with LLM generation for knowledge-grounded answers.",
        "Function Calling": "Enable LLMs to invoke structured functions and return typed outputs.",
        "Tool Integration": "Connect external tools, APIs, and databases to LLM workflows.",
        "Streamlit UI": "Create interactive web interfaces for LLM applications with Streamlit.",
        "Evaluation": "Measure and improve LLM output quality with automated and human evaluation.",
    },
    "Advanced": {
        "LangGraph State Management": "Design stateful, multi-step agent workflows with LangGraph.",
        "Agentic RAG": "Build autonomous retrieval agents that decide when and what to retrieve.",
        "Long-Term Memory": "Persist conversation history and user context across sessions.",
        "Human-in-the-Loop": "Add approval gates and human oversight to automated agent pipelines.",
        "Checkpointers": "Save and restore agent state for resilience and debugging.",
        "Observability": "Monitor, trace, and debug LLM applications in development and production.",
        "Production Deployment": "Deploy, scale, and operate LLM applications in production environments.",
    },
}

# Estimated effort per topic (varied, realistic)
_LEARN_PATH_TOPIC_EFFORT: dict[str, dict[str, str]] = {
    "Beginner": {
        "LLM Basics": "~2–3 hours",
        "Prompt Engineering": "~3–4 hours",
        "Development Environment": "~1–2 hours",
        "API Usage": "~2–3 hours",
        "First Working Application": "~3–5 hours",
    },
    "Intermediate": {
        "LangChain Chains": "~3–4 hours",
        "Retrieval-Augmented Generation": "~4–5 hours",
        "Function Calling": "~2–3 hours",
        "Tool Integration": "~3–4 hours",
        "Streamlit UI": "~3–4 hours",
        "Evaluation": "~2–3 hours",
    },
    "Advanced": {
        "LangGraph State Management": "~4–5 hours",
        "Agentic RAG": "~4–6 hours",
        "Long-Term Memory": "~3–4 hours",
        "Human-in-the-Loop": "~3–4 hours",
        "Checkpointers": "~2–3 hours",
        "Observability": "~3–4 hours",
        "Production Deployment": "~4–6 hours",
    },
}

# Learn Path mode: maps path level to a guided topic string
_LEARN_PATH_TOPIC_MAP = {
    "Beginner": (
        "Foundations of LLM Application Development: LLM basics, prompt engineering, "
        "development environment, API usage, and first working application"
    ),
    "Intermediate": (
        "Building Applications with LangChain, RAGs, and Streamlit: LangChain chains, "
        "retrieval-augmented generation, function calling, tool integration, "
        "Streamlit UI, and evaluation"
    ),
    "Advanced": (
        "AI Agents and Orchestration: LangGraph state management, agentic RAG, "
        "long-term memory, human-in-the-loop, checkpointers, observability, "
        "and production deployment"
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
            snippet = info["snippet"]
            # Show a clean, word-boundary-aware preview
            if len(snippet) > 500:
                cut = snippet[:500].rsplit(" ", 1)[0]
                snippet = cut + " …"
            st.markdown(snippet)


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
