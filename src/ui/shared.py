"""Shared UI helpers and constants used across page modules."""

import re

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
    """Render source transparency for a study guide.

    Shows deduplicated unique sources as primary cards, then lists
    additional retrieved chunks in a collapsed section so reviewers
    can see the full retrieval breadth.
    """
    all_sources = guide.sources if guide else []
    sources = deduplicate_sources(all_sources) if all_sources else []
    st.markdown("#### Sources")
    st.caption("Sources used to ground this generated learning content.")

    if not sources:
        st.info(
            "No sources found. If the KB has not been ingested yet, "
            "run the ingestion pipeline first, then try again."
        )
        return

    total_retrieved = len(all_sources)
    unique_count = len(sources)
    if total_retrieved > unique_count:
        passage_word = "passage" if total_retrieved == 1 else "passages"
        file_word = "source" if unique_count == 1 else "sources"
        st.markdown(
            f"_{total_retrieved} context {passage_word} retrieved "
            f"→ {unique_count} unique {file_word} displayed._"
        )
    else:
        st.markdown(f"_{format_sources_summary(sources)}_")

    for src in sources:
        info = format_source_display(src)
        label = info["title"]
        if info["relevance_label"]:
            label += f"  ·  relevance {info['relevance_label']}"

        with st.expander(label):
            # Metadata as separate readable lines
            for key, value in info["metadata_items"]:
                st.markdown(f"**{key}:** {value}")

            # Snippet preview — always end at a sentence boundary
            snippet = info["snippet"]
            snippet = _trim_snippet_to_sentence(snippet, max_len=600)
            st.markdown(snippet)

    # Show additional passages that were deduplicated away
    if total_retrieved > unique_count:
        seen_titles = {getattr(s, "title", "") for s in sources}
        extra_lines: list[str] = []
        for src in all_sources:
            t = getattr(src, "title", "") or "Untitled"
            if t in seen_titles:
                continue
            seen_titles.add(t)
            meta = getattr(src, "metadata", {}) or {}
            file_info = meta.get("filename", "")
            line = f"- **{t}**"
            if file_info:
                line += f" ({file_info})"
            extra_lines.append(line)
        if extra_lines:
            n = len(extra_lines)
            passage_word = "passage" if n == 1 else "passages"
            with st.expander(
                f"{n} additional retrieved {passage_word} (duplicates removed)"
            ):
                for line in extra_lines:
                    st.markdown(line)


def _display_memory_section(result: dict) -> None:
    """Render memory transparency for a workflow result.

    Always renders the expander.  When no memory profile exists, a clear
    info banner is shown *inside* the expander so the container is never
    visually empty.
    """
    profile = result.get("memory_profile")
    mem = format_memory_transparency(profile)

    _EMPTY_MEMORY_MSG = (
        "No learning memory available yet. "
        "Complete quizzes and learning sessions "
        "to build personalized learning memory."
    )

    with st.expander("Memory Profile", expanded=False):
        if not mem["loaded"]:
            st.info(_EMPTY_MEMORY_MSG)
            return

        has_content = False
        if mem.get("recent_topics"):
            st.markdown("**Recent topics:** " + ", ".join(mem["recent_topics"]))
            has_content = True
        if mem.get("weak_areas"):
            st.markdown("**Recurring weak areas:** " + ", ".join(mem["weak_areas"]))
            has_content = True
        if mem.get("average_score") is not None:
            st.markdown(f"**Average score:** {mem['average_score']:.0f}%")
            has_content = True
        if mem.get("suggested_focus"):
            st.markdown("**Suggested focus:** " + ", ".join(mem["suggested_focus"]))
            has_content = True
        if mem.get("preferred_style"):
            st.markdown(f"**Preferred style:** {mem['preferred_style']}")
            has_content = True

        if not has_content:
            st.info(_EMPTY_MEMORY_MSG)


def _trim_snippet_to_sentence(text: str, max_len: int = 600) -> str:
    """Trim *text* to at most *max_len* chars, ending at a sentence boundary.

    Tries to cut at the last sentence-ending punctuation (`. `, `? `, `! `,
    or a final `.`).  Falls back to a word boundary with trailing ``...``.
    """
    if len(text) <= max_len:
        # Even short text may end mid-sentence — ensure clean ending
        stripped = text.rstrip()
        if stripped and stripped[-1] not in '.!?:;\'")':
            for sep in ['. ', '? ', '! ']:
                pos = stripped.rfind(sep)
                if pos > len(stripped) // 3:
                    return stripped[: pos + 1]
            last_dot = stripped.rfind('.')
            if last_dot > len(stripped) // 3:
                return stripped[: last_dot + 1]
            # No sentence boundary found — add ellipsis to signal truncation
            space = stripped.rfind(' ')
            if space > len(stripped) // 3:
                return stripped[:space] + " ..."
            return stripped + " ..."
        return text

    cut = text[:max_len]
    # Prefer sentence-ending punctuation
    for sep in ['. ', '? ', '! ']:
        pos = cut.rfind(sep)
        if pos > max_len // 3:
            return cut[: pos + 1]
    # Try bare period at end of a word
    last_dot = cut.rfind('.')
    if last_dot > max_len // 3:
        return cut[: last_dot + 1]
    # Fall back to word boundary
    space = cut.rfind(' ')
    if space > max_len // 3:
        return cut[:space] + " ..."
    return cut + " ..."


def _build_workflow_summary(result: dict) -> list[str]:
    """Build a human-readable workflow summary from trace entries and state.

    Returns a list of short, reviewer-friendly bullet strings with concise
    metadata (chunk counts, source quality, cache status, token usage).
    """
    steps: list[str] = []
    trace = result.get("trace", [])

    # Derive high-level steps from trace entries
    has_validate = any("validate_input" in e for e in trace)
    has_retrieve = any("retrieve_sources" in e for e in trace)
    has_topic_aware = any("topic-aware" in e for e in trace)
    has_refine = any("refine_query" in e for e in trace)
    has_generate = any("generate" in e.lower() for e in trace)
    has_cache = any("cache" in e.lower() for e in trace)

    if has_validate:
        steps.append("Input validated")

    if has_retrieve:
        label = "Topic-aware retrieval completed" if has_topic_aware else "Retrieval completed"
        # Append passage count when available
        sources = result.get("sources") or []
        if sources:
            label += f": {len(sources)} passages"
        steps.append(label)

    # Source quality — use the authoritative state flag set by assess_source_quality
    source_quality_ok = result.get("source_quality_ok")
    if source_quality_ok is not None:
        steps.append("Source quality: sufficient" if source_quality_ok else "Source quality: insufficient")
    elif has_retrieve:
        # Fallback: infer from trace text
        trace_text = " ".join(trace).lower()
        if "insufficient" in trace_text:
            steps.append("Source quality: insufficient")
        elif "sufficient" in trace_text or sources:
            steps.append("Source quality: sufficient")

    if has_refine:
        steps.append("Query refined for better results")
    if has_generate:
        steps.append("Content generated")

    if has_cache:
        cache_hit = any("cache_hit" in e.lower() or "cached result" in e.lower() for e in trace)
        if cache_hit:
            steps.append("Result cached (cache hit)")
        else:
            steps.append("Result cached")

    # Token usage
    tokens = result.get("token_usage", {}) or {}
    total = tokens.get("total_tokens")
    if total:
        steps.append(f"Tokens used: {total}")

    return steps


def _display_debug_trace(result: dict, label: str = "Learn Workflow Trace") -> None:
    """Render workflow trace with summary and raw details inside one expander."""
    with st.expander(label):
        # --- Human-readable summary ---
        summary_steps = _build_workflow_summary(result)
        if summary_steps:
            st.markdown("**Workflow Summary**")
            for step in summary_steps:
                st.markdown(f"- {step}")
            st.markdown("---")

        # --- Structured state fields ---
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
                elif lbl in ("Passages Retrieved", "Retrieval Attempts", "Query Refined"):
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

        # --- Raw trace (nested expander) ---
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


def _display_feedback_widget(
    context_type: str,
    topic: str,
    *,
    expanded: bool = False,
) -> None:
    """Display a rating + comment feedback form for learn or quiz."""
    if not topic:
        return

    key_prefix = f"fb_{context_type}"
    saved_key = f"{key_prefix}_saved"

    if st.session_state.get(saved_key):
        st.success("Feedback saved. Thank you!")
        return

    with st.expander(f"Rate this {context_type} experience", expanded=expanded):
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
