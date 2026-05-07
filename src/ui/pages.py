"""Streamlit UI page renderers for each app section."""

import streamlit as st

from src.ui.display_helpers import (
    format_graph_state_summary,
    format_memory_transparency,
    format_sources_summary,
)

# Re-export symbols from shared and learn_page for backward compatibility
from src.ui.shared import (  # noqa: F401
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
    _show_friendly_error,
)
from src.ui.learn_page import render_learn  # noqa: F401
from src.ui.quiz_page import (  # noqa: F401
    _display_hitl_save,
    _display_quiz_questions,
    _display_quiz_results,
    render_quiz,
)

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



# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

def render_progress() -> None:
    """Render the Progress section with learning memory data."""
    st.header("Progress")
    st.markdown("Track your studied topics, quiz scores, weak areas, and feedback.")
    st.info(
        "Progress is recorded when you complete a quiz and choose **Save** "
        "in the review step (Quiz → Submit → Save memory → Progress tracking)."
    )

    from src.memory.memory_service import get_recent_topics, get_weak_areas_summary

    recent = get_recent_topics(limit=10)
    if not recent:
        _show_friendly_error("empty_progress")
    else:
        st.subheader("Recent Learning Sessions")
        for evt in recent:
            weak_str = ", ".join(evt["weak_areas"]) if evt["weak_areas"] else "—"
            st.markdown(
                f"- **{evt['topic']}** — Score: {evt['score']:.0f}% · "
                f"Weak areas: {weak_str} · {evt['timestamp'][:10]}"
            )

        st.subheader("Weak Areas Summary")
        summary = get_weak_areas_summary()
        if summary:
            for area, count in sorted(summary.items(), key=lambda x: -x[1]):
                st.markdown(f"- **{area}** — appeared {count} time(s)")
        else:
            st.info("No weak areas recorded yet.")

    # Memory transparency
    st.subheader("Memory Profile")
    try:
        from src.memory.memory_service import get_user_profile_summary

        profile = get_user_profile_summary()
        mem = format_memory_transparency(profile)
        if mem["loaded"]:
            if mem.get("recent_topics"):
                st.markdown("**Recent topics:** " + ", ".join(mem["recent_topics"]))
            if mem.get("weak_areas"):
                st.markdown("**Recurring weak areas:** " + ", ".join(mem["weak_areas"]))
            if mem.get("average_score") is not None:
                st.markdown(f"**Average score:** {mem['average_score']:.0f}%")
            if mem.get("suggested_focus"):
                st.markdown("**Suggested focus topics:** " + ", ".join(mem["suggested_focus"]))
        else:
            st.info(
                "Memory profile will be built automatically as you study "
                "and save quiz results."
            )
    except Exception:
        st.info("Memory profile not available.")

    # Feedback section
    st.subheader("Recent Feedback")
    from src.memory.feedback_service import get_recent_feedback, get_feedback_summary

    fb_entries = get_recent_feedback(limit=5)
    if fb_entries:
        for fb in fb_entries:
            stars = fb["rating"]
            comment = fb["comment"] if fb["comment"] else "—"
            st.markdown(
                f"- Rating: {stars}/5 — **{fb['context_type']}** / {fb['topic']} — "
                f"{comment} · {fb['timestamp'][:10]}"
            )
    else:
        st.info("No feedback recorded yet.")

    fb_summary = get_feedback_summary()
    if fb_summary.get("total_count", 0) > 0:
        st.subheader("Feedback Summary")
        st.markdown(f"- **Average rating:** {fb_summary['average_rating']}")
        st.markdown(f"- **Total feedback entries:** {fb_summary['total_count']}")
        if fb_summary.get("suggestion"):
            st.markdown(f"- **Personalization suggestion:** {fb_summary['suggestion']}")


# ---------------------------------------------------------------------------
# Cost tracking helpers
# ---------------------------------------------------------------------------

def _accumulate_usage_records(records: list[dict]) -> None:
    """Append new usage records to session-level accumulator."""
    existing = st.session_state.get("session_usage_records", [])
    st.session_state["session_usage_records"] = existing + records


def _display_session_cost_summary() -> None:
    """Display aggregated token/cost data for the current session."""
    records = st.session_state.get("session_usage_records", [])
    if not records:
        st.info("No usage data yet. Generate a Learn Path or quiz to see cost estimates.")
        return

    from src.services.cost_tracker import aggregate_usage

    summary = aggregate_usage(records)
    rows = "".join(
        f"| {op['operation']} | {op['total_tokens']:,} | ${op['estimated_cost_usd']:.6f} |\n"
        for op in summary["operations"]
    )
    st.markdown(
        f"| Operation | Tokens | Est. Cost |\n"
        f"|---|---|---|\n"
        f"{rows}"
        f"| **Total** | **{summary['total_tokens']:,}** | **${summary['estimated_cost_usd']:.6f}** |"
    )

    with st.expander("Raw details"):
        st.markdown(f"**Total records:** {len(records)}")
        st.json(records)


# ---------------------------------------------------------------------------
# Dashboard (formerly Advanced / Debug)
# ---------------------------------------------------------------------------

def render_advanced() -> None:
    """Render the Dashboard section."""
    st.header("Dashboard")
    st.markdown(
        "Application overview, observability, cost tracking, "
        "and workflow diagnostics."
    )

    # ── Overview ──────────────────────────────────────────────────────────
    st.subheader("Overview")
    from src.config import get_settings
    from src.services.observability import format_tracing_status, get_tracing_status

    settings = get_settings()
    status = get_tracing_status()
    info = format_tracing_status(status)

    api_ok = "Configured" if settings.openai_api_key else "Missing"
    st.markdown(
        f"| Setting | Value |\n"
        f"|---|---|\n"
        f"| OpenAI API | {api_ok} |\n"
        f"| Model | {settings.app_default_model} |\n"
        f"| LangSmith | {info['status_label']} |\n"
        f"| Project | {info['project']} |\n"
        f"| Embedding | {settings.embedding_model} |\n"
        f"| Chunk Size / Overlap | {settings.chunk_size} / {settings.chunk_overlap} |"
    )

    with st.expander("All application settings"):
        st.json({
            "default_model": settings.app_default_model,
            "log_level": settings.app_log_level,
            "langchain_tracing_v2": settings.langchain_tracing_v2,
            "langchain_project": settings.langchain_project,
            "langchain_endpoint": settings.langchain_endpoint,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "api_key_configured": bool(settings.openai_api_key),
        })

    # ── Costs ─────────────────────────────────────────────────────────────
    st.subheader("Costs")
    _display_session_cost_summary()

    # ── Memory ────────────────────────────────────────────────────────────
    st.subheader("Memory")
    try:
        from src.memory.memory_service import get_user_profile_summary

        profile = get_user_profile_summary()
        mem = format_memory_transparency(profile)
        if mem["loaded"]:
            avg = f"{mem['average_score']:.0f}%" if mem.get("average_score") is not None else "—"
            weak = ", ".join(mem.get("weak_areas", [])) or "—"
            recent = ", ".join(mem.get("recent_topics", [])) or "—"
            focus = ", ".join(mem.get("suggested_focus", [])) or "—"
            st.markdown(
                f"| Field | Value |\n"
                f"|---|---|\n"
                f"| Average Score | {avg} |\n"
                f"| Weak Areas | {weak} |\n"
                f"| Recent Topics | {recent} |\n"
                f"| Suggested Focus | {focus} |"
            )
        else:
            st.info(
                "Memory profile will be built automatically as you study "
                "and save quiz results."
            )
    except Exception:
        st.info("Memory profile not available.")

    # ── Feedback ──────────────────────────────────────────────────────────
    st.subheader("Feedback")
    from src.memory.feedback_service import get_feedback_summary

    fb_summary = get_feedback_summary()
    if fb_summary.get("total_count", 0) > 0:
        suggestion = fb_summary.get("suggestion", "—") or "—"
        st.markdown(
            f"| Field | Value |\n"
            f"|---|---|\n"
            f"| Average Rating | {fb_summary['average_rating']}/5 |\n"
            f"| Total Entries | {fb_summary['total_count']} |\n"
            f"| Suggestion | {suggestion} |"
        )
        with st.expander("Raw feedback details"):
            st.json(fb_summary)
    else:
        st.info("No feedback data yet.")

    # ── Workflow Traces ───────────────────────────────────────────────────
    st.subheader("Workflow Traces")

    # Learn trace
    with st.expander("Learn Workflow Trace"):
        learn_result = st.session_state.get("last_learn_result", {})
        trace = learn_result.get("trace") or st.session_state.get("last_learn_trace", [])
        if trace or learn_result:
            fields = format_graph_state_summary(learn_result) if learn_result else []
            if fields:
                for f in fields:
                    st.markdown(f"- **{f['label']}:** {f['value']}")
            tokens = learn_result.get("token_usage") or st.session_state.get("last_learn_tokens", {})
            if tokens and tokens.get("total_tokens"):
                st.markdown(
                    f"**Tokens:** {tokens.get('total_tokens', 0):,} "
                    f"(prompt: {tokens.get('prompt_tokens', 0):,}, "
                    f"completion: {tokens.get('completion_tokens', 0):,})"
                )
            with st.expander("Raw trace"):
                if trace:
                    for entry in trace:
                        st.text(entry)
                else:
                    st.info("No trace entries.")
        else:
            st.info("No trace available yet. Generate a Learn Path first.")

    # Quiz trace
    with st.expander("Quiz Workflow Trace"):
        quiz_gen = st.session_state.get("last_quiz_gen_result", {})
        trace = quiz_gen.get("trace") or st.session_state.get("last_quiz_trace", [])
        if trace or quiz_gen:
            fields = format_graph_state_summary(quiz_gen) if quiz_gen else []
            if fields:
                for f in fields:
                    st.markdown(f"- **{f['label']}:** {f['value']}")
            tokens = quiz_gen.get("token_usage") or st.session_state.get("last_quiz_tokens", {})
            if tokens and tokens.get("total_tokens"):
                st.markdown(
                    f"**Tokens:** {tokens.get('total_tokens', 0):,} "
                    f"(prompt: {tokens.get('prompt_tokens', 0):,}, "
                    f"completion: {tokens.get('completion_tokens', 0):,})"
                )
            with st.expander("Raw trace"):
                if trace:
                    for entry in trace:
                        st.text(entry)
                else:
                    st.info("No trace entries.")
        else:
            st.info("No quiz trace available yet. Generate a quiz first.")
