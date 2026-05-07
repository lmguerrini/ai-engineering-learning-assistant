"""Dashboard (formerly Advanced / Debug) page renderer."""

import streamlit as st

from src.ui.display_helpers import (
    format_graph_state_summary,
    format_memory_transparency,
)


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
