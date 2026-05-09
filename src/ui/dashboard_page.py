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
# RAGAs content quality evaluation
# ---------------------------------------------------------------------------

def _check_ragas_available() -> tuple[bool, str]:
    """Check whether ragas and OpenAI key are available."""
    try:
        import ragas  # noqa: F401
    except ImportError:
        return False, "The `ragas` package is not installed. Run `pip install ragas>=0.4,<1`."

    from src.config import get_settings
    if not get_settings().openai_api_key:
        return False, "OpenAI API key is not configured. RAGAs requires an LLM judge."

    return True, ""


def _metric_color(value: float | None, threshold: float = 0.6) -> str:
    """Return an emoji indicator for a metric value."""
    if value is None:
        return "⬜"
    return "🟢" if value >= threshold else "🟡" if value >= 0.4 else "🔴"


def _fmt_metric(value: float | None) -> str:
    """Format a metric value for display."""
    return f"{value:.4f}" if value is not None else "N/A"


def _render_ragas_section() -> None:
    """Render the RAGAs Content Quality Evaluation section."""
    st.subheader("Content Quality Evaluation (RAGAs)")
    st.markdown(
        "RAGAs evaluates the **quality of generated Learn content** — not only "
        "retrieval accuracy, but also faithfulness to sources, answer relevancy, "
        "and context utilisation. It uses an LLM judge to score real generated "
        "study guides against retrieved contexts and reference answers."
    )

    available, reason = _check_ragas_available()
    if not available:
        st.warning(reason)
        return

    # Show cached results if available and no live report yet
    if "ragas_report" not in st.session_state:
        from src.eval.ragas_evaluation import load_ragas_results

        cached = load_ragas_results()
        if cached is not None:
            st.session_state["ragas_report"] = cached

    st.info(
        "💡 Results below are from the **latest saved benchmark**. "
        "Click the button to run a fresh evaluation (costs money and takes 1–3 min)."
    )
    st.warning(
        "⚠️ Running RAGAs evaluation calls the OpenAI API (LLM judge) for each "
        "case and metric. The default 3 cases typically cost ~$0.01–0.03 and "
        "take 1–3 minutes. Do not run repeatedly without reason.",
        icon="💰",
    )

    if st.button("▶ Run RAGAs Evaluation", key="run_ragas_eval"):
        _run_ragas_and_display()
    elif "ragas_report" in st.session_state:
        _display_ragas_report(st.session_state["ragas_report"])


def _run_ragas_and_display() -> None:
    """Execute RAGAs evaluation and store + display results."""
    from src.eval.ragas_evaluation import run_ragas_evaluation

    with st.spinner("Running RAGAs evaluation (generating content + scoring)…"):
        try:
            report = run_ragas_evaluation()
        except Exception as e:
            st.error(f"RAGAs evaluation failed: {e}")
            return

    st.session_state["ragas_report"] = report
    _display_ragas_report(report)


def _display_ragas_report(report) -> None:
    """Display a RAGAsReport in the Dashboard."""
    from src.eval.ragas_evaluation import (
        ANSWER_CORRECTNESS_NOTE,
        PRIMARY_METRICS,
        format_ragas_report,
    )

    # ── Metadata (timestamp / model / case count) ────────────────────
    meta_parts: list[str] = []
    if getattr(report, "timestamp", ""):
        meta_parts.append(f"**Run:** {report.timestamp}")
    if getattr(report, "model", ""):
        meta_parts.append(f"**Model:** {report.model}")
    if getattr(report, "case_count", 0):
        meta_parts.append(f"**Cases:** {report.case_count}")
    if meta_parts:
        st.caption(" · ".join(meta_parts))

    # ── Primary average metrics ──────────────────────────────────────
    st.markdown("#### Primary Metrics")
    primary_avg = [
        ("Faithfulness", report.avg_faithfulness),
        ("Answer Relevancy", report.avg_answer_relevancy),
        ("Context Precision", report.avg_context_precision),
        ("Context Recall", report.avg_context_recall),
    ]

    cols = st.columns(len(primary_avg))
    for col, (name, val) in zip(cols, primary_avg):
        col.metric(label=name, value=_fmt_metric(val))

    # ── Pass/fail summary (primary metrics only) ─────────────────────
    all_pass = all(v is not None and v >= 0.6 for _, v in primary_avg)
    if all_pass:
        st.success("✅ All primary metrics above 0.6 threshold — Learn quality is acceptable.")
    else:
        failing = [n for n, v in primary_avg if v is None or v < 0.6]
        st.warning(f"⚠️ Primary metrics below threshold: {', '.join(failing)}. Review recommended.")

    # ── Diagnostic metric ────────────────────────────────────────────
    st.markdown("#### Diagnostic Metric")
    diag_col, _ = st.columns([1, 3])
    diag_col.metric(
        label="Answer Correctness",
        value=_fmt_metric(report.avg_answer_correctness),
        help="Diagnostic only — not used for pass/fail",
    )
    st.caption(ANSWER_CORRECTNESS_NOTE)

    # ── Per-case breakdown ───────────────────────────────────────────
    st.markdown("#### Per-Case Breakdown")
    for r in report.results:
        label = f"{r.topic} ({r.difficulty})"
        if r.error:
            st.error(f"**{label}** — Error: {r.error}")
            continue

        with st.expander(label):
            header = (
                f"| Metric | Score | Status | Role |\n"
                f"|---|---|---|---|\n"
            )
            rows = ""
            for name, field_name, val in [
                ("Faithfulness", "faithfulness", r.faithfulness),
                ("Answer Relevancy", "answer_relevancy", r.answer_relevancy),
                ("Context Precision", "context_precision", r.context_precision),
                ("Context Recall", "context_recall", r.context_recall),
                ("Answer Correctness", "answer_correctness", r.answer_correctness),
            ]:
                role = "Primary" if field_name in PRIMARY_METRICS else "Diagnostic"
                rows += f"| {name} | {_fmt_metric(val)} | {_metric_color(val)} | {role} |\n"
            st.markdown(header + rows)
            st.caption(f"Contexts: {r.num_contexts} · Answer length: {r.answer_length:,} chars")

    # ── Raw report ───────────────────────────────────────────────────
    with st.expander("Raw RAGAs Report"):
        st.code(format_ragas_report(report), language="text")


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

    # ── Content Quality Evaluation (RAGAs) ─────────────────────────────
    _render_ragas_section()

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
