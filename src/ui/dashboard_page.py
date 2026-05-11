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
    from src.ui.shared import _accumulate_usage_records as _shared_accumulate_usage_records

    _shared_accumulate_usage_records(records)


def _get_session_usage_summary() -> dict | None:
    """Return aggregated token/cost data for the current session when available."""
    records = st.session_state.get("session_usage_records", [])
    if not records:
        return None

    from src.services.cost_tracker import aggregate_usage

    return aggregate_usage(records)


def _has_cache_hit(result: dict) -> bool:
    """Return whether a stored workflow result came from cache."""
    trace = result.get("trace", [])
    return any("cache hit" in str(step).lower() for step in trace)


def _estimate_result_cost_usd(result: dict) -> float:
    """Return the estimated USD cost for one stored workflow result."""
    records = result.get("usage_records", []) or []
    return round(sum(float(rec.get("estimated_cost_usd", 0.0) or 0.0) for rec in records), 6)


def _operation_category(operation: str) -> str:
    """Map a low-level usage operation to a reviewer-friendly category."""
    if operation.startswith("learn_"):
        return "Learn"
    if operation.startswith("quiz_"):
        return "Quiz"
    return "System"


def _operation_label(operation: str) -> str:
    """Humanize a tracked operation name for dashboard display."""
    return operation.replace("_", " ").strip().title()


def _format_bool_label(value: bool) -> str:
    """Return a compact on/off label for dashboard tables."""
    return "On" if value else "Off"


def _format_context_value(value) -> str:
    """Format optional context values for reviewer-facing tables."""
    if value in (None, ""):
        return "—"
    if isinstance(value, bool):
        return _format_bool_label(value)
    return str(value)


def _build_capability_registry_rows(
    *,
    kb_health: dict,
    tracing_info: dict,
    ragas_available: bool,
    memory_loaded: bool,
    feedback_count: int,
    progressive_streaming: bool,
    cache_bypass: bool,
    has_api_key: bool,
) -> list[dict]:
    """Build reviewer-facing rows that describe the app's agent capabilities."""
    collections = kb_health.get("collections", {})
    curated = collections.get("curated", {})
    official = collections.get("official", {})
    curated_ready = bool(curated.get("chunk_count"))
    official_ready = bool(official.get("chunk_count"))

    return [
        {
            "Capability": "Curated KB Retrieval",
            "Used By": "Learn, Quiz",
            "Status": "Active" if curated_ready else "Off",
            "Mode": "Required",
            "User Control": "Automatic",
            "Description": "Primary grounded retrieval over curated course notes for Learn and Quiz.",
        },
        {
            "Capability": "Official Docs Retrieval",
            "Used By": "Learn, Quiz",
            "Status": "Available" if official_ready else "Off",
            "Mode": "Optional",
            "User Control": "Automatic",
            "Description": "Supplements retrieval with indexed official docs when domain matching applies.",
        },
        {
            "Capability": "Memory Profile",
            "Used By": "Learn, Quiz, Progress, Dashboard",
            "Status": "Active" if memory_loaded else "Available",
            "Mode": "Optional",
            "User Control": "Save quiz results",
            "Description": "Uses stored learner history to personalize study, quizzes, and progress summaries.",
        },
        {
            "Capability": "Feedback Logger",
            "Used By": "Learn, Quiz, Progress, Dashboard",
            "Status": "Active" if feedback_count > 0 else "Available",
            "Mode": "Optional",
            "User Control": "Submit in Learn/Quiz",
            "Description": "Stores ratings and comments that feed reviewer-facing feedback signals.",
        },
        {
            "Capability": "Cost Tracker",
            "Used By": "Learn, Quiz, Dashboard",
            "Status": "Active",
            "Mode": "Required",
            "User Control": "Automatic",
            "Description": "Captures token usage and estimated cost for each tracked LLM operation.",
        },
        {
            "Capability": "LangSmith Tracing",
            "Used By": "Learn, Quiz, Dashboard",
            "Status": "Active" if tracing_info.get("tracing_enabled") else "Off",
            "Mode": "Optional",
            "User Control": "Environment config",
            "Description": "Publishes workflow and LLM traces to LangSmith when tracing is enabled.",
        },
        {
            "Capability": "RAGAs Evaluator",
            "Used By": "Dashboard",
            "Status": "Available" if ragas_available else "Off",
            "Mode": "Manual",
            "User Control": "Run in Dashboard",
            "Description": "Runs manual Learn benchmarks with LLM-judged RAGAs quality metrics.",
        },
        {
            "Capability": "KB Rebuild Tool",
            "Used By": "Dashboard",
            "Status": "Available" if has_api_key else "Off",
            "Mode": "Manual",
            "User Control": "Run in Dashboard",
            "Description": "Rebuilds curated and official KB collections and refreshes health metadata.",
        },
        {
            "Capability": "Progressive Streaming",
            "Used By": "Learn",
            "Status": "Active" if progressive_streaming else "Off",
            "Mode": "User-controlled",
            "User Control": "Controlled in Learn",
            "Description": "Replays eligible Deep Study output progressively without changing final Learn content.",
        },
        {
            "Capability": "Cache Bypass",
            "Used By": "Learn",
            "Status": "Active" if cache_bypass else "Off",
            "Mode": "User-controlled",
            "User Control": "Controlled in Learn",
            "Description": "Forces fresh Learn generation instead of reusing a cached result.",
        },
    ]


def _build_session_operation_rows(records: list[dict]) -> list[dict]:
    """Convert stored usage records into compact dashboard table rows."""
    return [
        {
            "Type": _operation_category(rec.get("operation", "unknown")),
            "Mode": _format_context_value(rec.get("learning_mode")),
            "Depth": _format_context_value(rec.get("learning_depth")),
            "Stream": _format_context_value(rec.get("progressive_streaming")),
            "Bypass": _format_context_value(rec.get("cache_bypass")),
            "Cache": _format_context_value(rec.get("cache_hit")),
            "Model": rec.get("model", "Unknown"),
            "Operation": _operation_label(rec.get("operation", "unknown")),
            "Tokens": f"{rec.get('total_tokens', 0):,}",
            "Cost": f"${float(rec.get('estimated_cost_usd', 0.0) or 0.0):.6f}",
        }
        for rec in records
    ]


def _format_ragas_case_label(result) -> str:
    """Return a reviewer-friendly RAGAs case label."""
    difficulty = getattr(result, "difficulty", "") or ""
    difficulty_label = difficulty.capitalize() if difficulty else "Unknown"
    return f"{result.topic} ({difficulty_label})"


def _metric_status_label(field_name: str, value: float | None) -> str:
    """Return the dashboard status label for one RAGAs metric row."""
    return _metric_color(value)


def _display_latest_run_contexts(learn_result: dict, quiz_result: dict) -> None:
    """Render concise run-level context before the raw operation breakdown."""
    st.markdown("#### Latest Run Context")
    st.caption(
        "These tables show the latest Learn run and latest Quiz run only. "
        "The operation table below includes all tracked LLM calls from this app session, "
        "which is why progressive Learn runs may appear as multiple rows."
    )

    learn_col, quiz_col = st.columns(2)
    with learn_col:
        st.markdown("##### Learn")
        if learn_result:
            learning_mode = st.session_state.get("last_learn_mode", "—")
            learning_depth = st.session_state.get("last_learn_depth", "—")
            progressive = st.session_state.get("last_learn_progressive_streaming", False)
            regenerate = st.session_state.get("last_learn_force_regenerate", False)
            tokens = learn_result.get("token_usage", {})
            st.markdown(
                f"| Field | Value |\n"
                f"|---|---|\n"
                f"| Learning Mode | {learning_mode} |\n"
                f"| Learning Depth | {learning_depth} |\n"
                f"| Progressive Streaming | {_format_bool_label(progressive)} |\n"
                f"| Cache Bypass | {_format_bool_label(regenerate)} |\n"
                f"| Cache Hit | {_format_bool_label(_has_cache_hit(learn_result))} |\n"
                f"| Tokens | {tokens.get('total_tokens', 0):,} |\n"
                f"| Estimated Cost | ${_estimate_result_cost_usd(learn_result):.6f} |"
            )
        else:
            st.info(
                "No Learn run captured yet. Generate a Learn topic or Learn Path to populate run context."
            )

    with quiz_col:
        st.markdown("##### Quiz")
        if quiz_result:
            tokens = quiz_result.get("token_usage", {})
            topic = quiz_result.get("topic") or st.session_state.get("quiz_selected_topic", "—")
            st.markdown(
                f"| Field | Value |\n"
                f"|---|---|\n"
                f"| Topic | {topic} |\n"
                f"| Cache Hit | {_format_bool_label(_has_cache_hit(quiz_result))} |\n"
                f"| Tokens | {tokens.get('total_tokens', 0):,} |\n"
                f"| Estimated Cost | ${_estimate_result_cost_usd(quiz_result):.6f} |"
            )
        else:
            st.info(
                "No Quiz run captured yet. Generate and evaluate a quiz to populate run context."
            )


def _display_session_cost_summary() -> None:
    """Display aggregated token/cost data for the current session."""
    records = st.session_state.get("session_usage_records", [])
    summary = _get_session_usage_summary()
    if not summary:
        st.info(
            "No session usage data yet. Generate a Learn topic or quiz to populate "
            "token and cost tracking."
        )
        return

    table_rows = _build_session_operation_rows(records)
    st.markdown("#### All Session Operations")
    st.caption(
        "The table below includes all tracked LLM operations from this app session across "
        "Learn and Quiz. Cache hits typically reuse prior outputs and add no new token usage."
    )
    st.dataframe(
        table_rows,
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Session total: {summary['total_tokens']:,} tokens · ${summary['estimated_cost_usd']:.6f}"
    )

    with st.expander("Raw details"):
        st.markdown(f"**Total records:** {len(records)}")
        st.json(records)


def _display_capability_registry_section(
    *,
    kb_health: dict,
    tracing_info: dict,
    ragas_available: bool,
    memory_loaded: bool,
    feedback_count: int,
    progressive_streaming: bool,
    cache_bypass: bool,
    has_api_key: bool,
) -> None:
    """Render a reviewer-facing table of the app's agent capabilities."""
    st.subheader("Agent Capabilities / Tool Registry")
    st.caption(
        "Maps the app's grounded retrieval, personalization, observability, and "
        "manual review tools without adding new risky controls."
    )
    rows = _build_capability_registry_rows(
        kb_health=kb_health,
        tracing_info=tracing_info,
        ragas_available=ragas_available,
        memory_loaded=memory_loaded,
        feedback_count=feedback_count,
        progressive_streaming=progressive_streaming,
        cache_bypass=cache_bypass,
        has_api_key=has_api_key,
    )
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption(
        "Required capabilities stay automatic. Optional controls remain in Learn "
        "or Dashboard where they already exist."
    )


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


def _get_ragas_report():
    """Load the latest cached RAGAs report into session state when available."""
    if "ragas_report" in st.session_state:
        return st.session_state["ragas_report"]

    from src.eval.ragas_evaluation import load_ragas_results

    cached = load_ragas_results()
    if cached is not None:
        st.session_state["ragas_report"] = cached
    return cached


def _ragas_snapshot_value(report) -> str:
    """Return a short reviewer-facing RAGAs readiness label."""
    if report is None:
        return "Not run"
    return "Cached"


def _trace_snapshot_value(result: dict, trace_key: str) -> str:
    """Return a short workflow readiness label from stored trace state."""
    trace = result.get("trace") or st.session_state.get(trace_key, [])
    return "Ready" if trace or result else "—"


def _display_workflow_trace_panel(
    title: str,
    result: dict,
    *,
    trace_key: str,
    tokens_key: str,
    empty_message: str,
) -> None:
    """Render one workflow-readiness panel with concise summary and raw trace details."""
    with st.expander(title):
        trace = result.get("trace") or st.session_state.get(trace_key, [])
        if not trace and not result:
            st.info(empty_message)
            return

        fields = format_graph_state_summary(result) if result else []
        if fields:
            st.markdown("**Run summary**")
            for field in fields:
                st.markdown(f"- **{field['label']}:** {field['value']}")

        tokens = result.get("token_usage") or st.session_state.get(tokens_key, {})
        if tokens and tokens.get("total_tokens"):
            st.caption(
                f"Tokens: {tokens.get('total_tokens', 0):,} "
                f"(prompt: {tokens.get('prompt_tokens', 0):,}, "
                f"completion: {tokens.get('completion_tokens', 0):,})"
            )

        with st.expander("Raw trace"):
            if trace:
                for entry in trace:
                    st.text(entry)
            else:
                st.info("No trace entries.")


def _render_ragas_section() -> None:
    """Render the RAGAs Content Quality Evaluation section."""
    st.subheader("Evaluation Readiness (RAGAs)")
    st.markdown(
        "RAGAs scores generated Learn content for faithfulness, relevancy, and "
        "context use. The saved benchmark is review-safe to inspect; reruns stay "
        "manual because they call an LLM judge and cost money."
    )

    available, reason = _check_ragas_available()
    if not available:
        st.warning(reason)
        return

    # Show cached results if available and no live report yet
    _get_ragas_report()

    st.info(
        "Results below are from the **latest saved benchmark**. "
        "Click the button to run a fresh evaluation (costs money and takes 5–10 min)."
    )
    st.warning(
        "⚠️ Running RAGAs evaluation calls the OpenAI API (LLM judge) for each "
        "case and metric. The default 3 cases typically cost ~$0.01–0.03 and "
        "take 5–10 minutes. Do not run repeatedly without reason.",
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
    st.caption(
        "Borderline yellow scores deserve review, but they are not automatic failures. "
        "Answer Relevancy near the threshold can shift slightly across judge runs."
    )

    # ── Diagnostic metric ────────────────────────────────────────────
    st.markdown("#### Diagnostic Metric")
    diag_col, diag_note_col = st.columns([1, 3])
    diag_col.metric(
        label="Answer Correctness",
        value=_fmt_metric(report.avg_answer_correctness),
        help="Diagnostic only — not used for pass/fail or primary Learn quality decisions.",
    )
    diag_note_col.info(
        "Low Answer Correctness usually reflects mismatch between a long generated study guide "
        "and a short reference answer, not poor grounded Learn quality."
    )
    st.caption(ANSWER_CORRECTNESS_NOTE)

    # ── Per-case breakdown ───────────────────────────────────────────
    st.markdown("#### Per-Case Breakdown")
    for r in report.results:
        label = _format_ragas_case_label(r)
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
                rows += (
                    f"| {name} | {_fmt_metric(val)} | "
                    f"{_metric_status_label(field_name, val)} | {role} |\n"
                )
            st.markdown(header + rows)
            st.caption(
                f"Contexts: {r.num_contexts} · Answer length: {r.answer_length:,} chars · "
                f"Answer Correctness is diagnostic only."
            )

    # ── Raw report ───────────────────────────────────────────────────
    with st.expander("Raw RAGAs Report"):
        st.code(format_ragas_report(report), language="text")


# ---------------------------------------------------------------------------
# Knowledge Base Health
# ---------------------------------------------------------------------------

def _display_kb_health_section() -> None:
    """Render KB / Chroma freshness and manual rebuild controls."""
    from src.config import get_settings
    from src.kb.index_health import get_kb_index_health, rebuild_kb_index

    settings = get_settings()
    health = get_kb_index_health()

    st.subheader("Knowledge Base Health")
    st.caption(
        "Tracks whether the local Chroma index matches the current curated and official markdown files."
    )

    if health["status"] == "up_to_date":
        st.success("✅ KB index is up to date.")
    elif health["status"] == "metadata_missing":
        st.info(
            "Chroma collections already exist, but no KB health metadata baseline "
            "was found yet. Rebuild once to enable freshness tracking."
        )
    elif health["status"] == "outdated":
        st.warning("KB index is outdated. Rebuild recommended.")
    else:
        st.warning("KB index is missing or incomplete. Rebuild required.")

    summary_left, summary_right = st.columns(2)
    with summary_left:
        st.markdown(
            f"| Field | Value |\n"
            f"|---|---|\n"
            f"| Index Status | {health['status_label']} |\n"
            f"| Reindex Needed | {'Yes' if health['reindex_required'] else 'No'} |\n"
            f"| Embedding Model | {health['embedding_model']} |\n"
            f"| Last Rebuild | {health.get('last_rebuild_at') or '—'} |"
        )
    with summary_right:
        curated = health["collections"]["curated"]
        official = health["collections"]["official"]
        st.markdown(
            f"| Field | Value |\n"
            f"|---|---|\n"
            f"| Raw Docs | {health['raw_docs_count']} |\n"
            f"| Official Docs | {health['official_docs_count']} |\n"
            f"| Curated Chunks / Sources | {curated.get('chunk_count') if curated.get('chunk_count') is not None else '—'} / {curated.get('source_count') if curated.get('source_count') is not None else '—'} |\n"
            f"| Official Chunks / Sources | {official.get('chunk_count') if official.get('chunk_count') is not None else '—'} / {official.get('source_count') if official.get('source_count') is not None else '—'} |"
        )

    if health["notes"] and health["status"] != "metadata_missing":
        st.markdown("**Why a rebuild may be needed**")
        for note in health["notes"]:
            st.markdown(f"- {note}")

    st.warning(
        "Rebuilding the KB index may take time and may call the embedding model. "
        "The app will not rebuild automatically."
    )
    rebuild_disabled = not bool(settings.openai_api_key)
    if rebuild_disabled:
        st.caption("OpenAI API key required for KB rebuild.")

    if st.button(
        "Rebuild KB Index",
        key="btn_rebuild_kb_index",
        disabled=rebuild_disabled,
    ):
        with st.spinner("Rebuilding KB index..."):
            rebuild_kb_index()
        st.session_state["kb_rebuild_notice"] = "KB index rebuilt successfully."
        st.rerun()


# ---------------------------------------------------------------------------
# Dashboard (formerly Advanced / Debug)
# ---------------------------------------------------------------------------

def render_advanced() -> None:
    """Render the Dashboard section."""
    st.header("Dashboard")
    st.markdown(
        "Reviewer-facing overview of observability, evaluation, cost tracking, "
        "learning signals, and workflow readiness."
    )

    from src.config import get_settings
    from src.kb.index_health import get_kb_index_health
    from src.services.observability import format_tracing_status, get_tracing_status

    settings = get_settings()
    status = get_tracing_status()
    info = format_tracing_status(status)
    ragas_report = _get_ragas_report()
    ragas_available, _ragas_reason = _check_ragas_available()
    session_usage = _get_session_usage_summary()
    learn_result = st.session_state.get("last_learn_result", {})
    quiz_result = st.session_state.get("last_quiz_gen_result", {})
    kb_health = get_kb_index_health()
    progressive_streaming = bool(st.session_state.get("last_learn_progressive_streaming", False))
    cache_bypass = bool(st.session_state.get("last_learn_force_regenerate", False))

    try:
        from src.memory.memory_service import get_user_profile_summary

        profile = get_user_profile_summary()
        mem = format_memory_transparency(profile)
    except Exception:
        profile = None
        mem = format_memory_transparency(None)

    from src.memory.feedback_service import get_feedback_summary

    fb_summary = get_feedback_summary()

    # ── Review Snapshot ───────────────────────────────────────────────────
    st.subheader("Review Snapshot")
    st.caption(
        "The most important review signals are surfaced first; detailed diagnostics "
        "remain available farther down the page."
    )
    top_cols = st.columns(4)
    top_cols[0].metric(
        "LangSmith Tracing",
        info["status_label"],
        help="Tracing configuration used for observability and workflow review.",
    )
    top_cols[1].metric(
        "RAGAs Benchmark",
        _ragas_snapshot_value(ragas_report),
        help="Cached means a saved benchmark report is available to inspect without rerunning it.",
    )
    top_cols[2].metric(
        "Session Tokens",
        f"{session_usage['total_tokens']:,}" if session_usage else "—",
        help="Session-level token usage aggregated across Learn and Quiz runs.",
    )
    top_cols[3].metric(
        "Session Cost",
        f"${session_usage['estimated_cost_usd']:.6f}" if session_usage else "—",
        help="Estimated session-level cost based on tracked token usage.",
    )

    signal_cols = st.columns(4)
    signal_cols[0].metric(
        "Memory Profile",
        "Loaded" if mem["loaded"] else "Empty",
        help="Personalization becomes richer after saved quiz results and repeated study sessions accumulate.",
    )
    signal_cols[1].metric(
        "Feedback Entries",
        str(fb_summary.get("total_count", 0)),
        help="Captured rating/comment signals from Learn and Quiz feedback widgets.",
    )
    signal_cols[2].metric(
        "Learn Workflow",
        _trace_snapshot_value(learn_result, "last_learn_trace"),
        help="Whether a Learn run is available for trace inspection.",
    )
    signal_cols[3].metric(
        "Quiz Workflow",
        _trace_snapshot_value(quiz_result, "last_quiz_trace"),
        help="Whether a Quiz run is available for trace inspection.",
    )

    st.markdown("**Project Strengths**")
    st.markdown(
        "- LangGraph workflows with reviewer-visible traces and LangSmith observability.\n"
        "- Cached RAGAs benchmark for evaluation readiness, with reruns kept manual and cost-aware.\n"
        "- Session-level token and cost tracking across Learn and Quiz.\n"
        "- Personalization signals from learning memory and feedback summaries."
    )
    st.divider()

    # ── Observability ─────────────────────────────────────────────────────
    st.subheader("Observability")
    st.caption("High-level runtime configuration for tracing, model selection, and retrieval setup.")
    api_ok = "Configured" if settings.openai_api_key else "Missing"
    obs_left, obs_right = st.columns(2)
    with obs_left:
        st.markdown(
            f"| Setting | Value |\n"
            f"|---|---|\n"
            f"| OpenAI API | {api_ok} |\n"
            f"| Model | {settings.app_default_model} |\n"
            f"| LangSmith | {info['status_label']} |\n"
            f"| Project | {info['project']} |"
        )
    with obs_right:
        st.markdown(
            f"| Setting | Value |\n"
            f"|---|---|\n"
            f"| Embedding | {settings.embedding_model} |\n"
            f"| Chunk Size / Overlap | {settings.chunk_size} / {settings.chunk_overlap} |\n"
            f"| Learn Trace Ready | {_trace_snapshot_value(learn_result, 'last_learn_trace')} |\n"
            f"| Quiz Trace Ready | {_trace_snapshot_value(quiz_result, 'last_quiz_trace')} |"
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
    st.divider()

    _display_capability_registry_section(
        kb_health=kb_health,
        tracing_info=info,
        ragas_available=ragas_available,
        memory_loaded=mem["loaded"],
        feedback_count=fb_summary.get("total_count", 0),
        progressive_streaming=progressive_streaming,
        cache_bypass=cache_bypass,
        has_api_key=bool(settings.openai_api_key),
    )
    st.divider()

    notice = st.session_state.pop("kb_rebuild_notice", None)
    if notice:
        st.success(notice)

    _display_kb_health_section()
    st.divider()

    # ── Token and Cost Tracking ───────────────────────────────────────────
    st.subheader("Token and Cost Tracking")
    _display_latest_run_contexts(learn_result, quiz_result)
    _display_session_cost_summary()
    st.divider()

    # ── Content Quality Evaluation (RAGAs) ─────────────────────────────
    _render_ragas_section()
    st.divider()

    # ── Learning Signals ──────────────────────────────────────────────────
    st.subheader("Learning Signals")
    st.caption(
        "These signals show whether personalization, saved progress, and reviewer-visible feedback are populated."
    )
    memory_col, feedback_col = st.columns(2)
    with memory_col:
        st.markdown("#### Memory and Progress")
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
                "No saved learning memory yet. Save quiz results to populate progress trends, weak areas, and suggested focus."
            )
    with feedback_col:
        st.markdown("#### Feedback Signals")
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
            st.info(
                "No feedback captured yet. Submit ratings from Learn or Quiz to populate this reviewer-facing feedback summary."
            )
    st.divider()

    # ── Workflow Readiness ────────────────────────────────────────────────
    st.subheader("Workflow Readiness")
    st.caption(
        "Latest Learn and Quiz runs stay inspectable here. Expand a workflow only when you need detailed state or raw trace entries."
    )
    learn_col, quiz_col = st.columns(2)
    with learn_col:
        _display_workflow_trace_panel(
            "Learn Workflow Trace",
            learn_result,
            trace_key="last_learn_trace",
            tokens_key="last_learn_tokens",
            empty_message="No Learn trace available yet. Generate a Learn topic or Learn Path to populate workflow diagnostics.",
        )
    with quiz_col:
        _display_workflow_trace_panel(
            "Quiz Workflow Trace",
            quiz_result,
            trace_key="last_quiz_trace",
            tokens_key="last_quiz_tokens",
            empty_message="No Quiz trace available yet. Generate and evaluate a quiz to populate workflow diagnostics.",
        )
