"""Streamlit UI page renderers for each app section."""

import streamlit as st

from src.schemas import DifficultyLevel, QuizQuestion, StudyGuide
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
# Quiz
# ---------------------------------------------------------------------------

def _display_quiz_questions(questions: list[QuizQuestion]) -> list[str]:
    """Display quiz questions and collect user answers via radio buttons."""
    answers: list[str] = []
    for i, q in enumerate(questions):
        st.markdown(f"**Q{i + 1}.** {q.question}")
        answer = st.radio(
            f"Select answer for Q{i + 1}",
            q.options,
            index=None,
            key=f"quiz_answer_{i}",
            label_visibility="collapsed",
        )
        answers.append(answer or "")
    return answers


def _display_quiz_results(result: dict) -> None:
    """Display quiz evaluation results with score and feedback."""
    score = result.get("score", 0)
    total = result.get("total_questions", 0)
    correct = result.get("correct_count", 0)

    st.markdown(f"### Score: {score:.0f}% ({correct}/{total})")

    per_question = result.get("per_question_feedback", [])
    for fb in per_question:
        is_correct = fb.get("correct", False)
        icon = "Correct" if is_correct else "Incorrect"
        st.markdown(f"**Q{fb.get('question_number', '?')}.** {icon}")
        if fb.get("explanation"):
            st.caption(fb["explanation"])

    weak = result.get("weak_areas", [])
    if weak:
        st.markdown("### Weak Areas")
        for area in weak:
            st.markdown(f"- {area}")

    next_steps = result.get("next_steps", [])
    if next_steps:
        st.markdown("### Suggested Next Steps")
        for step in next_steps:
            st.markdown(f"- {step}")

    suggested_topics = result.get("suggested_topics", [])
    if suggested_topics:
        st.markdown("### Suggested Topics to Study Next")
        for t in suggested_topics:
            st.markdown(f"- {t}")


def render_quiz() -> None:
    """Render the Quiz section with generation, answering, and evaluation."""
    st.header("Quiz")
    st.markdown("Test your understanding with AI-generated quiz questions.")

    col1, col2, col3 = st.columns(3)
    with col1:
        topic = st.selectbox("Topic", LEARN_TOPICS, key="quiz_topic")
    with col2:
        difficulty = st.selectbox(
            "Learn Path",
            [d.value for d in DifficultyLevel],
            index=1,
            key="quiz_difficulty",
        )
    with col3:
        num_questions = st.number_input(
            "Number of Questions",
            min_value=1,
            max_value=10,
            value=5,
            key="quiz_num_q",
        )

    study_context = ""
    last_guide = st.session_state.get("last_study_guide")
    if last_guide and hasattr(last_guide, "topic") and last_guide.topic == topic:
        study_context = last_guide.detailed_notes or last_guide.summary or ""
        st.caption("Using context from your last Learn Path.")

    generate = st.button("Generate Quiz", key="btn_generate_quiz")

    if generate:
        with st.spinner("Generating quiz..."):
            from src.graphs.quiz_graph import run_quiz_generation

            result = run_quiz_generation(
                topic=topic,
                difficulty=DifficultyLevel(difficulty),
                num_questions=num_questions,
                study_guide_context=study_context,
            )

        error = result.get("error")
        if error:
            st.error(error)
            _show_friendly_error("quiz_generation_failure")
        else:
            questions = result.get("questions", [])
            if questions:
                st.session_state["quiz_questions"] = questions
                st.session_state["quiz_selected_topic"] = topic
                st.session_state["quiz_eval_result"] = None
                val_errors = result.get("validation_errors", [])
                if val_errors:
                    st.warning("Quiz validation warnings: " + "; ".join(val_errors))
            else:
                st.warning("No questions were generated. Try a different topic.")

        st.session_state["last_quiz_gen_result"] = result
        st.session_state["last_quiz_trace"] = result.get("trace", [])
        st.session_state["last_quiz_tokens"] = result.get("token_usage", {})
        _accumulate_usage_records(result.get("usage_records", []))

    questions = st.session_state.get("quiz_questions", [])
    if questions:
        st.subheader(f"Quiz: {st.session_state.get('quiz_selected_topic', topic)}")
        user_answers = _display_quiz_questions(questions)

        unanswered = sum(1 for a in user_answers if not a)
        if unanswered > 0:
            st.caption(f"{unanswered} question(s) unanswered — they will count as incorrect.")

        submit = st.button("Submit Answers", key="btn_submit_quiz")
        if submit:
            with st.spinner("Evaluating answers..."):
                from src.graphs.quiz_graph import run_quiz_evaluation

                eval_result = run_quiz_evaluation(
                    topic=st.session_state.get("quiz_selected_topic", topic),
                    questions=questions,
                    user_answers=user_answers,
                )

            st.session_state["quiz_eval_result"] = eval_result
            st.session_state["last_quiz_trace"] = eval_result.get("trace", [])
            st.session_state["last_quiz_tokens"] = eval_result.get("token_usage", {})

    eval_result = st.session_state.get("quiz_eval_result")
    if eval_result:
        eval_error = eval_result.get("error")
        if eval_error:
            st.error(eval_error)
        else:
            _display_quiz_results(eval_result)
            _display_hitl_save(eval_result)

        _display_feedback_widget("quiz", st.session_state.get("quiz_selected_topic", topic))
        _display_debug_trace(eval_result, "Quiz Evaluation Trace")


# ---------------------------------------------------------------------------
# HITL — Save / Skip
# ---------------------------------------------------------------------------

def _display_hitl_save(eval_result: dict) -> None:
    """Display save/skip buttons for HITL memory persistence."""
    memory_candidate = eval_result.get("memory_candidate")
    if not memory_candidate:
        return

    save_key = "hitl_saved"
    saved = st.session_state.get(save_key)

    if saved is None:
        st.markdown("---")
        st.markdown("**Save this result to your learning memory?**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save", key="btn_hitl_save"):
                try:
                    from src.memory.memory_service import save_learning_event

                    save_learning_event(
                        topic=memory_candidate.get("topic", ""),
                        score=memory_candidate.get("score", 0),
                        weak_areas=memory_candidate.get("weak_areas", []),
                        metadata={"source": "quiz_evaluation"},
                    )
                    st.session_state[save_key] = True
                except Exception:
                    st.session_state[save_key] = "error"
                st.rerun()
        with col2:
            if st.button("Skip", key="btn_hitl_skip"):
                st.session_state[save_key] = False
                st.rerun()

    if saved is True:
        st.success("Result saved to learning memory.")
    elif saved == "error":
        _show_friendly_error("memory_save_failure")
    elif saved is False:
        st.info("Skipped — result not saved.")


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
