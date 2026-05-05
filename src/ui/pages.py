"""Streamlit UI page renderers for each app section."""

import streamlit as st

from src.schemas import DifficultyLevel, QuizQuestion, ResponseStyle, StudyGuide
from src.demo.review_examples import get_demo_by_title, get_demo_titles
from src.ui.display_helpers import (
    format_error_message,
    format_graph_state_summary,
    format_memory_transparency,
    format_source_display,
    format_sources_summary,
)


# ---------------------------------------------------------------------------
# Available topics for the Learn section
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
    st.warning(f"{err['icon']} **{err['title']}** — {err['message']}")
    st.caption(f"💡 {err['suggestion']}")


def _display_sources_section(guide: StudyGuide) -> None:
    """Render source transparency for a study guide."""
    sources = guide.sources if guide else []
    st.markdown(f"### 📚 Sources — {format_sources_summary(sources)}")

    if not sources:
        _show_friendly_error("no_sources")
        return

    for src in sources:
        info = format_source_display(src)
        label = f"📄 {info['title']} (relevance: {info['relevance_label']})"
        with st.expander(label):
            if info["metadata_items"]:
                meta_str = " · ".join(f"**{k}:** {v}" for k, v in info["metadata_items"])
                st.markdown(meta_str)
            st.markdown(info["snippet"])


def _display_memory_section(result: dict) -> None:
    """Render memory transparency for a workflow result."""
    profile = result.get("memory_profile")
    mem = format_memory_transparency(profile)

    with st.expander("🧠 Memory Profile"):
        if not mem["loaded"]:
            st.info(mem["message"])
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


def _display_debug_trace(result: dict, label: str = "Debug Trace") -> None:
    """Render workflow trace with state summary."""
    with st.expander(f"🔍 {label}"):
        # State summary fields
        fields = format_graph_state_summary(result)
        if fields:
            for f in fields:
                st.markdown(f"- **{f['label']}:** {f['value']}")
            st.markdown("---")

        trace = result.get("trace", [])
        if trace:
            for entry in trace:
                st.text(entry)
        else:
            st.info("No trace entries recorded.")

        tokens = result.get("token_usage", {})
        if tokens and tokens.get("total_tokens"):
            st.json(tokens)


# ---------------------------------------------------------------------------
# Intro / Help
# ---------------------------------------------------------------------------

def render_intro() -> None:
    """Render the Intro / Help section."""
    st.header("👋 Welcome to the AI Engineering Learning Assistant")
    st.markdown(
        """
        This app helps you study AI Engineering concepts through a guided workflow:

        1. **Learn** — Get a structured study guide on a chosen topic.
        2. **Quiz** — Test your understanding with generated questions.
        3. **Progress / Feedback** — Review your scores, weak areas, and preferences.
        4. **Advanced / Debug** — Inspect retrieval data, graph traces, and settings.

        Select a section from the sidebar to get started.
        """
    )

    # Check for missing API key and show helpful hint
    try:
        from src.config import get_settings
        settings = get_settings()
        if not settings.openai_api_key:
            _show_friendly_error("no_api_key")
        else:
            st.success("✅ OpenAI API key configured. You're ready to learn!")
    except Exception:
        st.info("🚧 This is an early version. Configure your .env file to get started.")


# ---------------------------------------------------------------------------
# Learn
# ---------------------------------------------------------------------------

def _display_study_guide(guide: StudyGuide) -> None:
    """Render a structured study guide in the Streamlit UI."""
    st.subheader(f"📘 {guide.topic}")
    st.caption(f"Difficulty: {guide.difficulty.value}")

    st.markdown("### Summary")
    st.markdown(guide.summary)

    if guide.key_concepts:
        st.markdown("### Key Concepts")
        for concept in guide.key_concepts:
            st.markdown(f"- {concept}")

    if guide.detailed_notes:
        st.markdown("### Detailed Notes")
        st.markdown(guide.detailed_notes)

    _display_sources_section(guide)


def _display_feedback_widget(context_type: str, topic: str) -> None:
    """Display a rating + comment feedback form for learn or quiz."""
    if not topic:
        return

    key_prefix = f"fb_{context_type}"
    saved_key = f"{key_prefix}_saved"

    if st.session_state.get(saved_key):
        st.success("✅ Feedback saved. Thank you!")
        return

    with st.expander(f"💬 Rate this {context_type} experience"):
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


def render_learn() -> None:
    """Render the Learn section with topic input and study guide generation."""
    st.header("📖 Learn")
    st.markdown(
        "Select a topic and generate a structured study guide powered by "
        "Agentic RAG."
    )

    # Demo examples selector
    with st.expander("💡 Try a demo example"):
        demo_titles = get_demo_titles()
        selected_demo = st.selectbox(
            "Demo topic", [""] + demo_titles, key="demo_select",
        )
        if selected_demo:
            demo = get_demo_by_title(selected_demo)
            if demo:
                st.caption(demo.description)
                st.caption(f"Features: {', '.join(demo.features_exercised)}")
                if st.button("Use this demo topic", key="btn_use_demo"):
                    st.session_state["learn_topic_custom"] = demo.topic
                    st.session_state["learn_difficulty_demo"] = demo.difficulty
                    st.session_state["learn_style_demo"] = demo.response_style

    # Use demo topic if set, else manual selection
    demo_topic = st.session_state.pop("learn_topic_custom", None)
    demo_diff = st.session_state.pop("learn_difficulty_demo", None)
    demo_style = st.session_state.pop("learn_style_demo", None)

    col1, col2, col3 = st.columns(3)
    with col1:
        if demo_topic:
            topic = st.text_input("Topic", value=demo_topic, key="learn_topic_input")
        else:
            topic = st.selectbox("Topic", LEARN_TOPICS, key="learn_topic")
    with col2:
        difficulty = st.selectbox(
            "Difficulty",
            [d.value for d in DifficultyLevel],
            index=1,
            key="learn_difficulty",
        )
    with col3:
        style = st.selectbox(
            "Response Style",
            [s.value for s in ResponseStyle],
            index=1,
            key="learn_style",
        )

    generate = st.button("🚀 Generate Study Guide", key="btn_learn")

    if generate:
        with st.spinner("Running Learn workflow…"):
            from src.graphs.learn_graph import run_learn_workflow

            result = run_learn_workflow(
                topic=topic,
                difficulty=DifficultyLevel(difficulty),
                style=ResponseStyle(style),
            )

        st.session_state["last_learn_result"] = result

        error = result.get("error")
        if error:
            st.error(f"⚠️ {error}")
        else:
            guide = result.get("study_guide")
            if guide:
                _display_study_guide(guide)
                st.session_state["last_study_guide"] = guide
            else:
                st.warning("No study guide was generated. Try a different topic.")

            st.session_state["last_learn_topic"] = topic

        # Store trace, token usage, and cost records for debug view
        st.session_state["last_learn_trace"] = result.get("trace", [])
        st.session_state["last_learn_tokens"] = result.get("token_usage", {})
        _accumulate_usage_records(result.get("usage_records", []))

        # Memory and debug transparency
        _display_memory_section(result)
        _display_debug_trace(result, "Learn Workflow Trace")

    # Feedback after study guide
    _display_feedback_widget("learn", st.session_state.get("last_learn_topic", ""))


# ---------------------------------------------------------------------------
# Quiz
# ---------------------------------------------------------------------------

def _display_quiz_questions(questions: list[QuizQuestion]) -> list[str]:
    """Display quiz questions and collect user answers via radio buttons."""
    answers: list[str] = []
    for i, q in enumerate(questions):
        st.markdown(f"**Q{i + 1}.** {q.question}")
        choice = st.radio(
            f"Select your answer for Q{i + 1}:",
            options=q.options,
            key=f"quiz_q_{i}",
            index=None,
        )
        answers.append(choice if choice else "")
        st.divider()
    return answers


def _display_quiz_results(result: dict) -> None:
    """Display evaluation results: score, feedback, weak areas, next steps."""
    score = result.get("score", 0.0)
    explanations = result.get("explanations", [])
    weak_areas = result.get("weak_areas", [])
    next_steps = result.get("suggested_next_steps", [])
    quiz_result = result.get("quiz_result")

    if quiz_result:
        st.subheader(f"📊 Score: {quiz_result.correct_count}/{quiz_result.total_questions} ({quiz_result.score_percent}%)")
        st.markdown(quiz_result.feedback)
    else:
        st.subheader(f"📊 Score: {score:.0f}%")

    if explanations:
        st.markdown("### Per-Question Feedback")
        for exp in explanations:
            st.markdown(exp)

    if weak_areas:
        st.markdown("### 🔴 Weak Areas")
        for area in weak_areas:
            st.markdown(f"- {area}")

    if next_steps:
        st.markdown("### 💡 Suggested Next Steps")
        for step in next_steps:
            st.markdown(f"- {step}")

    suggested_topics = result.get("suggested_topics", [])
    if suggested_topics:
        st.markdown("### 🎯 Suggested Topics to Study Next")
        for t in suggested_topics:
            st.markdown(f"- {t}")


def render_quiz() -> None:
    """Render the Quiz section with generation, answering, and evaluation."""
    st.header("🧠 Quiz")
    st.markdown("Test your understanding with AI-generated quiz questions.")

    col1, col2, col3 = st.columns(3)
    with col1:
        topic = st.selectbox("Topic", LEARN_TOPICS, key="quiz_topic")
    with col2:
        difficulty = st.selectbox(
            "Difficulty",
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

    # Use study guide context from session if available
    study_context = ""
    last_guide = st.session_state.get("last_study_guide")
    if last_guide and hasattr(last_guide, "topic") and last_guide.topic == topic:
        study_context = last_guide.detailed_notes or last_guide.summary or ""
        st.caption("📘 Using context from your last study guide.")

    generate = st.button("🎯 Generate Quiz", key="btn_generate_quiz")

    if generate:
        with st.spinner("Generating quiz…"):
            from src.graphs.quiz_graph import run_quiz_generation

            result = run_quiz_generation(
                topic=topic,
                difficulty=DifficultyLevel(difficulty),
                num_questions=num_questions,
                study_guide_context=study_context,
            )

        error = result.get("error")
        if error:
            st.error(f"⚠️ {error}")
            _show_friendly_error("quiz_generation_failure")
        else:
            questions = result.get("questions", [])
            if questions:
                st.session_state["quiz_questions"] = questions
                st.session_state["quiz_topic"] = topic
                st.session_state["quiz_eval_result"] = None
                val_errors = result.get("validation_errors", [])
                if val_errors:
                    st.warning("⚠️ Quiz validation warnings: " + "; ".join(val_errors))
            else:
                st.warning("No questions were generated. Try a different topic.")

        st.session_state["last_quiz_gen_result"] = result
        st.session_state["last_quiz_trace"] = result.get("trace", [])
        st.session_state["last_quiz_tokens"] = result.get("token_usage", {})
        _accumulate_usage_records(result.get("usage_records", []))

    # Display questions if available
    questions = st.session_state.get("quiz_questions", [])
    if questions:
        st.subheader(f"📝 Quiz: {st.session_state.get('quiz_topic', topic)}")
        user_answers = _display_quiz_questions(questions)

        # Warn about incomplete answers
        unanswered = sum(1 for a in user_answers if not a)
        if unanswered > 0:
            st.caption(f"⚠️ {unanswered} question(s) unanswered — they will count as incorrect.")

        submit = st.button("✅ Submit Answers", key="btn_submit_quiz")
        if submit:
            with st.spinner("Evaluating answers…"):
                from src.graphs.quiz_graph import run_quiz_evaluation

                eval_result = run_quiz_evaluation(
                    topic=st.session_state.get("quiz_topic", topic),
                    questions=questions,
                    user_answers=user_answers,
                )

            st.session_state["quiz_eval_result"] = eval_result
            st.session_state["last_quiz_trace"] = eval_result.get("trace", [])
            st.session_state["last_quiz_tokens"] = eval_result.get("token_usage", {})

    # Display evaluation results if available
    eval_result = st.session_state.get("quiz_eval_result")
    if eval_result:
        eval_error = eval_result.get("error")
        if eval_error:
            st.error(f"⚠️ {eval_error}")
        else:
            _display_quiz_results(eval_result)
            _display_hitl_save(eval_result)

        # Feedback after quiz results
        _display_feedback_widget("quiz", st.session_state.get("quiz_topic", topic))

        _display_debug_trace(eval_result, "Quiz Evaluation Trace")


# ---------------------------------------------------------------------------
# HITL — Save / Skip
# ---------------------------------------------------------------------------

def _display_hitl_save(eval_result: dict) -> None:
    """Show save/skip buttons for the memory candidate (HITL)."""
    candidate = eval_result.get("memory_candidate")
    if not candidate:
        return

    st.markdown("---")
    st.markdown("### 💾 Save this result to your learning memory?")
    st.markdown(
        f"**Topic:** {candidate.get('topic', 'N/A')} · "
        f"**Score:** {candidate.get('score', 0):.0f}%"
    )
    weak = candidate.get("weak_areas", [])
    if weak:
        st.markdown("**Weak areas:** " + ", ".join(weak))

    col_save, col_skip, _ = st.columns([1, 1, 4])
    with col_save:
        if st.button("💾 Save", key="btn_hitl_save"):
            try:
                from src.memory.memory_service import save_learning_event

                save_learning_event(
                    topic=candidate["topic"],
                    score=candidate["score"],
                    weak_areas=candidate.get("weak_areas", []),
                )
                st.session_state["hitl_saved"] = True
            except Exception:
                st.session_state["hitl_saved"] = "error"
    with col_skip:
        if st.button("⏭️ Skip", key="btn_hitl_skip"):
            st.session_state["hitl_saved"] = False

    saved = st.session_state.get("hitl_saved")
    if saved is True:
        st.success("✅ Result saved to learning memory.")
    elif saved == "error":
        _show_friendly_error("memory_save_failure")
    elif saved is False:
        st.info("Skipped — result not saved.")


# ---------------------------------------------------------------------------
# Progress / Feedback
# ---------------------------------------------------------------------------

def render_progress() -> None:
    """Render the Progress / Feedback section with learning memory data."""
    st.header("📊 Progress / Feedback")
    st.markdown("Track your studied topics, quiz scores, weak areas, and feedback.")

    from src.memory.memory_service import get_recent_topics, get_weak_areas_summary

    recent = get_recent_topics(limit=10)
    if not recent:
        _show_friendly_error("empty_progress")
    else:
        st.subheader("📋 Recent Learning Sessions")
        for evt in recent:
            weak_str = ", ".join(evt["weak_areas"]) if evt["weak_areas"] else "—"
            st.markdown(
                f"- **{evt['topic']}** — Score: {evt['score']:.0f}% · "
                f"Weak areas: {weak_str} · {evt['timestamp'][:10]}"
            )

        st.subheader("🔴 Weak Areas Summary")
        summary = get_weak_areas_summary()
        if summary:
            for area, count in sorted(summary.items(), key=lambda x: -x[1]):
                st.markdown(f"- **{area}** — appeared {count} time(s)")
        else:
            st.info("No weak areas recorded yet.")

    # Memory transparency
    st.subheader("🧠 Memory Profile")
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
            st.info(mem["message"])
    except Exception:
        st.info("Memory profile not available.")

    # Feedback section
    st.subheader("💬 Recent Feedback")
    from src.memory.feedback_service import get_recent_feedback, get_feedback_summary

    fb_entries = get_recent_feedback(limit=5)
    if fb_entries:
        for fb in fb_entries:
            stars = "⭐" * fb["rating"]
            comment = fb["comment"] if fb["comment"] else "—"
            st.markdown(
                f"- {stars} **{fb['context_type']}** / {fb['topic']} — "
                f"{comment} · {fb['timestamp'][:10]}"
            )
    else:
        st.info("No feedback recorded yet.")

    fb_summary = get_feedback_summary()
    if fb_summary.get("total_count", 0) > 0:
        st.subheader("📈 Feedback Summary")
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
        st.info("No usage data yet. Generate a study guide or quiz to see cost estimates.")
        return

    from src.services.cost_tracker import aggregate_usage

    summary = aggregate_usage(records)
    st.metric("Total Tokens", f"{summary['total_tokens']:,}")
    st.metric("Estimated Cost", f"${summary['estimated_cost_usd']:.6f}")

    st.markdown("**Breakdown by operation:**")
    for op in summary["operations"]:
        st.markdown(
            f"- **{op['operation']}** — {op['total_tokens']:,} tokens · "
            f"${op['estimated_cost_usd']:.6f}"
        )

    st.markdown(f"**Total records:** {len(records)}")


# ---------------------------------------------------------------------------
# Advanced / Debug
# ---------------------------------------------------------------------------

def render_advanced() -> None:
    """Render the Advanced / Debug section."""
    st.header("⚙️ Advanced / Debug")
    st.markdown(
        "Inspect retrieval chunks, graph execution traces, token usage, "
        "and application settings."
    )

    # --- Tracing status ---
    with st.expander("🔗 LangSmith Tracing Status"):
        from src.services.observability import format_tracing_status, get_tracing_status

        status = get_tracing_status()
        info = format_tracing_status(status)
        st.markdown(f"**Status:** {info['status_label']}")
        st.markdown(f"**Project:** {info['project']}")
        st.markdown(f"**Endpoint:** {info['endpoint']}")
        if info["has_api_key"]:
            st.markdown("**API Key:** ✅ Configured")
        elif any("API_KEY" in i for i in info.get("issues", [])):
            st.warning("⚠️ LANGCHAIN_API_KEY is missing. Tracing will not work.")
        if info["issues"]:
            for issue in info["issues"]:
                st.caption(f"ℹ️ {issue}")

    # --- Application settings ---
    with st.expander("Application Settings"):
        from src.config import get_settings

        settings = get_settings()
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

    # --- Learn workflow trace ---
    with st.expander("Last Learn Workflow Trace"):
        learn_result = st.session_state.get("last_learn_result", {})
        trace = learn_result.get("trace") or st.session_state.get("last_learn_trace", [])
        if trace or learn_result:
            fields = format_graph_state_summary(learn_result) if learn_result else []
            if fields:
                for f in fields:
                    st.markdown(f"- **{f['label']}:** {f['value']}")
                st.markdown("---")
            if trace:
                for entry in trace:
                    st.text(entry)
            else:
                st.info("No trace entries.")
        else:
            st.info("No trace available yet. Generate a study guide first.")

    # --- Learn token usage ---
    with st.expander("Last Learn Token Usage"):
        tokens = st.session_state.get("last_learn_tokens", {})
        if tokens and tokens.get("total_tokens"):
            st.json(tokens)
        else:
            st.info("No token usage data yet.")

    # --- Quiz workflow trace ---
    with st.expander("Last Quiz Workflow Trace"):
        quiz_gen = st.session_state.get("last_quiz_gen_result", {})
        trace = quiz_gen.get("trace") or st.session_state.get("last_quiz_trace", [])
        if trace or quiz_gen:
            fields = format_graph_state_summary(quiz_gen) if quiz_gen else []
            if fields:
                for f in fields:
                    st.markdown(f"- **{f['label']}:** {f['value']}")
                st.markdown("---")
            if trace:
                for entry in trace:
                    st.text(entry)
            else:
                st.info("No trace entries.")
        else:
            st.info("No quiz trace available yet. Generate a quiz first.")

    # --- Quiz token usage ---
    with st.expander("Last Quiz Token Usage"):
        tokens = st.session_state.get("last_quiz_tokens", {})
        if tokens and tokens.get("total_tokens"):
            st.json(tokens)
        else:
            st.info("No quiz token usage data yet.")

    # --- Session cost summary ---
    with st.expander("Session Cost Summary"):
        _display_session_cost_summary()

    # --- Memory profile ---
    with st.expander("🧠 Memory Profile"):
        try:
            from src.memory.memory_service import get_user_profile_summary

            profile = get_user_profile_summary()
            mem = format_memory_transparency(profile)
            if mem["loaded"]:
                st.json({
                    "recent_topics": mem["recent_topics"],
                    "recurring_weak_areas": mem["weak_areas"],
                    "average_score": mem["average_score"],
                    "suggested_focus_topics": mem["suggested_focus"],
                    "preferred_style": mem["preferred_style"],
                })
            else:
                st.info(mem["message"])
        except Exception:
            st.info("Memory profile not available.")

    # --- Feedback summary ---
    with st.expander("Feedback Summary"):
        from src.memory.feedback_service import get_feedback_summary

        fb_summary = get_feedback_summary()
        if fb_summary.get("total_count", 0) > 0:
            st.json(fb_summary)
        else:
            st.info("No feedback data yet.")
