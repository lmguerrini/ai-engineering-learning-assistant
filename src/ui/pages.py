"""Streamlit UI page renderers for each app section."""

import streamlit as st

from src.schemas import DifficultyLevel, QuizQuestion, ResponseStyle, StudyGuide


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
    st.info("🚧 This is an early version. Memory and progress features are coming soon.")


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

    if guide.sources:
        st.markdown("### Sources Used")
        for src in guide.sources:
            with st.expander(f"📄 {src.title} (relevance: {src.relevance_score:.1f})"):
                st.markdown(src.content_snippet if src.content_snippet else "_No snippet available._")


def render_learn() -> None:
    """Render the Learn section with topic input and study guide generation."""
    st.header("📖 Learn")
    st.markdown(
        "Select a topic and generate a structured study guide powered by "
        "Agentic RAG."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
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

        error = result.get("error")
        if error:
            st.error(f"⚠️ {error}")
        else:
            guide = result.get("study_guide")
            if guide:
                _display_study_guide(guide)
            else:
                st.warning("No study guide was generated. Try a different topic.")

        # Store trace and token usage for debug view
        st.session_state["last_learn_trace"] = result.get("trace", [])
        st.session_state["last_learn_tokens"] = result.get("token_usage", {})

        # Show debug trace inline when requested
        with st.expander("🔍 Debug Trace"):
            for entry in result.get("trace", []):
                st.text(entry)
            tokens = result.get("token_usage", {})
            if tokens:
                st.json(tokens)


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

        st.session_state["last_quiz_trace"] = result.get("trace", [])
        st.session_state["last_quiz_tokens"] = result.get("token_usage", {})

    # Display questions if available
    questions = st.session_state.get("quiz_questions", [])
    if questions:
        st.subheader(f"📝 Quiz: {st.session_state.get('quiz_topic', topic)}")
        user_answers = _display_quiz_questions(questions)

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

        with st.expander("🔍 Debug Trace"):
            for entry in eval_result.get("trace", []):
                st.text(entry)


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
            from src.memory.memory_service import save_learning_event

            save_learning_event(
                topic=candidate["topic"],
                score=candidate["score"],
                weak_areas=candidate.get("weak_areas", []),
            )
            st.session_state["hitl_saved"] = True
    with col_skip:
        if st.button("⏭️ Skip", key="btn_hitl_skip"):
            st.session_state["hitl_saved"] = False

    saved = st.session_state.get("hitl_saved")
    if saved is True:
        st.success("✅ Result saved to learning memory.")
    elif saved is False:
        st.info("Skipped — result not saved.")


# ---------------------------------------------------------------------------
# Progress / Feedback
# ---------------------------------------------------------------------------

def render_progress() -> None:
    """Render the Progress / Feedback section with learning memory data."""
    st.header("📊 Progress / Feedback")
    st.markdown("Track your studied topics, quiz scores, and weak areas.")

    from src.memory.memory_service import get_recent_topics, get_weak_areas_summary

    recent = get_recent_topics(limit=10)
    if not recent:
        st.info("No progress data yet. Complete a quiz and save your result.")
        return

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

    with st.expander("Application Settings"):
        from src.config import get_settings

        settings = get_settings()
        st.json({
            "default_model": settings.app_default_model,
            "log_level": settings.app_log_level,
            "langsmith_tracing": settings.langchain_tracing_v2,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        })

    with st.expander("Last Learn Workflow Trace"):
        trace = st.session_state.get("last_learn_trace", [])
        if trace:
            for entry in trace:
                st.text(entry)
        else:
            st.info("No trace available yet. Generate a study guide first.")

    with st.expander("Last Learn Token Usage"):
        tokens = st.session_state.get("last_learn_tokens", {})
        if tokens:
            st.json(tokens)
        else:
            st.info("No token usage data yet.")

    with st.expander("Last Quiz Workflow Trace"):
        trace = st.session_state.get("last_quiz_trace", [])
        if trace:
            for entry in trace:
                st.text(entry)
        else:
            st.info("No quiz trace available yet. Generate a quiz first.")

    with st.expander("Last Quiz Token Usage"):
        tokens = st.session_state.get("last_quiz_tokens", {})
        if tokens:
            st.json(tokens)
        else:
            st.info("No quiz token usage data yet.")
