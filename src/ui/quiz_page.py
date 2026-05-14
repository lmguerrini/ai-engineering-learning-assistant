"""Streamlit Quiz page renderer."""

import streamlit as st

from src.schemas import DifficultyLevel, QuizQuestion
from src.ui.shared import (
    LEARN_TOPICS,
    _accumulate_usage_records,
    _display_debug_trace,
    _display_feedback_widget,
    _show_friendly_error,
)


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
    quiz_result = result.get("quiz_result")
    score = result.get("score", 0)
    total = result.get("total_questions")
    correct = result.get("correct_count")

    if quiz_result is not None:
        if total is None:
            total = getattr(quiz_result, "total_questions", 0)
        if correct is None:
            correct = getattr(quiz_result, "correct_count", 0)

    total = total or 0
    correct = correct or 0

    st.markdown(f"### Score: {score:.0f}% ({correct}/{total})")

    per_question = result.get("per_question_feedback", [])
    if per_question:
        st.markdown("### Answer Review")
    for fb in per_question:
        is_correct = fb.get("correct", False)
        icon = "✅" if is_correct else "❌"
        st.markdown(
            f"**Q{fb.get('question_number', '?')}. {icon}** {fb.get('question', '')}"
        )
        selected_answer = fb.get("selected_answer") or "No answer selected"
        if is_correct:
            st.success(f"Selected answer: {selected_answer}")
        else:
            st.error(f"Selected answer: {selected_answer}")
            st.success(f"Correct answer: {fb.get('correct_answer', '')}")
        if fb.get("explanation"):
            st.caption(fb["explanation"])

    weak = result.get("weak_areas", [])
    if weak:
        st.markdown("### Weak Areas")
        for area in weak:
            st.markdown(f"- {area}")

    next_steps = result.get("next_steps") or result.get("suggested_next_steps", [])
    if next_steps:
        st.markdown("### Suggested Next Steps")
        for step in next_steps:
            st.markdown(f"- {step}")

    suggested_topics = result.get("suggested_topics", [])
    if suggested_topics:
        st.markdown("### Suggested Topics to Study Next")
        for t in suggested_topics:
            st.markdown(f"- {t}")


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
                        metadata={
                            "source": "quiz_evaluation",
                            "difficulty": st.session_state.get("quiz_difficulty", "").capitalize(),
                        },
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


def render_quiz() -> None:
    """Render the Quiz section with generation, answering, and evaluation."""
    st.header("Quiz")
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
    regenerate_quiz = st.checkbox(
        "Regenerate quiz (bypass cache)",
        value=False,
        key="quiz_force_regenerate",
    )

    study_context = ""
    last_guide = st.session_state.get("last_study_guide")
    if last_guide and hasattr(last_guide, "topic") and last_guide.topic == topic:
        study_context = last_guide.detailed_notes or last_guide.summary or ""
        st.caption("Using context from your last Learn session.")

    generate = st.button("Generate Quiz", key="btn_generate_quiz")

    if generate:
        with st.spinner("Generating quiz..."):
            from src.graphs.quiz_graph import run_quiz_generation

            result = run_quiz_generation(
                topic=topic,
                difficulty=DifficultyLevel(difficulty),
                num_questions=num_questions,
                study_guide_context=study_context,
                force_regenerate=regenerate_quiz,
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

        quiz_topic = st.session_state.get("quiz_selected_topic", topic)
        quiz_difficulty = st.session_state.get("quiz_difficulty", difficulty).capitalize()
        _display_feedback_widget(
            "quiz",
            quiz_topic,
            metadata={
                "difficulty": quiz_difficulty,
                "context_title": quiz_topic,
            },
            result_signature=f"Quiz | {quiz_topic} | {quiz_difficulty}",
        )
        _display_debug_trace(eval_result, "Quiz Evaluation Trace")
