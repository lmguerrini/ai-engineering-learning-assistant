"""Node functions for the Quiz LangGraph workflow.

Each function takes a QuizState dict and returns a partial state update.
"""

import json
from typing import Any

from loguru import logger
from openai import OpenAI

from src.config import get_settings
from src.graphs.quiz_prompts import _build_quiz_memory_context, _build_quiz_prompt  # noqa: F401 — re-export
from src.graphs.quiz_state import QuizState
from src.schemas import DifficultyLevel, QuizQuestion, QuizResult
from src.services.cache import build_cache_key, get_cached_value, set_cached_value
from src.services.cost_tracker import build_usage_record
from src.services.retry import with_retry

# Default number of questions when not specified
_DEFAULT_NUM_QUESTIONS = 5
# Required minimum number of options per question
_MIN_OPTIONS = 3


# ---------------------------------------------------------------------------
# Node: load_topic_context
# ---------------------------------------------------------------------------

def load_topic_context(state: QuizState) -> dict:
    """Ensure topic and context are available for quiz generation."""
    trace = list(state.get("trace", []))
    trace.append("load_topic_context: started")

    topic = state.get("topic", "").strip()
    if not topic:
        trace.append("load_topic_context: failed — no topic provided")
        return {"error": "Please select a topic before generating a quiz.", "trace": trace}

    context = state.get("study_guide_context", "")
    if not context:
        # Build a minimal context from the topic itself
        context = f"AI Engineering topic: {topic}"
        trace.append("load_topic_context: no study guide context, using topic as context")
    else:
        trace.append(f"load_topic_context: using study guide context ({len(context)} chars)")

    num_q = state.get("num_questions", _DEFAULT_NUM_QUESTIONS)
    difficulty = state.get("difficulty", DifficultyLevel.INTERMEDIATE)

    trace.append(f"load_topic_context: topic='{topic}', difficulty={difficulty}, num_questions={num_q}")
    return {
        "topic": topic,
        "study_guide_context": context,
        "difficulty": difficulty,
        "num_questions": num_q,
        "error": None,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Node: load_user_memory
# ---------------------------------------------------------------------------

def load_user_memory(state: QuizState) -> dict:
    """Load user memory profile and feedback summary."""
    trace = list(state.get("trace", []))
    trace.append("load_user_memory: started")

    try:
        from src.memory.memory_service import get_user_profile_summary as _get_profile

        profile = _get_profile()
    except Exception as e:
        logger.warning("Failed to load memory profile: {}", e)
        profile = {
            "recent_topics": [],
            "recurring_weak_areas": [],
            "average_score": None,
            "preferred_style": None,
            "suggested_focus_topics": [],
        }

    # Attach feedback suggestion to profile for downstream use
    try:
        from src.memory.feedback_service import get_feedback_summary

        fb = get_feedback_summary()
        if fb.get("suggestion"):
            profile["feedback_suggestion"] = fb["suggestion"]
    except Exception:
        pass

    has_data = bool(profile.get("recent_topics"))
    trace.append(
        f"load_user_memory: {'profile loaded' if has_data else 'no memory data'}"
    )
    return {"user_memory": profile, "memory_profile": profile, "trace": trace}


# ---------------------------------------------------------------------------
# Node: generate_quiz
# ---------------------------------------------------------------------------



def _build_fallback_questions(state: QuizState) -> list[QuizQuestion]:
    """Build minimal placeholder questions when the LLM fails."""
    topic = state.get("topic", "AI Engineering")
    return [
        QuizQuestion(
            question=f"What is a key concept in {topic}?",
            options=[
                "A) It is a fundamental technique",
                "B) It is unrelated to AI",
                "C) It only applies to web development",
                "D) It is deprecated",
            ],
            correct_answer="A) It is a fundamental technique",
            explanation=f"This is a placeholder question about {topic}. "
                        "Quiz generation failed — please try again.",
        )
    ]


def _parse_quiz_response(raw: str) -> list[QuizQuestion]:
    """Parse LLM JSON response into a list of QuizQuestion objects."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    data = json.loads(text)
    questions_data = data.get("questions", data if isinstance(data, list) else [])
    return [QuizQuestion(**q) for q in questions_data]


def _build_quiz_cache_key(state: QuizState) -> str:
    """Build a cache key for quiz generation based on inputs + memory hash."""
    profile = state.get("memory_profile", {})
    payload = {
        "topic": state.get("topic", ""),
        "difficulty": str(state.get("difficulty", "")),
        "num_questions": state.get("num_questions", _DEFAULT_NUM_QUESTIONS),
        "memory_hash": str(sorted(profile.items())) if profile else "",
    }
    return build_cache_key("quiz", payload)


def generate_quiz(state: QuizState) -> dict:
    """Generate quiz questions using the LLM."""
    trace = list(state.get("trace", []))
    trace.append("generate_quiz: started")
    token_usage = dict(state.get("token_usage", {}))

    # --- Check cache ---
    cache_key = _build_quiz_cache_key(state)
    cached = get_cached_value(cache_key)
    if cached is not None:
        try:
            questions = [QuizQuestion(**q) for q in cached]
            trace.append("generate_quiz: cache hit")
            return {"questions": questions, "trace": trace, "token_usage": token_usage}
        except Exception:
            pass  # invalid cache entry — continue to LLM

    settings = get_settings()
    if not settings.openai_api_key:
        trace.append("generate_quiz: no API key — using fallback questions")
        questions = _build_fallback_questions(state)
        return {"questions": questions, "trace": trace, "token_usage": token_usage}

    prompt = _build_quiz_prompt(state)

    model = settings.app_default_model
    usage_records = list(state.get("usage_records", []))

    try:
        client = OpenAI(api_key=settings.openai_api_key)

        def _llm_call() -> Any:
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )

        response = with_retry(
            callable=_llm_call,
            max_attempts=2,
            base_delay=1.0,
            handled_exceptions=(Exception,),
        )
        raw = response.choices[0].message.content or ""
        usage = response.usage
        if usage:
            token_usage["prompt_tokens"] = token_usage.get("prompt_tokens", 0) + (usage.prompt_tokens or 0)
            token_usage["completion_tokens"] = token_usage.get("completion_tokens", 0) + (usage.completion_tokens or 0)
            token_usage["total_tokens"] = token_usage.get("total_tokens", 0) + (usage.total_tokens or 0)

        usage_dict = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        }
        usage_records.append(build_usage_record(model, "quiz_generation", usage_dict))

        trace.append(f"generate_quiz: LLM returned {len(raw)} chars")
        questions = _parse_quiz_response(raw)
        trace.append(f"generate_quiz: parsed {len(questions)} questions")

        # Cache the result
        try:
            set_cached_value(cache_key, [q.model_dump() for q in questions], ttl_seconds=3600)
            trace.append("generate_quiz: cached")
        except Exception:
            pass

        return {"questions": questions, "trace": trace, "token_usage": token_usage, "usage_records": usage_records}

    except json.JSONDecodeError as e:
        logger.warning("Malformed quiz JSON output: {}", e)
        trace.append(f"generate_quiz: malformed JSON — {e}, using fallback")
        questions = _build_fallback_questions(state)
        return {"questions": questions, "trace": trace, "token_usage": token_usage, "usage_records": usage_records}

    except Exception as e:
        logger.error("Quiz LLM call failed: {}", e)
        trace.append(f"generate_quiz: LLM error — {e}, using fallback")
        questions = _build_fallback_questions(state)
        return {"questions": questions, "trace": trace, "token_usage": token_usage, "usage_records": usage_records}


# ---------------------------------------------------------------------------
# Node: validate_quiz
# ---------------------------------------------------------------------------

def validate_quiz(state: QuizState) -> dict:
    """Validate generated quiz questions meet quality requirements."""
    trace = list(state.get("trace", []))
    trace.append("validate_quiz: started")

    questions: list[QuizQuestion] = state.get("questions", [])
    num_expected = state.get("num_questions", _DEFAULT_NUM_QUESTIONS)
    errors: list[str] = []

    if len(questions) == 0:
        errors.append("No questions were generated.")
    elif len(questions) != num_expected:
        errors.append(f"Expected {num_expected} questions, got {len(questions)}.")

    for i, q in enumerate(questions):
        prefix = f"Q{i + 1}"
        if len(q.options) < _MIN_OPTIONS:
            errors.append(f"{prefix}: has {len(q.options)} options (minimum {_MIN_OPTIONS}).")
        if not q.correct_answer:
            errors.append(f"{prefix}: missing correct_answer.")
        elif q.correct_answer not in q.options:
            errors.append(f"{prefix}: correct_answer not in options.")
        if not q.explanation:
            errors.append(f"{prefix}: missing explanation.")

    passed = len(errors) == 0
    trace.append(f"validate_quiz: {'passed' if passed else 'failed'} ({len(errors)} errors)")
    return {"validation_passed": passed, "validation_errors": errors, "trace": trace}


# ---------------------------------------------------------------------------
# Node: evaluate_answers
# ---------------------------------------------------------------------------

def evaluate_answers(state: QuizState) -> dict:
    """Evaluate user answers against correct answers."""
    trace = list(state.get("trace", []))
    trace.append("evaluate_answers: started")

    questions: list[QuizQuestion] = state.get("questions", [])
    user_answers: list[str] = state.get("user_answers", [])

    if not questions:
        trace.append("evaluate_answers: no questions to evaluate")
        return {
            "per_question_correct": [],
            "score": 0.0,
            "explanations": [],
            "error": "No quiz questions available to evaluate.",
            "trace": trace,
        }

    # Pad missing answers with empty strings
    padded_answers = list(user_answers) + [""] * max(0, len(questions) - len(user_answers))

    per_correct: list[bool] = []
    explanations: list[str] = []

    for i, q in enumerate(questions):
        answer = padded_answers[i] if i < len(padded_answers) else ""
        is_correct = answer == q.correct_answer
        per_correct.append(is_correct)
        status = "✅ Correct" if is_correct else f"❌ Incorrect (correct: {q.correct_answer})"
        explanations.append(f"Q{i + 1}: {status} — {q.explanation}")

    correct_count = sum(per_correct)
    total = len(questions)
    score_pct = (correct_count / total * 100) if total > 0 else 0.0

    topic = state.get("topic", "Unknown")
    quiz_result = QuizResult(
        topic=topic,
        total_questions=total,
        correct_count=correct_count,
        score_percent=round(score_pct, 1),
        weak_areas=[],
        feedback=f"You scored {correct_count}/{total} ({score_pct:.0f}%).",
    )

    trace.append(f"evaluate_answers: {correct_count}/{total} correct ({score_pct:.0f}%)")
    return {
        "per_question_correct": per_correct,
        "score": round(score_pct, 1),
        "explanations": explanations,
        "quiz_result": quiz_result,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Helper: build suggested topics
# ---------------------------------------------------------------------------

def _build_suggested_topics(weak_areas: list[str], state: QuizState) -> list[str]:
    """Build suggested next topics from weak areas and memory profile."""
    seen: set[str] = set()
    suggested: list[str] = []

    # Add weak areas from this quiz
    for area in weak_areas:
        if area not in seen:
            seen.add(area)
            suggested.append(area)

    # Add focus topics from memory profile
    profile = state.get("memory_profile", {})
    for topic in profile.get("suggested_focus_topics", []):
        if topic not in seen:
            seen.add(topic)
            suggested.append(topic)

    # Add recurring weak areas from memory
    for area in profile.get("recurring_weak_areas", []):
        if area not in seen:
            seen.add(area)
            suggested.append(area)

    return suggested[:10]


# ---------------------------------------------------------------------------
# Node: extract_weak_areas
# ---------------------------------------------------------------------------

def extract_weak_areas(state: QuizState) -> dict:
    """Identify weak areas from incorrect answers."""
    trace = list(state.get("trace", []))
    trace.append("extract_weak_areas: started")

    questions: list[QuizQuestion] = state.get("questions", [])
    per_correct: list[bool] = state.get("per_question_correct", [])

    weak_areas: list[str] = []
    for i, (q, correct) in enumerate(zip(questions, per_correct)):
        if not correct:
            # Prefer concept field if available, otherwise fall back to question text
            area = q.concept.strip() if q.concept and q.concept.strip() else q.question[:100]
            weak_areas.append(area)

    # Suggested next steps based on performance
    score = state.get("score", 0.0)
    next_steps: list[str] = []
    if score >= 80:
        next_steps.append("Great job! Consider advancing to a harder difficulty level.")
    elif score >= 50:
        next_steps.append("Review the weak areas listed below, then try again.")
        next_steps.append("Re-read the study guide focusing on missed concepts.")
    else:
        next_steps.append("Study the topic again using the Learn section.")
        next_steps.append("Focus on understanding the core concepts before retaking the quiz.")

    # Build suggested topics from weak areas + memory profile
    suggested_topics = _build_suggested_topics(weak_areas, state)

    # Update quiz_result with weak areas
    quiz_result = state.get("quiz_result")
    if quiz_result is not None:
        quiz_result = quiz_result.model_copy(update={"weak_areas": weak_areas})

    trace.append(f"extract_weak_areas: {len(weak_areas)} weak areas, {len(suggested_topics)} suggested topics")
    return {
        "weak_areas": weak_areas,
        "suggested_next_steps": next_steps,
        "suggested_topics": suggested_topics,
        "quiz_result": quiz_result,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Node: create_memory_candidate
# ---------------------------------------------------------------------------

def create_memory_candidate(state: QuizState) -> dict:
    """Build a memory candidate dict from quiz results for HITL approval.

    The graph does NOT write to the database — the candidate is returned
    in the state so that the UI can ask the user whether to save it.
    """
    trace = list(state.get("trace", []))
    trace.append("create_memory_candidate: started")

    topic = state.get("topic", "Unknown")
    score = state.get("score", 0.0)
    weak_areas = state.get("weak_areas", [])

    candidate = {
        "topic": topic,
        "score": score,
        "weak_areas": weak_areas,
    }

    trace.append(f"create_memory_candidate: candidate ready (score={score})")
    return {"memory_candidate": candidate, "trace": trace}


# ---------------------------------------------------------------------------
# Node: return_results
# ---------------------------------------------------------------------------

def return_results(state: QuizState) -> dict:
    """Final node — ensures output fields are set."""
    trace = list(state.get("trace", []))
    trace.append("return_results: done")
    return {"trace": trace}
