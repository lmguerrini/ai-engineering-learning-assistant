"""Node functions for the Quiz LangGraph workflow.

Each function takes a QuizState dict and returns a partial state update.
"""

import json

from loguru import logger
from openai import OpenAI

from src.config import get_settings
from src.graphs.quiz_state import QuizState
from src.schemas import DifficultyLevel, QuizQuestion, QuizResult

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
# Node: load_user_memory_placeholder
# ---------------------------------------------------------------------------

def load_user_memory_placeholder(state: QuizState) -> dict:
    """Placeholder for loading user memory (implemented in Phase 5)."""
    trace = list(state.get("trace", []))
    trace.append("load_user_memory_placeholder: no memory loaded (placeholder)")
    return {"user_memory": {}, "trace": trace}


# ---------------------------------------------------------------------------
# Node: generate_quiz
# ---------------------------------------------------------------------------

def _build_quiz_prompt(state: QuizState) -> str:
    """Build the LLM prompt for quiz generation."""
    topic = state.get("topic", "")
    difficulty = state.get("difficulty", DifficultyLevel.INTERMEDIATE)
    num_q = state.get("num_questions", _DEFAULT_NUM_QUESTIONS)
    context = state.get("study_guide_context", "")

    return (
        f"You are an AI Engineering quiz generator.\n\n"
        f"Generate a multiple-choice quiz on the topic '{topic}' "
        f"at {difficulty.value} level.\n\n"
        f"Context:\n{context}\n\n"
        f"Requirements:\n"
        f"- Generate exactly {num_q} questions.\n"
        f"- Each question must have exactly 4 options (A, B, C, D).\n"
        f"- Each question must have exactly one correct answer.\n"
        f"- Each question must include an explanation for the correct answer.\n\n"
        f"Respond with valid JSON matching this schema:\n"
        f'{{\n'
        f'  "questions": [\n'
        f'    {{\n'
        f'      "question": "What is ...?",\n'
        f'      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
        f'      "correct_answer": "A) ...",\n'
        f'      "explanation": "Because ..."\n'
        f'    }}\n'
        f'  ]\n'
        f'}}\n\n'
        f"Return ONLY the JSON object, no extra text."
    )


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


def generate_quiz(state: QuizState) -> dict:
    """Generate quiz questions using the LLM."""
    trace = list(state.get("trace", []))
    trace.append("generate_quiz: started")
    token_usage = dict(state.get("token_usage", {}))

    settings = get_settings()
    if not settings.openai_api_key:
        trace.append("generate_quiz: no API key — using fallback questions")
        questions = _build_fallback_questions(state)
        return {"questions": questions, "trace": trace, "token_usage": token_usage}

    prompt = _build_quiz_prompt(state)

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.app_default_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        raw = response.choices[0].message.content or ""
        usage = response.usage
        if usage:
            token_usage["prompt_tokens"] = token_usage.get("prompt_tokens", 0) + (usage.prompt_tokens or 0)
            token_usage["completion_tokens"] = token_usage.get("completion_tokens", 0) + (usage.completion_tokens or 0)
            token_usage["total_tokens"] = token_usage.get("total_tokens", 0) + (usage.total_tokens or 0)

        trace.append(f"generate_quiz: LLM returned {len(raw)} chars")
        questions = _parse_quiz_response(raw)
        trace.append(f"generate_quiz: parsed {len(questions)} questions")
        return {"questions": questions, "trace": trace, "token_usage": token_usage}

    except json.JSONDecodeError as e:
        logger.warning("Malformed quiz JSON output: {}", e)
        trace.append(f"generate_quiz: malformed JSON — {e}, using fallback")
        questions = _build_fallback_questions(state)
        return {"questions": questions, "trace": trace, "token_usage": token_usage}

    except Exception as e:
        logger.error("Quiz LLM call failed: {}", e)
        trace.append(f"generate_quiz: LLM error — {e}, using fallback")
        questions = _build_fallback_questions(state)
        return {"questions": questions, "trace": trace, "token_usage": token_usage}


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

    # Update quiz_result with weak areas
    quiz_result = state.get("quiz_result")
    if quiz_result is not None:
        quiz_result = quiz_result.model_copy(update={"weak_areas": weak_areas})

    trace.append(f"extract_weak_areas: {len(weak_areas)} weak areas found")
    return {
        "weak_areas": weak_areas,
        "suggested_next_steps": next_steps,
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
