"""Quiz prompt-building helpers.

Pure functions that compose prompts and memory context for quiz generation.
Extracted from quiz_nodes.py for maintainability.
"""

from src.graphs.quiz_state import QuizState
from src.schemas import DifficultyLevel

# Default number of questions when not specified
_DEFAULT_NUM_QUESTIONS = 5


def _build_quiz_memory_context(state: QuizState) -> str:
    """Build personalization context from memory profile for quiz generation."""
    profile = state.get("memory_profile", {})
    if not profile or not profile.get("recent_topics"):
        return ""

    parts: list[str] = []
    weak = profile.get("recurring_weak_areas", [])
    if weak:
        parts.append(f"The learner has recurring weak areas in: {', '.join(weak[:5])}.")
        parts.append("Bias some questions toward these concepts to help reinforce them.")

    avg = profile.get("average_score")
    if avg is not None:
        if avg < 50:
            parts.append("The learner's average score is low — include more foundational questions.")
        elif avg >= 80:
            parts.append("The learner's average score is high — include slightly more challenging questions.")

    fb_suggestion = profile.get("feedback_suggestion")
    if fb_suggestion == "simplify":
        parts.append("User feedback indicates questions should be simpler and clearer.")
    elif fb_suggestion == "increase_difficulty":
        parts.append("User feedback indicates questions could be more challenging.")

    return "\n".join(parts)


def _build_quiz_prompt(state: QuizState) -> str:
    """Build the LLM prompt for quiz generation."""
    topic = state.get("topic", "")
    difficulty = state.get("difficulty", DifficultyLevel.INTERMEDIATE)
    num_q = state.get("num_questions", _DEFAULT_NUM_QUESTIONS)
    context = state.get("study_guide_context", "")

    memory_context = _build_quiz_memory_context(state)
    personalization = ""
    if memory_context:
        personalization = f"\nPersonalization context:\n{memory_context}\n"

    return (
        f"You are an AI Engineering quiz generator.\n\n"
        f"Generate a multiple-choice quiz on the topic '{topic}' "
        f"at {difficulty.value} level.\n\n"
        f"Context:\n{context}\n"
        f"{personalization}\n"
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
