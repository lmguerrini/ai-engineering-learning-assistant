"""Typed state for the Quiz graph workflow."""

from typing import Any, TypedDict

from src.schemas import DifficultyLevel, QuizQuestion, QuizResult


class QuizState(TypedDict, total=False):
    """State carried through the Quiz LangGraph workflow.

    Fields are optional (total=False) so nodes can set them incrementally.
    """

    # --- Input ---
    topic: str
    difficulty: DifficultyLevel
    num_questions: int
    study_guide_context: str

    # --- Memory (placeholder) ---
    user_memory: dict[str, Any]

    # --- Generation ---
    questions: list[QuizQuestion]
    validation_passed: bool
    validation_errors: list[str]

    # --- Evaluation ---
    user_answers: list[str]
    per_question_correct: list[bool]
    score: float
    explanations: list[str]
    weak_areas: list[str]
    suggested_next_steps: list[str]
    quiz_result: QuizResult | None

    # --- Memory candidate (HITL — not persisted inside graph) ---
    memory_candidate: dict[str, Any] | None

    # --- Output / control ---
    error: str | None
    trace: list[str]
    token_usage: dict[str, int]
    usage_records: list[dict[str, Any]]
