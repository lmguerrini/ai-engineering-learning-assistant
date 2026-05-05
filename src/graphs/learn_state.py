"""Typed state for the Learn graph workflow."""

from typing import Any, TypedDict

from src.kb.loader import Document
from src.schemas import DifficultyLevel, ResponseStyle, StudyGuide


class LearningState(TypedDict, total=False):
    """State carried through the Learn LangGraph workflow.

    Fields are optional (total=False) so nodes can set them incrementally.
    """

    # --- Input ---
    topic: str
    difficulty: DifficultyLevel
    style: ResponseStyle

    # --- Memory (placeholder) ---
    user_memory: dict[str, Any]

    # --- Retrieval ---
    query: str
    retrieved_docs: list[Document]
    source_quality_ok: bool
    query_refined: bool
    attempts: int

    # --- Generation ---
    study_guide: StudyGuide | None

    # --- Quality / output ---
    quality_passed: bool
    error: str | None
    trace: list[str]

    # --- Token / cost tracking ---
    token_usage: dict[str, int]
    usage_records: list[dict[str, Any]]
