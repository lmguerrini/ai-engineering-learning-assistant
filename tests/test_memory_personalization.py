"""Tests for Phase 7 — Memory-based personalization."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.graphs.learn_nodes import _build_memory_context, load_user_memory as learn_load_user_memory
from src.graphs.learn_state import LearningState
from src.graphs.quiz_nodes import (
    _build_suggested_topics,
    _build_quiz_memory_context,
    extract_weak_areas,
    load_user_memory as quiz_load_user_memory,
)
from src.graphs.quiz_state import QuizState
from src.memory.memory_service import (
    get_user_profile_summary,
    save_learning_event,
)
from src.schemas import DifficultyLevel, QuizQuestion, QuizResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp_db() -> Path:
    """Return a temporary DB path for testing."""
    return Path(tempfile.mktemp(suffix=".db"))


def _seed_db(db_path: Path) -> None:
    """Insert a few learning events into a temp DB."""
    save_learning_event("AI Agents", 80.0, ["planning", "tool use"], db_path=db_path)
    save_learning_event("RAG", 40.0, ["embeddings", "chunking"], db_path=db_path)
    save_learning_event("LangGraph", 90.0, ["planning"], db_path=db_path)


# ===================================================================
# Part 1 — Memory profile summary
# ===================================================================

class TestGetUserProfileSummary:
    def test_empty_db(self):
        db = _tmp_db()
        profile = get_user_profile_summary(db_path=db)
        assert profile["recent_topics"] == []
        assert profile["recurring_weak_areas"] == []
        assert profile["average_score"] is None
        assert profile["preferred_style"] is None
        assert profile["suggested_focus_topics"] == []

    def test_with_data(self):
        db = _tmp_db()
        _seed_db(db)
        profile = get_user_profile_summary(db_path=db)

        # Recent topics (newest first, unique)
        assert "LangGraph" in profile["recent_topics"]
        assert "RAG" in profile["recent_topics"]
        assert "AI Agents" in profile["recent_topics"]

        # Recurring weak areas: "planning" appears 2x
        assert "planning" in profile["recurring_weak_areas"]

        # Average score: (80 + 40 + 90) / 3 = 70.0
        assert profile["average_score"] == 70.0

        # Suggested focus topics: RAG has score 40 < avg 70
        assert "RAG" in profile["suggested_focus_topics"]

    def test_recurring_weak_areas_threshold(self):
        """Only areas appearing 2+ times should be recurring."""
        db = _tmp_db()
        save_learning_event("Topic A", 50.0, ["area_once"], db_path=db)
        save_learning_event("Topic B", 50.0, ["area_twice"], db_path=db)
        save_learning_event("Topic C", 50.0, ["area_twice"], db_path=db)
        profile = get_user_profile_summary(db_path=db)
        assert "area_twice" in profile["recurring_weak_areas"]
        assert "area_once" not in profile["recurring_weak_areas"]

    def test_suggested_focus_includes_recurring_weak_areas(self):
        db = _tmp_db()
        _seed_db(db)
        profile = get_user_profile_summary(db_path=db)
        # "planning" is a recurring weak area and should appear in suggested focus
        assert "planning" in profile["suggested_focus_topics"]


# ===================================================================
# Part 2 — Learn graph memory injection
# ===================================================================

class TestLearnMemoryContext:
    def test_empty_profile_returns_empty(self):
        state: LearningState = {"memory_profile": {}, "trace": []}
        assert _build_memory_context(state) == ""

    def test_no_recent_topics_returns_empty(self):
        state: LearningState = {
            "memory_profile": {"recent_topics": [], "recurring_weak_areas": []},
            "trace": [],
        }
        assert _build_memory_context(state) == ""

    def test_with_weak_areas(self):
        state: LearningState = {
            "memory_profile": {
                "recent_topics": ["RAG"],
                "recurring_weak_areas": ["embeddings", "chunking"],
                "average_score": 65.0,
            },
            "trace": [],
        }
        ctx = _build_memory_context(state)
        assert "embeddings" in ctx
        assert "chunking" in ctx
        assert "Recently studied" in ctx

    def test_low_score_hint(self):
        state: LearningState = {
            "memory_profile": {
                "recent_topics": ["RAG"],
                "recurring_weak_areas": [],
                "average_score": 30.0,
            },
            "trace": [],
        }
        ctx = _build_memory_context(state)
        assert "simpler" in ctx.lower() or "foundational" in ctx.lower()

    def test_high_score_hint(self):
        state: LearningState = {
            "memory_profile": {
                "recent_topics": ["RAG"],
                "recurring_weak_areas": [],
                "average_score": 90.0,
            },
            "trace": [],
        }
        ctx = _build_memory_context(state)
        assert "advanced" in ctx.lower() or "nuance" in ctx.lower()


class TestLearnLoadUserMemory:
    @patch("src.memory.memory_service.get_user_profile_summary")
    def test_attaches_profile_to_state(self, mock_profile):
        mock_profile.return_value = {
            "recent_topics": ["RAG"],
            "recurring_weak_areas": ["embeddings"],
            "average_score": 65.0,
            "preferred_style": None,
            "suggested_focus_topics": [],
        }
        state: LearningState = {"topic": "AI Agents", "trace": []}
        result = learn_load_user_memory(state)
        assert result["memory_profile"]["recent_topics"] == ["RAG"]
        assert result["user_memory"] == result["memory_profile"]
        assert any("load_user_memory" in t for t in result["trace"])

    @patch("src.memory.memory_service.get_user_profile_summary", side_effect=Exception("boom"))
    def test_graceful_fallback(self, mock_profile):
        state: LearningState = {"topic": "AI Agents", "trace": []}
        result = learn_load_user_memory(state)
        assert result["memory_profile"]["recent_topics"] == []
        assert any("no memory data" in t for t in result["trace"])


# ===================================================================
# Part 3 — Quiz graph memory injection
# ===================================================================

class TestQuizMemoryContext:
    def test_empty_profile_returns_empty(self):
        state: QuizState = {"memory_profile": {}, "trace": []}
        assert _build_quiz_memory_context(state) == ""

    def test_with_weak_areas(self):
        state: QuizState = {
            "memory_profile": {
                "recent_topics": ["RAG"],
                "recurring_weak_areas": ["embeddings"],
                "average_score": 65.0,
            },
            "trace": [],
        }
        ctx = _build_quiz_memory_context(state)
        assert "embeddings" in ctx
        assert "reinforce" in ctx.lower()

    def test_low_score_quiz_hint(self):
        state: QuizState = {
            "memory_profile": {
                "recent_topics": ["RAG"],
                "recurring_weak_areas": [],
                "average_score": 30.0,
            },
            "trace": [],
        }
        ctx = _build_quiz_memory_context(state)
        assert "foundational" in ctx.lower()

    def test_high_score_quiz_hint(self):
        state: QuizState = {
            "memory_profile": {
                "recent_topics": ["RAG"],
                "recurring_weak_areas": [],
                "average_score": 90.0,
            },
            "trace": [],
        }
        ctx = _build_quiz_memory_context(state)
        assert "challenging" in ctx.lower()


class TestQuizLoadUserMemory:
    @patch("src.memory.memory_service.get_user_profile_summary")
    def test_attaches_profile(self, mock_profile):
        mock_profile.return_value = {
            "recent_topics": ["AI Agents"],
            "recurring_weak_areas": [],
            "average_score": 80.0,
            "preferred_style": None,
            "suggested_focus_topics": [],
        }
        state: QuizState = {"trace": []}
        result = quiz_load_user_memory(state)
        assert result["memory_profile"]["average_score"] == 80.0
        assert any("profile loaded" in t for t in result["trace"])


# ===================================================================
# Part 4 — Suggested topics
# ===================================================================

class TestBuildSuggestedTopics:
    def test_from_weak_areas_only(self):
        state: QuizState = {"trace": []}
        topics = _build_suggested_topics(["planning", "embeddings"], state)
        assert "AI Agents" in topics
        assert "ReAct Pattern" in topics
        assert "Agentic RAG" in topics

    def test_from_memory_profile(self):
        state: QuizState = {
            "memory_profile": {
                "suggested_focus_topics": ["RAG"],
                "recurring_weak_areas": ["chunking"],
            },
            "trace": [],
        }
        topics = _build_suggested_topics([], state)
        assert "Agentic RAG" in topics
        assert "Building Applications with LangChain, RAGs, and Streamlit" in topics

    def test_deduplication(self):
        state: QuizState = {
            "memory_profile": {
                "suggested_focus_topics": ["planning"],
                "recurring_weak_areas": ["planning"],
            },
            "trace": [],
        }
        topics = _build_suggested_topics(["planning"], state)
        assert topics.count("AI Agents") == 1

    def test_max_10_topics(self):
        state: QuizState = {"trace": []}
        many = [f"area_{i}" for i in range(15)]
        topics = _build_suggested_topics(many, state)
        assert len(topics) <= 10

    def test_empty_everything(self):
        state: QuizState = {"trace": []}
        topics = _build_suggested_topics([], state)
        assert topics == []


class TestExtractWeakAreasWithSuggestedTopics:
    def test_suggested_topics_in_result(self):
        questions = [
            QuizQuestion(
                question="Q1?",
                options=["A", "B", "C", "D"],
                correct_answer="A",
                explanation="E1",
                concept="planning",
            ),
        ]
        state: QuizState = {
            "questions": questions,
            "per_question_correct": [False],
            "score": 0.0,
            "quiz_result": QuizResult(topic="AI", total_questions=1, correct_count=0, score_percent=0.0),
            "trace": [],
        }
        result = extract_weak_areas(state)
        assert "suggested_topics" in result
        assert "AI Agents" in result["suggested_topics"]


# ===================================================================
# Part 5 — No-memory fallback behavior
# ===================================================================

class TestNoMemoryFallback:
    def test_learn_memory_context_without_profile(self):
        """When no memory_profile key exists, returns empty string."""
        state: LearningState = {"trace": []}
        assert _build_memory_context(state) == ""

    def test_quiz_memory_context_without_profile(self):
        """When no memory_profile key exists, returns empty string."""
        state: QuizState = {"trace": []}
        assert _build_quiz_memory_context(state) == ""

    def test_suggested_topics_without_profile(self):
        """Without memory_profile, suggested topics come only from weak areas."""
        state: QuizState = {"trace": []}
        topics = _build_suggested_topics(["concept_a"], state)
        assert topics == ["concept_a"]
