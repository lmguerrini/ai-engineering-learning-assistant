"""Tests for the feedback service and personalization integration."""

from pathlib import Path

import pytest

from src.memory.feedback_service import (
    _derive_suggestion,
    delete_feedback,
    get_feedback_summary,
    get_recent_feedback,
    has_feedback_for_result,
    save_feedback,
)


@pytest.fixture()
def fb_db(tmp_path: Path) -> Path:
    return tmp_path / "test_feedback.db"


# ---------------------------------------------------------------------------
# save / retrieve
# ---------------------------------------------------------------------------

class TestSaveRetrieve:
    def test_save_and_get_recent(self, fb_db: Path):
        save_feedback("learn", "AI Agents", 5, "Great!", db_path=fb_db)
        save_feedback("quiz", "LangGraph", 3, "", db_path=fb_db)

        recent = get_recent_feedback(limit=10, db_path=fb_db)
        assert len(recent) == 2
        assert recent[0]["topic"] == "LangGraph"  # newest first
        assert recent[1]["topic"] == "AI Agents"

    def test_save_returns_row_id(self, fb_db: Path):
        row_id = save_feedback("learn", "Topic", 4, db_path=fb_db)
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_rating_clamped(self, fb_db: Path):
        save_feedback("learn", "T", 0, db_path=fb_db)   # clamped to 1
        save_feedback("learn", "T", 10, db_path=fb_db)  # clamped to 5

        recent = get_recent_feedback(limit=10, db_path=fb_db)
        ratings = [r["rating"] for r in recent]
        assert 1 in ratings
        assert 5 in ratings

    def test_limit(self, fb_db: Path):
        for i in range(5):
            save_feedback("learn", f"T{i}", 3, db_path=fb_db)
        recent = get_recent_feedback(limit=2, db_path=fb_db)
        assert len(recent) == 2

    def test_empty_db(self, fb_db: Path):
        assert get_recent_feedback(db_path=fb_db) == []

    def test_has_feedback_for_result_matches_result_signature(self, fb_db: Path):
        save_feedback(
            "learn",
            "AI Agents",
            4,
            metadata={
                "learning_mode": "Topic",
                "learning_depth": "Deep Study",
                "context_title": "AI Agents",
                "result_signature": "Topic | AI Agents | Deep Study",
            },
            db_path=fb_db,
        )

        assert has_feedback_for_result(
            "learn",
            "AI Agents",
            metadata={"result_signature": "Topic | AI Agents | Deep Study"},
            db_path=fb_db,
        ) is True
        assert has_feedback_for_result(
            "learn",
            "AI Agents",
            metadata={"result_signature": "Topic | AI Agents | Summary"},
            db_path=fb_db,
        ) is False

    def test_has_feedback_for_result_falls_back_to_metadata_fields(self, fb_db: Path):
        save_feedback(
            "learn",
            "Foundations of LLM Application Development",
            5,
            metadata={
                "learning_mode": "Learn Path",
                "learning_depth": "Summary",
                "difficulty": "Beginner",
                "context_title": "Foundations of LLM Application Development",
            },
            db_path=fb_db,
        )

        assert has_feedback_for_result(
            "learn",
            "Foundations of LLM Application Development",
            metadata={
                "learning_mode": "Learn Path",
                "learning_depth": "Summary",
                "difficulty": "Beginner",
                "context_title": "Foundations of LLM Application Development",
            },
            db_path=fb_db,
        ) is True

    def test_delete_feedback_removes_row(self, fb_db: Path):
        row_id = save_feedback("learn", "AI Agents", 4, "Helpful", db_path=fb_db)
        assert delete_feedback(row_id, db_path=fb_db) is True
        assert get_recent_feedback(db_path=fb_db) == []


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestFeedbackSummary:
    def test_empty_summary(self, fb_db: Path):
        summary = get_feedback_summary(db_path=fb_db)
        assert summary["average_rating"] is None
        assert summary["total_count"] == 0
        assert summary["suggestion"] is None

    def test_basic_summary(self, fb_db: Path):
        save_feedback("learn", "A", 5, db_path=fb_db)
        save_feedback("quiz", "B", 3, db_path=fb_db)

        summary = get_feedback_summary(db_path=fb_db)
        assert summary["average_rating"] == 4.0
        assert summary["total_count"] == 2
        assert summary["low_rating_count"] == 0
        assert summary["high_rating_count"] == 1

    def test_too_hard_detection(self, fb_db: Path):
        save_feedback("learn", "A", 2, comment="This was too hard", db_path=fb_db)
        summary = get_feedback_summary(db_path=fb_db)
        assert summary["mentions_too_hard"] is True
        assert summary["suggestion"] == "simplify"

    def test_too_easy_detection(self, fb_db: Path):
        save_feedback("quiz", "A", 4, comment="It was too easy for me", db_path=fb_db)
        summary = get_feedback_summary(db_path=fb_db)
        assert summary["mentions_too_easy"] is True
        assert summary["suggestion"] == "increase_difficulty"

    def test_low_ratings_trigger_simplify(self, fb_db: Path):
        for _ in range(3):
            save_feedback("learn", "T", 2, db_path=fb_db)
        summary = get_feedback_summary(db_path=fb_db)
        assert summary["suggestion"] == "simplify"


# ---------------------------------------------------------------------------
# _derive_suggestion (deterministic rules)
# ---------------------------------------------------------------------------

class TestDeriveSuggestion:
    def test_too_hard_wins(self):
        assert _derive_suggestion(4.0, 0, 5, too_easy=False, too_hard=True) == "simplify"

    def test_too_easy(self):
        assert _derive_suggestion(4.0, 0, 5, too_easy=True, too_hard=False) == "increase_difficulty"

    def test_too_hard_beats_too_easy(self):
        # If both present, too_hard takes priority
        assert _derive_suggestion(3.0, 1, 3, too_easy=True, too_hard=True) == "simplify"

    def test_majority_low_ratings(self):
        assert _derive_suggestion(2.0, 3, 4, too_easy=False, too_hard=False) == "simplify"

    def test_low_average(self):
        assert _derive_suggestion(2.0, 0, 3, too_easy=False, too_hard=False) == "simplify"

    def test_no_suggestion(self):
        assert _derive_suggestion(4.0, 0, 5, too_easy=False, too_hard=False) is None

    def test_single_entry_no_suggestion(self):
        # Only 1 entry with avg=2 — not enough data
        assert _derive_suggestion(2.0, 1, 1, too_easy=False, too_hard=False) is None


# ---------------------------------------------------------------------------
# Personalization effect from feedback
# ---------------------------------------------------------------------------

class TestPersonalizationEffect:
    def test_feedback_suggestion_in_learn_memory_context(self):
        """Verify _build_memory_context uses feedback_suggestion."""
        from src.graphs.learn_nodes import _build_memory_context

        state = {
            "memory_profile": {
                "recent_topics": ["AI Agents"],
                "recurring_weak_areas": [],
                "average_score": 70,
                "preferred_style": None,
                "suggested_focus_topics": [],
                "feedback_suggestion": "simplify",
            },
        }
        ctx = _build_memory_context(state)
        assert "simpler and clearer" in ctx

    def test_feedback_suggestion_increase_difficulty(self):
        from src.graphs.learn_nodes import _build_memory_context

        state = {
            "memory_profile": {
                "recent_topics": ["AI Agents"],
                "recurring_weak_areas": [],
                "average_score": 70,
                "preferred_style": None,
                "suggested_focus_topics": [],
                "feedback_suggestion": "increase_difficulty",
            },
        }
        ctx = _build_memory_context(state)
        assert "more challenging" in ctx

    def test_quiz_memory_context_uses_feedback(self):
        from src.graphs.quiz_nodes import _build_quiz_memory_context

        state = {
            "memory_profile": {
                "recent_topics": ["LangGraph"],
                "recurring_weak_areas": [],
                "average_score": 70,
                "preferred_style": None,
                "suggested_focus_topics": [],
                "feedback_suggestion": "simplify",
            },
        }
        ctx = _build_quiz_memory_context(state)
        assert "simpler and clearer" in ctx

    def test_no_feedback_no_effect(self):
        from src.graphs.learn_nodes import _build_memory_context

        state = {
            "memory_profile": {
                "recent_topics": ["AI Agents"],
                "recurring_weak_areas": [],
                "average_score": 70,
                "preferred_style": None,
                "suggested_focus_topics": [],
            },
        }
        ctx = _build_memory_context(state)
        assert "simpler" not in ctx
        assert "challenging" not in ctx
