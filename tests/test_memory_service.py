"""Tests for the SQLite memory service and memory candidate creation."""

import json
import sqlite3
from pathlib import Path

import pytest

from src.memory.db import get_connection
from src.memory.memory_service import (
    LEARN_STUDIED_SOURCE,
    delete_learning_event,
    get_completed_learn_sessions,
    get_recent_topics,
    get_quiz_performance_events,
    get_user_profile_summary,
    get_weak_areas_summary,
    has_completed_learn_session,
    save_learning_event,
)


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    """Return a temporary database path (file does not exist yet)."""
    return tmp_path / "test_learning.db"


# ---------------------------------------------------------------------------
# db.py
# ---------------------------------------------------------------------------


class TestGetConnection:
    def test_creates_db_and_table(self, tmp_db: Path) -> None:
        conn = get_connection(tmp_db)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='learning_events'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        deep_path = tmp_path / "a" / "b" / "c" / "test.db"
        conn = get_connection(deep_path)
        assert deep_path.exists()
        conn.close()

    def test_idempotent(self, tmp_db: Path) -> None:
        conn1 = get_connection(tmp_db)
        conn1.close()
        conn2 = get_connection(tmp_db)
        conn2.close()


# ---------------------------------------------------------------------------
# save_learning_event
# ---------------------------------------------------------------------------


class TestSaveLearningEvent:
    def test_save_and_retrieve(self, tmp_db: Path) -> None:
        row_id = save_learning_event(
            topic="AI Agents",
            score=85.0,
            weak_areas=["tool use", "planning"],
            db_path=tmp_db,
        )
        assert isinstance(row_id, int)
        assert row_id >= 1

        conn = get_connection(tmp_db)
        row = conn.execute("SELECT * FROM learning_events WHERE id=?", (row_id,)).fetchone()
        conn.close()

        assert row["topic"] == "AI Agents"
        assert row["score"] == 85.0
        assert json.loads(row["weak_areas"]) == ["tool use", "planning"]

    def test_save_defaults(self, tmp_db: Path) -> None:
        row_id = save_learning_event(topic="LangGraph", score=50.0, db_path=tmp_db)

        conn = get_connection(tmp_db)
        row = conn.execute("SELECT * FROM learning_events WHERE id=?", (row_id,)).fetchone()
        conn.close()

        assert json.loads(row["weak_areas"]) == []
        assert json.loads(row["metadata"]) == {}

    def test_save_with_metadata(self, tmp_db: Path) -> None:
        meta = {"difficulty": "hard", "questions": 5}
        row_id = save_learning_event(
            topic="RAG",
            score=60.0,
            metadata=meta,
            db_path=tmp_db,
        )
        conn = get_connection(tmp_db)
        row = conn.execute("SELECT * FROM learning_events WHERE id=?", (row_id,)).fetchone()
        conn.close()

        assert json.loads(row["metadata"]) == meta


# ---------------------------------------------------------------------------
# get_recent_topics
# ---------------------------------------------------------------------------


class TestGetRecentTopics:
    def test_returns_newest_first(self, tmp_db: Path) -> None:
        save_learning_event(topic="A", score=10.0, db_path=tmp_db)
        save_learning_event(topic="B", score=20.0, db_path=tmp_db)
        save_learning_event(topic="C", score=30.0, db_path=tmp_db)

        recent = get_recent_topics(limit=2, db_path=tmp_db)
        assert len(recent) == 2
        assert recent[0]["topic"] == "C"
        assert recent[1]["topic"] == "B"

    def test_empty_db(self, tmp_db: Path) -> None:
        recent = get_recent_topics(db_path=tmp_db)
        assert recent == []

    def test_fields_present(self, tmp_db: Path) -> None:
        save_learning_event(topic="X", score=90.0, weak_areas=["a"], db_path=tmp_db)
        recent = get_recent_topics(limit=1, db_path=tmp_db)
        evt = recent[0]
        assert "id" in evt
        assert "topic" in evt
        assert "timestamp" in evt
        assert "score" in evt
        assert "weak_areas" in evt
        assert "metadata" in evt


# ---------------------------------------------------------------------------
# get_weak_areas_summary
# ---------------------------------------------------------------------------


class TestGetWeakAreasSummary:
    def test_aggregation(self, tmp_db: Path) -> None:
        save_learning_event(topic="A", score=50.0, weak_areas=["x", "y"], db_path=tmp_db)
        save_learning_event(topic="B", score=60.0, weak_areas=["y", "z"], db_path=tmp_db)

        summary = get_weak_areas_summary(db_path=tmp_db)
        assert summary["y"] == 2
        assert summary["x"] == 1
        assert summary["z"] == 1

    def test_empty_db(self, tmp_db: Path) -> None:
        assert get_weak_areas_summary(db_path=tmp_db) == {}

    def test_learn_studied_events_do_not_affect_weak_area_summary(self, tmp_db: Path) -> None:
        save_learning_event(
            topic="AI Agents",
            score=80.0,
            weak_areas=["planning"],
            metadata={"source": "quiz_evaluation"},
            db_path=tmp_db,
        )
        save_learning_event(
            topic="LangGraph",
            score=0.0,
            weak_areas=["placeholder-area"],
            metadata={"source": "learn_studied"},
            db_path=tmp_db,
        )

        summary = get_weak_areas_summary(db_path=tmp_db)
        assert summary == {"planning": 1}


class TestStudyProgressEvents:
    def test_learn_studied_events_show_in_recent_topics_but_not_average_score(self, tmp_db: Path) -> None:
        save_learning_event(
            topic="AI Agents",
            score=82.0,
            weak_areas=["tool use"],
            metadata={"source": "quiz_evaluation"},
            db_path=tmp_db,
        )
        save_learning_event(
            topic="Advanced Agentic RAG",
            score=0.0,
            metadata={
                "source": "learn_studied",
                "learning_mode": "Learn Path",
                "learning_depth": "Deep Study",
                "summary": "Reviewed retrieval and orchestration tradeoffs.",
            },
            db_path=tmp_db,
        )

        recent = get_recent_topics(limit=2, db_path=tmp_db)
        profile = get_user_profile_summary(db_path=tmp_db)

        assert recent[0]["topic"] == "Advanced Agentic RAG"
        assert recent[0]["metadata"]["source"] == "learn_studied"
        assert profile["recent_topics"][0] == "Advanced Agentic RAG"
        assert profile["average_score"] == 82.0
        assert "Advanced Agentic RAG" not in profile["suggested_focus_topics"]

    def test_get_completed_learn_sessions_filters_only_studied_events(self, tmp_db: Path) -> None:
        save_learning_event(
            topic="Foundations of LLM Application Development",
            score=0.0,
            metadata={
                "source": "learn_studied",
                "learning_mode": "Learn Path",
                "learning_depth": "Summary",
                "difficulty": "Beginner",
            },
            db_path=tmp_db,
        )
        save_learning_event(
            topic="AI Agents",
            score=78.0,
            metadata={"source": "quiz_evaluation"},
            db_path=tmp_db,
        )

        completed = get_completed_learn_sessions(db_path=tmp_db)
        assert len(completed) == 1
        assert completed[0]["topic"] == "Foundations of LLM Application Development"

    def test_get_quiz_performance_events_filters_out_studied_events(self, tmp_db: Path) -> None:
        save_learning_event(
            topic="Foundations of LLM Application Development",
            score=0.0,
            metadata={"source": "learn_studied"},
            db_path=tmp_db,
        )
        save_learning_event(
            topic="AI Agents",
            score=78.0,
            metadata={"source": "quiz_evaluation", "difficulty": "Intermediate"},
            db_path=tmp_db,
        )

        quizzes = get_quiz_performance_events(db_path=tmp_db)
        assert len(quizzes) == 1
        assert quizzes[0]["topic"] == "AI Agents"

    def test_has_completed_learn_session_detects_duplicate(self, tmp_db: Path) -> None:
        save_learning_event(
            topic="AI Agents and Orchestration",
            score=0.0,
            metadata={
                "source": "learn_studied",
                "learning_mode": "Learn Path",
                "learning_depth": "Deep Study",
                "difficulty": "Advanced",
            },
            db_path=tmp_db,
        )

        assert has_completed_learn_session(
            "AI Agents and Orchestration",
            learning_mode="Learn Path",
            learning_depth="Deep Study",
            difficulty="Advanced",
            db_path=tmp_db,
        ) is True
        assert has_completed_learn_session(
            "AI Agents and Orchestration",
            learning_mode="Learn Path",
            learning_depth="Summary",
            difficulty="Advanced",
            db_path=tmp_db,
        ) is False

    def test_delete_learning_event_removes_learn_studied_row(self, tmp_db: Path) -> None:
        event_id = save_learning_event(
            topic="Foundations of LLM Application Development",
            score=0.0,
            metadata={"source": LEARN_STUDIED_SOURCE},
            db_path=tmp_db,
        )

        deleted = delete_learning_event(
            event_id,
            source=LEARN_STUDIED_SOURCE,
            db_path=tmp_db,
        )

        assert deleted is True
        assert get_completed_learn_sessions(db_path=tmp_db) == []

    def test_delete_learning_event_does_not_remove_quiz_rows_when_source_guarded(self, tmp_db: Path) -> None:
        event_id = save_learning_event(
            topic="AI Agents",
            score=78.0,
            metadata={"source": "quiz_evaluation", "difficulty": "Intermediate"},
            db_path=tmp_db,
        )

        deleted = delete_learning_event(
            event_id,
            source=LEARN_STUDIED_SOURCE,
            db_path=tmp_db,
        )

        assert deleted is False
        quizzes = get_quiz_performance_events(db_path=tmp_db)
        assert len(quizzes) == 1
        assert quizzes[0]["id"] == event_id


# ---------------------------------------------------------------------------
# Memory candidate creation (quiz node)
# ---------------------------------------------------------------------------


class TestMemoryCandidate:
    def test_create_memory_candidate(self) -> None:
        from src.graphs.quiz_nodes import create_memory_candidate

        state = {
            "topic": "LangGraph",
            "score": 80.0,
            "weak_areas": ["state management"],
            "trace": [],
        }
        result = create_memory_candidate(state)
        candidate = result["memory_candidate"]

        assert candidate["topic"] == "LangGraph"
        assert candidate["score"] == 80.0
        assert candidate["weak_areas"] == ["state management"]

    def test_candidate_defaults(self) -> None:
        from src.graphs.quiz_nodes import create_memory_candidate

        result = create_memory_candidate({"trace": []})
        candidate = result["memory_candidate"]

        assert candidate["topic"] == "Unknown"
        assert candidate["score"] == 0.0
        assert candidate["weak_areas"] == []


# ---------------------------------------------------------------------------
# No-crash behaviour when DB path is inaccessible
# ---------------------------------------------------------------------------


class TestNoCrashBehavior:
    def test_get_connection_on_fresh_path(self, tmp_path: Path) -> None:
        """DB creation should not crash on a fresh directory."""
        db_path = tmp_path / "fresh" / "learning.db"
        conn = get_connection(db_path)
        assert db_path.exists()
        conn.close()
