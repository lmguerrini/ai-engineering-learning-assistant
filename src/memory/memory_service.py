"""Learning memory service — save and query learning events via SQLite."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.memory.db import get_connection

LEARN_STUDIED_SOURCE = "learn_studied"
QUIZ_EVALUATION_SOURCE = "quiz_evaluation"


def _deserialize_learning_event(row: sqlite3.Row) -> dict[str, Any]:
    """Convert one SQLite row into a typed learning-event dictionary."""
    metadata = json.loads(row["metadata"])
    return {
        "id": row["id"],
        "topic": row["topic"],
        "timestamp": row["timestamp"],
        "score": row["score"],
        "weak_areas": json.loads(row["weak_areas"]),
        "metadata": metadata,
    }


def _is_scored_learning_event(event: dict[str, Any]) -> bool:
    """Return whether one learning event should contribute to score-based summaries."""
    metadata = event.get("metadata", {}) or {}
    return metadata.get("source") != LEARN_STUDIED_SOURCE


def _is_completed_learn_event(event: dict[str, Any]) -> bool:
    """Return whether one learning event represents an explicitly saved Learn session."""
    metadata = event.get("metadata", {}) or {}
    return metadata.get("source") == LEARN_STUDIED_SOURCE


def save_learning_event(
    topic: str,
    score: float,
    weak_areas: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    db_path: Path | None = None,
) -> int:
    """Persist a learning event and return its row id."""
    conn = get_connection(db_path)
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        weak_json = json.dumps(weak_areas or [])
        meta_json = json.dumps(metadata or {})

        cursor = conn.execute(
            "INSERT INTO learning_events (topic, timestamp, score, weak_areas, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            (topic, timestamp, score, weak_json, meta_json),
        )
        conn.commit()
        row_id = cursor.lastrowid
        logger.info("Saved learning event id={} topic={!r} score={}", row_id, topic, score)
        return row_id
    finally:
        conn.close()


def delete_learning_event(
    event_id: int,
    *,
    source: str | None = None,
    db_path: Path | None = None,
) -> bool:
    """Delete one persisted learning event, optionally guarded by metadata.source."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT metadata FROM learning_events WHERE id = ?",
            (event_id,),
        ).fetchone()
        if row is None:
            return False

        if source is not None:
            metadata = json.loads(row["metadata"])
            if metadata.get("source") != source:
                return False

        cursor = conn.execute("DELETE FROM learning_events WHERE id = ?", (event_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Deleted learning event id={} source={!r}", event_id, source)
        return deleted
    finally:
        conn.close()


def get_recent_topics(limit: int = 5, db_path: Path | None = None) -> list[dict[str, Any]]:
    """Return the most recent learning events (newest first)."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT id, topic, timestamp, score, weak_areas, metadata "
            "FROM learning_events ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_deserialize_learning_event(r) for r in rows]
    finally:
        conn.close()


def get_completed_learn_sessions(limit: int = 10, db_path: Path | None = None) -> list[dict[str, Any]]:
    """Return explicitly completed Learn sessions, newest first."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT id, topic, timestamp, score, weak_areas, metadata FROM learning_events ORDER BY id DESC"
        ).fetchall()
        completed = [
            event
            for event in (_deserialize_learning_event(row) for row in rows)
            if _is_completed_learn_event(event)
        ]
        return completed[:limit]
    finally:
        conn.close()


def get_quiz_performance_events(limit: int = 10, db_path: Path | None = None) -> list[dict[str, Any]]:
    """Return scored quiz-derived learning events, newest first."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT id, topic, timestamp, score, weak_areas, metadata FROM learning_events ORDER BY id DESC"
        ).fetchall()
        scored = [
            event
            for event in (_deserialize_learning_event(row) for row in rows)
            if _is_scored_learning_event(event)
        ]
        return scored[:limit]
    finally:
        conn.close()


def has_completed_learn_session(
    topic: str,
    *,
    learning_mode: str,
    learning_depth: str,
    difficulty: str = "",
    db_path: Path | None = None,
) -> bool:
    """Return whether the same Learn session was already saved in persisted memory."""
    for event in get_completed_learn_sessions(limit=1000, db_path=db_path):
        metadata = event.get("metadata", {}) or {}
        if event.get("topic") != topic:
            continue
        if metadata.get("learning_mode") != learning_mode:
            continue
        if metadata.get("learning_depth") != learning_depth:
            continue
        if (metadata.get("difficulty") or "") != (difficulty or ""):
            continue
        return True
    return False


def get_weak_areas_summary(db_path: Path | None = None) -> dict[str, int]:
    """Aggregate weak areas across all events, returning {area: count}."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT id, topic, timestamp, score, weak_areas, metadata FROM learning_events"
        ).fetchall()
        counts: dict[str, int] = {}
        for event in (_deserialize_learning_event(r) for r in rows):
            if not _is_scored_learning_event(event):
                continue
            for area in event["weak_areas"]:
                counts[area] = counts.get(area, 0) + 1
        return counts
    finally:
        conn.close()


def get_user_profile_summary(db_path: Path | None = None) -> dict[str, Any]:
    """Build a lightweight user profile from stored learning memory.

    Returns a dict with:
      - recent_topics: list of recently studied topic strings
      - recurring_weak_areas: list of weak areas that appeared more than once
      - average_score: average quiz score or None if no data
      - preferred_style: None (reserved for future use)
      - suggested_focus_topics: topics with below-average scores or recurring weak areas
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT id, topic, timestamp, score, weak_areas, metadata FROM learning_events ORDER BY id DESC",
        ).fetchall()

        if not rows:
            return {
                "recent_topics": [],
                "recent_weak_areas": [],
                "recurring_weak_areas": [],
                "average_score": None,
                "preferred_style": None,
                "suggested_focus_topics": [],
            }

        events = [_deserialize_learning_event(row) for row in rows]

        # Recent topics (unique, preserving order, max 10)
        seen: set[str] = set()
        recent_topics: list[str] = []
        for event in events:
            t = event["topic"]
            if t not in seen:
                seen.add(t)
                recent_topics.append(t)
            if len(recent_topics) >= 10:
                break

        # Weak-area counts
        weak_counts: dict[str, int] = {}
        for event in events:
            if not _is_scored_learning_event(event):
                continue
            for area in event["weak_areas"]:
                weak_counts[area] = weak_counts.get(area, 0) + 1
        recurring_weak_areas = [a for a, c in weak_counts.items() if c >= 2]

        scored_events = [event for event in events if _is_scored_learning_event(event)]
        scores = [event["score"] for event in scored_events]
        average_score = round(sum(scores) / len(scores), 1) if scores else None
        latest_scored_event = scored_events[0] if scored_events else None
        recent_weak_areas = list(dict.fromkeys((latest_scored_event or {}).get("weak_areas", [])))

        # Suggested focus topics: topics whose latest score is below average
        suggested: list[str] = []
        if average_score is not None:
            topic_latest_score: dict[str, float] = {}
            for event in scored_events:
                t = event["topic"]
                if t not in topic_latest_score:
                    topic_latest_score[t] = event["score"]
            suggested = [t for t, s in topic_latest_score.items() if s < average_score]

        # Also add topics related to recurring weak areas if not already present
        for area in recurring_weak_areas:
            if area not in suggested:
                suggested.append(area)

        # Preserve quiz-page suggested topics when available; otherwise derive
        # dashboard-safe focus topics from the latest saved quiz weak areas.
        latest_metadata = (latest_scored_event or {}).get("metadata", {}) or {}
        stored_suggested_topics = latest_metadata.get("suggested_topics", []) or []
        for topic in stored_suggested_topics:
            if topic not in suggested:
                suggested.append(topic)

        if recent_weak_areas:
            try:
                from src.graphs.quiz_nodes import _map_weak_area_to_study_topics

                for area in recent_weak_areas:
                    for topic in _map_weak_area_to_study_topics(area):
                        if topic not in suggested:
                            suggested.append(topic)
            except Exception:
                for area in recent_weak_areas:
                    if area not in suggested:
                        suggested.append(area)

        logger.debug(
            "User profile: {} recent topics, {} recurring weak areas, avg={}, {} focus topics",
            len(recent_topics), len(recurring_weak_areas), average_score, len(suggested),
        )

        return {
            "recent_topics": recent_topics,
            "recent_weak_areas": recent_weak_areas,
            "recurring_weak_areas": recurring_weak_areas,
            "average_score": average_score,
            "preferred_style": None,
            "suggested_focus_topics": suggested,
        }
    finally:
        conn.close()
