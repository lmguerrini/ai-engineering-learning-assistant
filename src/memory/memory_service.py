"""Learning memory service — save and query learning events via SQLite."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.memory.db import get_connection


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


def get_recent_topics(limit: int = 5, db_path: Path | None = None) -> list[dict[str, Any]]:
    """Return the most recent learning events (newest first)."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT id, topic, timestamp, score, weak_areas, metadata "
            "FROM learning_events ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "topic": r["topic"],
                "timestamp": r["timestamp"],
                "score": r["score"],
                "weak_areas": json.loads(r["weak_areas"]),
                "metadata": json.loads(r["metadata"]),
            }
            for r in rows
        ]
    finally:
        conn.close()


def get_weak_areas_summary(db_path: Path | None = None) -> dict[str, int]:
    """Aggregate weak areas across all events, returning {area: count}."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT weak_areas FROM learning_events").fetchall()
        counts: dict[str, int] = {}
        for r in rows:
            for area in json.loads(r["weak_areas"]):
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
            "SELECT topic, score, weak_areas FROM learning_events ORDER BY id DESC",
        ).fetchall()

        if not rows:
            return {
                "recent_topics": [],
                "recurring_weak_areas": [],
                "average_score": None,
                "preferred_style": None,
                "suggested_focus_topics": [],
            }

        # Recent topics (unique, preserving order, max 10)
        seen: set[str] = set()
        recent_topics: list[str] = []
        for r in rows:
            t = r["topic"]
            if t not in seen:
                seen.add(t)
                recent_topics.append(t)
            if len(recent_topics) >= 10:
                break

        # Weak-area counts
        weak_counts: dict[str, int] = {}
        for r in rows:
            for area in json.loads(r["weak_areas"]):
                weak_counts[area] = weak_counts.get(area, 0) + 1
        recurring_weak_areas = [a for a, c in weak_counts.items() if c >= 2]

        # Average score
        scores = [r["score"] for r in rows]
        average_score = round(sum(scores) / len(scores), 1) if scores else None

        # Suggested focus topics: topics whose latest score is below average
        suggested: list[str] = []
        if average_score is not None:
            topic_latest_score: dict[str, float] = {}
            for r in rows:
                t = r["topic"]
                if t not in topic_latest_score:
                    topic_latest_score[t] = r["score"]
            suggested = [t for t, s in topic_latest_score.items() if s < average_score]

        # Also add topics related to recurring weak areas if not already present
        for area in recurring_weak_areas:
            if area not in suggested:
                suggested.append(area)

        logger.debug(
            "User profile: {} recent topics, {} recurring weak areas, avg={}, {} focus topics",
            len(recent_topics), len(recurring_weak_areas), average_score, len(suggested),
        )

        return {
            "recent_topics": recent_topics,
            "recurring_weak_areas": recurring_weak_areas,
            "average_score": average_score,
            "preferred_style": None,
            "suggested_focus_topics": suggested,
        }
    finally:
        conn.close()
