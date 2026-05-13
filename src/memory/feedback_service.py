"""Feedback service — store and query user feedback via SQLite."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.memory.db import DEFAULT_DB_PATH

CREATE_FEEDBACK_TABLE = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    context_type TEXT NOT NULL,
    topic TEXT NOT NULL,
    rating INTEGER NOT NULL,
    comment TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}'
);
"""


def _get_conn(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a connection and ensure the feedback table exists."""
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.Connection(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute(CREATE_FEEDBACK_TABLE)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_feedback(
    context_type: str,
    topic: str,
    rating: int,
    comment: str = "",
    metadata: dict[str, Any] | None = None,
    db_path: Path | None = None,
) -> int:
    """Persist a feedback entry and return its row id.

    Args:
        context_type: ``"learn"`` or ``"quiz"``.
        topic: The topic the feedback relates to.
        rating: 1–5 integer rating.
        comment: Optional free-text comment.
        metadata: Optional JSON-serialisable dict.
        db_path: Override DB path (used in tests).
    """
    rating = max(1, min(5, int(rating)))
    conn = _get_conn(db_path)
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata or {})
        cursor = conn.execute(
            "INSERT INTO feedback (timestamp, context_type, topic, rating, comment, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (timestamp, context_type, topic, rating, comment, meta_json),
        )
        conn.commit()
        row_id = cursor.lastrowid
        logger.info("Saved feedback id={} type={!r} topic={!r} rating={}", row_id, context_type, topic, rating)
        return row_id
    finally:
        conn.close()


def get_recent_feedback(
    limit: int = 10,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return the most recent feedback entries (newest first)."""
    conn = _get_conn(db_path)
    try:
        rows = conn.execute(
            "SELECT id, timestamp, context_type, topic, rating, comment, metadata "
            "FROM feedback ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "timestamp": r["timestamp"],
                "context_type": r["context_type"],
                "topic": r["topic"],
                "rating": r["rating"],
                "comment": r["comment"],
                "metadata": json.loads(r["metadata"]),
            }
            for r in rows
        ]
    finally:
        conn.close()


def has_feedback_for_result(
    context_type: str,
    topic: str,
    *,
    metadata: dict[str, Any] | None = None,
    db_path: Path | None = None,
) -> bool:
    """Return whether matching feedback already exists for one logical result."""
    expected = metadata or {}
    expected_signature = expected.get("result_signature")

    conn = _get_conn(db_path)
    try:
        rows = conn.execute(
            "SELECT metadata FROM feedback WHERE context_type = ? AND topic = ?",
            (context_type, topic),
        ).fetchall()
    finally:
        conn.close()

    for row in rows:
        stored = json.loads(row["metadata"])
        if expected_signature and stored.get("result_signature") == expected_signature:
            return True
        comparable = {
            key: value
            for key, value in expected.items()
            if key in {"learning_mode", "learning_depth", "difficulty", "context_title"}
            and value not in (None, "")
        }
        if comparable and all(stored.get(key) == value for key, value in comparable.items()):
            return True
    return False


def delete_feedback(feedback_id: int, db_path: Path | None = None) -> bool:
    """Delete one persisted feedback entry by id."""
    conn = _get_conn(db_path)
    try:
        cursor = conn.execute("DELETE FROM feedback WHERE id = ?", (feedback_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_feedback_summary(db_path: Path | None = None) -> dict[str, Any]:
    """Build a simple deterministic feedback summary.

    Returns:
        average_rating: mean rating across all entries, or None
        total_count: total feedback count
        low_rating_count: entries with rating <= 2
        high_rating_count: entries with rating >= 4
        mentions_too_easy: True if any comment mentions "too easy"
        mentions_too_hard: True if any comment mentions "too hard"
        suggestion: a simple personalisation hint string or None
    """
    conn = _get_conn(db_path)
    try:
        rows = conn.execute(
            "SELECT rating, comment FROM feedback",
        ).fetchall()

        if not rows:
            return {
                "average_rating": None,
                "total_count": 0,
                "low_rating_count": 0,
                "high_rating_count": 0,
                "mentions_too_easy": False,
                "mentions_too_hard": False,
                "suggestion": None,
            }

        ratings = [r["rating"] for r in rows]
        comments = " ".join((r["comment"] or "").lower() for r in rows)

        avg = round(sum(ratings) / len(ratings), 1)
        low = sum(1 for r in ratings if r <= 2)
        high = sum(1 for r in ratings if r >= 4)
        too_easy = "too easy" in comments
        too_hard = "too hard" in comments

        suggestion = _derive_suggestion(avg, low, len(rows), too_easy, too_hard)

        return {
            "average_rating": avg,
            "total_count": len(rows),
            "low_rating_count": low,
            "high_rating_count": high,
            "mentions_too_easy": too_easy,
            "mentions_too_hard": too_hard,
            "suggestion": suggestion,
        }
    finally:
        conn.close()


def _derive_suggestion(
    avg: float,
    low_count: int,
    total: int,
    too_easy: bool,
    too_hard: bool,
) -> str | None:
    """Deterministic rule-based suggestion from feedback signals."""
    if too_hard:
        return "simplify"
    if too_easy:
        return "increase_difficulty"
    if total >= 3 and low_count / total >= 0.5:
        return "simplify"
    if avg is not None and avg <= 2.5 and total >= 2:
        return "simplify"
    return None
