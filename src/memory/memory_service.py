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
