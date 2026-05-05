"""Lightweight caching service backed by SQLite."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from loguru import logger

DEFAULT_CACHE_DB = Path("data/memory/cache.db")
DEFAULT_TTL = 3600  # 1 hour

CREATE_CACHE_TABLE = """
CREATE TABLE IF NOT EXISTS cache_entries (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    created_at REAL NOT NULL,
    ttl_seconds INTEGER NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_conn(db_path: Path | None = None) -> sqlite3.Connection:
    """Open (or create) the cache database."""
    path = db_path or DEFAULT_CACHE_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.Connection(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute(CREATE_CACHE_TABLE)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_cache_key(operation: str, payload: dict[str, Any]) -> str:
    """Build a deterministic cache key from an operation name and payload dict.

    Uses SHA-256 of the JSON-serialised payload so that identical inputs
    always produce the same key.
    """
    raw = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{operation}:{digest}"


def set_cached_value(
    key: str,
    value: Any,
    ttl_seconds: int = DEFAULT_TTL,
    metadata: dict[str, Any] | None = None,
    db_path: Path | None = None,
) -> None:
    """Store a value in the cache (upsert)."""
    try:
        conn = _get_conn(db_path)
        try:
            val_json = json.dumps(value, default=str)
            meta_json = json.dumps(metadata or {})
            conn.execute(
                "INSERT OR REPLACE INTO cache_entries "
                "(key, value, created_at, ttl_seconds, metadata) VALUES (?, ?, ?, ?, ?)",
                (key, val_json, time.time(), ttl_seconds, meta_json),
            )
            conn.commit()
            logger.debug("Cache SET key={}", key)
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("Cache set failed for key={}: {}", key, exc)


def get_cached_value(
    key: str,
    db_path: Path | None = None,
) -> Any | None:
    """Retrieve a cached value if it exists and is not expired.

    Returns ``None`` on miss, expiry, or any error.
    """
    try:
        conn = _get_conn(db_path)
        try:
            row = conn.execute(
                "SELECT value, created_at, ttl_seconds FROM cache_entries WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None

            age = time.time() - row["created_at"]
            if age > row["ttl_seconds"]:
                # Expired — clean up lazily
                conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                conn.commit()
                logger.debug("Cache EXPIRED key={}", key)
                return None

            logger.debug("Cache HIT key={}", key)
            return json.loads(row["value"])
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("Cache get failed for key={}: {}", key, exc)
        return None


def clear_expired_cache(db_path: Path | None = None) -> int:
    """Remove all expired entries and return the count deleted."""
    try:
        conn = _get_conn(db_path)
        try:
            now = time.time()
            cursor = conn.execute(
                "DELETE FROM cache_entries WHERE (? - created_at) > ttl_seconds",
                (now,),
            )
            conn.commit()
            deleted = cursor.rowcount
            if deleted:
                logger.info("Cleared {} expired cache entries", deleted)
            return deleted
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("clear_expired_cache failed: {}", exc)
        return 0
