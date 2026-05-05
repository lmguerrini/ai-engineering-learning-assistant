"""SQLite database initialization for learning memory."""

import sqlite3
from pathlib import Path

from loguru import logger

DEFAULT_DB_PATH = Path("data/memory/learning.db")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS learning_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    score REAL NOT NULL,
    weak_areas TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}'
);
"""


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a connection to the SQLite database, creating it if needed.

    Creates parent directories and the learning_events table automatically.
    """
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.Connection(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute(CREATE_TABLE_SQL)
    conn.commit()
    logger.debug("SQLite connection ready: {}", path)
    return conn
