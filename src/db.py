"""
SQLite-backed persistent Q&A history.

Thread-safe (uses a connection per call + check_same_thread=False pattern).
Automatically creates the schema on first use.
"""
import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from . import config

logger = logging.getLogger(__name__)

_init_lock = threading.Lock()
_initialized = False


def _ensure_parent_dir(path: str) -> None:
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def _connect():
    """Yield a short-lived SQLite connection with row access by name."""
    conn = sqlite3.connect(config.DB_PATH, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist. Idempotent and thread-safe."""
    global _initialized
    with _init_lock:
        if _initialized:
            return
        _ensure_parent_dir(config.DB_PATH)
        with _connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS qa_history (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
                    source_url      TEXT,
                    source_type     TEXT,
                    product_title   TEXT,
                    question        TEXT    NOT NULL,
                    answer          TEXT    NOT NULL,
                    confidence      REAL    NOT NULL,
                    confidence_level TEXT   NOT NULL,
                    inference_ms    INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_history_created
                    ON qa_history(created_at DESC);
                """
            )
        _initialized = True
        logger.info(f"SQLite history ready at {config.DB_PATH}")


def save_qa(
    question: str,
    answer: str,
    confidence: float,
    confidence_level: str,
    inference_ms: int,
    source_url: Optional[str] = None,
    source_type: Optional[str] = None,
    product_title: Optional[str] = None,
) -> int:
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO qa_history
                (source_url, source_type, product_title,
                 question, answer, confidence, confidence_level, inference_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (source_url, source_type, product_title,
             question, answer, confidence, confidence_level, inference_ms),
        )
        return cur.lastrowid


def list_history(limit: int = None) -> list:
    init_db()
    limit = limit or config.HISTORY_LIMIT
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, source_url, source_type, product_title,
                   question, answer, confidence, confidence_level, inference_ms
              FROM qa_history
             ORDER BY id DESC
             LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def delete_entry(entry_id: int) -> bool:
    init_db()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM qa_history WHERE id = ?", (entry_id,))
        return cur.rowcount > 0


def clear_history() -> int:
    init_db()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM qa_history")
        return cur.rowcount
