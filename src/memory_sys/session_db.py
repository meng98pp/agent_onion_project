"""Layer 2：SQLite 会话短期记忆。"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path

from src.common.config import MEMORY_DB_PATH


class SessionDB:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else MEMORY_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time  TEXT NOT NULL,
                    end_time    TEXT,
                    title       TEXT,
                    flushed     INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  INTEGER NOT NULL,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    timestamp   TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );
                """
            )

    def new_session(self) -> int:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            cur = conn.execute("INSERT INTO sessions (start_time) VALUES (?)", (now,))
            return int(cur.lastrowid)

    def close_session(self, session_id: int, title: str | None = None) -> None:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET end_time=?, title=? WHERE id=?",
                (now, title, session_id),
            )

    def mark_flushed(self, session_id: int) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE sessions SET flushed=1 WHERE id=?", (session_id,))

    def add_message(self, session_id: int, role: str, content: str) -> None:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
                (session_id, role, content, now),
            )

    def get_session_messages(self, session_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, timestamp FROM messages WHERE session_id=? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_message_count(self, session_id: int) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM messages WHERE session_id=?",
                (session_id,),
            ).fetchone()
        return int(row["cnt"]) if row else 0

    def get_recent_sessions(self, limit: int = 5) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.id, s.start_time, s.title, s.flushed,
                       COUNT(m.id) AS msg_count
                FROM sessions s
                LEFT JOIN messages m ON m.session_id = s.id
                WHERE s.end_time IS NOT NULL
                GROUP BY s.id
                ORDER BY s.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_today_messages(self) -> list[dict]:
        today = date.today().isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT m.role, m.content, m.timestamp, m.session_id
                FROM messages m
                JOIN sessions s ON s.id = m.session_id
                WHERE DATE(m.timestamp) = ?
                ORDER BY m.id
                """,
                (today,),
            ).fetchall()
        return [dict(row) for row in rows]
