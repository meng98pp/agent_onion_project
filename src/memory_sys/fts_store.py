"""Layer 4：SQLite FTS5 / BM25。中文逐字分词；FTS5 不可用时降级。"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path

from src.common.config import MEMORY_DB_PATH

logger = logging.getLogger(__name__)
_CJK_CHAR = re.compile(r"([一-鿿㐀-䶿])")


def tokenize_zh(text: str) -> str:
    if not text:
        return ""
    spaced = _CJK_CHAR.sub(r" \1 ", text)
    return re.sub(r"\s+", " ", spaced).strip()


def _match_query(text: str) -> str:
    tokens = tokenize_zh(text).split()
    return " ".join(f'"{token}"' for token in tokens) if tokens else ""


class FTSStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else MEMORY_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.available = False
        try:
            self._init_db()
            self.available = True
        except sqlite3.OperationalError as exc:
            logger.warning("FTS5 不可用，BM25 禁用：%s", exc)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(memory_fts)").fetchall()}
            if cols and "content_raw" not in cols:
                conn.execute("DROP TABLE memory_fts")
            conn.executescript(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    entry_id UNINDEXED,
                    title,
                    content,
                    category UNINDEXED,
                    title_raw UNINDEXED,
                    content_raw UNINDEXED
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    def add_entries(self, entries: list[dict]) -> int:
        if not self.available or not entries:
            return 0
        conn = self._connect()
        try:
            for item in entries:
                title = str(item.get("title") or "")
                content = str(item.get("content") or "")
                conn.execute(
                    "INSERT INTO memory_fts "
                    "(entry_id, title, content, category, title_raw, content_raw) "
                    "VALUES (?,?,?,?,?,?)",
                    (
                        str(item.get("id") or ""),
                        tokenize_zh(title),
                        tokenize_zh(content),
                        str(item.get("category") or ""),
                        title,
                        content,
                    ),
                )
            conn.commit()
            return len(entries)
        finally:
            conn.close()

    def rebuild_from_entries(self, entries: list[dict]) -> None:
        if not self.available:
            return
        conn = self._connect()
        try:
            conn.execute("DELETE FROM memory_fts")
            conn.commit()
        finally:
            conn.close()
        self.add_entries(entries)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.available:
            return []
        match_query = _match_query(query)
        if not match_query:
            return []
        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT entry_id, title_raw, content_raw, category, "
                    "bm25(memory_fts) AS bm25 "
                    "FROM memory_fts WHERE memory_fts MATCH ? "
                    "ORDER BY bm25 LIMIT ?",
                    (match_query, top_k),
                ).fetchall()
            finally:
                conn.close()
        except sqlite3.OperationalError as exc:
            logger.warning("FTS 查询失败 query=%r：%s", query, exc)
            return []
        if not rows:
            return []
        raws = [-row["bm25"] for row in rows]
        lo, hi = min(raws), max(raws)
        span = hi - lo
        results: list[dict] = []
        for row, raw in zip(rows, raws):
            score = 1.0 if span == 0 else (raw - lo) / span
            results.append(
                {
                    "id": row["entry_id"],
                    "title": row["title_raw"],
                    "content": row["content_raw"],
                    "category": row["category"],
                    "score": score,
                }
            )
        return results

    @property
    def total_entries(self) -> int:
        if not self.available:
            return 0
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM memory_fts").fetchone()
            return int(row["cnt"]) if row else 0
        finally:
            conn.close()
