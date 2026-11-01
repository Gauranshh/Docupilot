"""Persistent conversations (SQLite checkpointer + thread titles) and answer feedback."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .config import get_settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ChatStore:
    """One SQLite file holds LangGraph checkpoints *and* a small table of conversation titles."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or get_settings().data_dir / "chats.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS threads (id TEXT PRIMARY KEY, title TEXT, namespace TEXT, "
            "created TEXT, updated TEXT)"
        )
        self.conn.commit()

    def checkpointer(self):
        from langgraph.checkpoint.sqlite import SqliteSaver

        return SqliteSaver(self.conn)

    def touch(self, thread_id: str, namespace: str, first_question: str) -> None:
        title = first_question.strip().replace("\n", " ")[:60]
        now = _now()
        self.conn.execute(
            "INSERT INTO threads (id, title, namespace, created, updated) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET updated = excluded.updated",
            (thread_id, title, namespace, now, now),
        )
        self.conn.commit()

    def recent(self, namespace: str, limit: int = 10) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, title, updated FROM threads WHERE namespace = ? ORDER BY updated DESC LIMIT ?",
            (namespace, limit),
        ).fetchall()
        return [{"id": r[0], "title": r[1], "updated": r[2]} for r in rows]

    def delete(self, thread_id: str) -> None:
        self.conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
        for table in ("checkpoints", "writes"):
            try:
                self.conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
            except sqlite3.OperationalError:
                pass
        self.conn.commit()


def log_feedback(record: dict, path: Path | None = None) -> None:
    """Append 👍/👎 feedback as JSON lines — reusable later as new evaluation cases."""
    p = Path(path or get_settings().data_dir / "feedback.jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": _now(), **record}, ensure_ascii=False) + "\n")
