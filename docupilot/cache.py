"""Answer cache: a repeated question is answered instantly with no AI calls.

Free tiers allow ~20 AI calls a day and a question costs up to 4, so asking the same thing
twice should not cost twice. A cached answer is reused only when ALL of these match:
the question (ignoring case, spaces and punctuation), the knowledge base, the file filter,
the chat model, and the documents themselves — any upload or delete changes the
`kb_signature`, which retires every older entry automatically. Only first questions of a
conversation are cached (follow-ups depend on what was said before), and only answers that
were found in the documents and passed verification.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path

_PUNCT = re.compile(r"[^\w\s]")


def normalize(question: str) -> str:
    return " ".join(_PUNCT.sub(" ", question.lower()).split())


def cache_key(question: str, namespace: str, model: str, sources: list[str] | None, kb_signature: str) -> str:
    raw = json.dumps([normalize(question), namespace, model, sorted(sources or []), kb_signature])
    return hashlib.sha256(raw.encode()).hexdigest()


class AnswerCache:
    def __init__(self, path: Path, ttl_hours: float = 168):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_hours * 3600
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, question TEXT, "
                          "result TEXT, created REAL, hits INTEGER DEFAULT 0)")
        self.conn.commit()

    def get(self, key: str) -> dict | None:
        row = self.conn.execute("SELECT result, created FROM answers WHERE key = ?", (key,)).fetchone()
        if not row:
            return None
        if time.time() - row[1] > self.ttl:
            self.conn.execute("DELETE FROM answers WHERE key = ?", (key,))
            self.conn.commit()
            return None
        self.conn.execute("UPDATE answers SET hits = hits + 1 WHERE key = ?", (key,))
        self.conn.commit()
        return json.loads(row[0])

    def put(self, key: str, question: str, result: dict) -> None:
        keep = {k: result.get(k) for k in ("answer", "sources", "grounded", "used_web", "route", "trace")}
        self.conn.execute("INSERT OR REPLACE INTO answers (key, question, result, created, hits) VALUES (?, ?, ?, ?, 0)",
                          (key, question, json.dumps(keep, ensure_ascii=False, default=str), time.time()))
        self.conn.commit()

    def clear(self) -> int:
        n = self.conn.execute("DELETE FROM answers").rowcount
        self.conn.commit()
        return n

    def stats(self) -> dict:
        n, hits = self.conn.execute("SELECT COUNT(*), COALESCE(SUM(hits), 0) FROM answers").fetchone()
        return {"answers": n, "reused": hits}
