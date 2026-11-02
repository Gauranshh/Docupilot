"""Local document registry.

Pinecone stores chunks for similarity search. The registry stores, per namespace and file:
page count, chunk count, an LLM-written summary and the full extracted text. This powers
"whole-document" questions ("summarise this", "how many questions are there?") that chunk
retrieval alone answers badly, plus the document manager in the UI.
"""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document

from .config import get_settings

_LOCK = threading.Lock()


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:120]


class DocumentRegistry:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or get_settings().data_dir)
        self.path = self.root / "registry.json"

    # ---------- persistence ----------
    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _save(self, data: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def _text_path(self, namespace: str, name: str) -> Path:
        return self.root / "texts" / _safe(namespace) / f"{_safe(name)}.txt"

    # ---------- API ----------
    def add(self, namespace: str, name: str, text: str, *, pages: int | None, chunks: int,
            summary: str | None) -> None:
        tp = self._text_path(namespace, name)
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text(text, encoding="utf-8")
        with _LOCK:
            data = self._load()
            data.setdefault(namespace, {})[name] = {
                "pages": pages,
                "chunks": chunks,
                "chars": len(text),
                "summary": summary,
                "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            self._save(data)

    def _chunks_path(self, namespace: str, name: str) -> Path:
        return self.root / "chunks" / _safe(namespace) / f"{_safe(name)}.json"

    def save_chunks(self, namespace: str, name: str, chunks: list[Document]) -> None:
        """Keep a local copy of a file's chunks for keyword (BM25) search."""
        p = self._chunks_path(namespace, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"text": c.page_content, "metadata": c.metadata} for c in chunks]
        p.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    def load_chunks(self, namespace: str) -> list[Document]:
        folder = self.root / "chunks" / _safe(namespace)
        docs: list[Document] = []
        for p in sorted(folder.glob("*.json")) if folder.exists() else []:
            try:
                rows = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            docs += [Document(r["text"], metadata=r["metadata"]) for r in rows]
        return docs

    def chunks_signature(self, namespace: str) -> tuple:
        """Changes whenever a chunk file is added, replaced or removed (used to rebuild BM25)."""
        folder = self.root / "chunks" / _safe(namespace)
        if not folder.exists():
            return ()
        return tuple(sorted((p.name, p.stat().st_mtime_ns) for p in folder.glob("*.json")))

    def remove(self, namespace: str, name: str) -> None:
        self._chunks_path(namespace, name).unlink(missing_ok=True)
        self._text_path(namespace, name).unlink(missing_ok=True)
        with _LOCK:
            data = self._load()
            data.get(namespace, {}).pop(name, None)
            self._save(data)

    def list(self, namespace: str) -> dict[str, dict]:
        return dict(sorted(self._load().get(namespace, {}).items()))

    def signature(self, namespace: str) -> str:
        """Fingerprint of a knowledge base: changes whenever a document is added, replaced or deleted."""
        import hashlib
        docs = self.list(namespace)
        raw = json.dumps({n: [m.get("ingested_at"), m.get("chunks"), m.get("chars")] for n, m in docs.items()}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def text(self, namespace: str, name: str) -> str | None:
        tp = self._text_path(namespace, name)
        return tp.read_text(encoding="utf-8") if tp.exists() else None


def make_document_loader(namespace: str, registry: DocumentRegistry | None = None,
                         char_limit: int | None = None):
    """Return fn(sources) -> full-document context for the 'document' route.

    - Fits all selected documents' full text if it stays under `char_limit`.
    - Otherwise uses each document's summary (and as much text as still fits).
    """
    reg = registry or DocumentRegistry()
    limit = char_limit or get_settings().full_doc_char_limit

    def load(sources: list[str] | None) -> list[Document]:
        docs = reg.list(namespace)
        names = [n for n in docs if not sources or n in sources]
        texts = {n: reg.text(namespace, n) or "" for n in names}
        total = sum(len(t) for t in texts.values())
        out: list[Document] = []
        if total <= limit:
            for n in names:
                meta = docs[n]
                out.append(Document(texts[n], metadata={"source": n, "mode": "full_document",
                                                        "pages": meta.get("pages")}))
            return out
        budget = limit // max(len(names), 1)
        for n in names:
            meta = docs[n]
            body = (f"SUMMARY: {meta.get('summary') or '(none)'}\n\n"
                    f"BEGINNING OF DOCUMENT (truncated):\n{texts[n][:budget]}")
            out.append(Document(body, metadata={"source": n, "mode": "summary+excerpt",
                                                "pages": meta.get("pages")}))
        return out

    return load
