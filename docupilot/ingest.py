"""Ingestion pipeline: load -> validate -> clean -> chunk -> embed -> upsert into Pinecone
-> register (summary + full text) locally.

Re-ingesting a file replaces its old chunks (IDs are prefixed with the source name),
so the index never fills up with stale duplicates.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from . import prompts
from .config import get_settings
from .errors import DocuPilotError
from .registry import DocumentRegistry

SUPPORTED = {".pdf", ".md", ".txt"}
MIN_CHARS_PER_PAGE = 25  # below this a PDF page is almost certainly a scanned image


def _slug(text: str) -> str:
    """Pinecone IDs must be ASCII; keep them readable."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:80] or "doc"


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # re-join hyphenated line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_file(path: Path, display_name: str | None = None) -> tuple[list[Document], list[str]]:
    """Return (documents, warnings)."""
    name = display_name or path.name
    suffix = path.suffix.lower()
    warnings: list[str] = []
    if suffix == ".pdf":
        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted:
                reader.decrypt("")
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception as e:  # noqa: BLE001
            raise DocuPilotError(f"Could not read '{name}': {e}") from e
        docs = [Document(_clean(t), metadata={"source": name, "page": i}) for i, t in enumerate(pages, 1)]
        empty = [d.metadata["page"] for d in docs if len(d.page_content) < MIN_CHARS_PER_PAGE]
        if pages and len(empty) == len(pages):
            raise DocuPilotError(
                f"'{name}' has no selectable text — it looks like a scanned/image PDF. "
                "Export it with text (or run OCR) and upload again."
            )
        if empty:
            shown = ", ".join(map(str, empty[:10])) + ("…" if len(empty) > 10 else "")
            warnings.append(f"{name}: {len(empty)} page(s) had little or no text (pages {shown}); "
                            "they may be images or scans.")
    elif suffix in {".md", ".txt"}:
        docs = [Document(_clean(path.read_text(encoding="utf-8", errors="ignore")), metadata={"source": name})]
    else:
        raise DocuPilotError(f"Unsupported file type: {suffix} (supported: {', '.join(sorted(SUPPORTED))})")

    docs = [d for d in docs if d.page_content]
    if not docs:
        raise DocuPilotError(f"'{name}' is empty.")
    return docs, warnings


def _section_title(text: str) -> str | None:
    """Best-effort: nearest markdown heading inside the chunk."""
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()[:120]
    return None


def merge_short_pages(docs: list[Document], target: int) -> list[Document]:
    """Slide decks have many short pages (our Unit 5 deck averages ~170 characters per slide).
    One chunk per slide gives the search engine too little context, so consecutive pages are
    merged until a block reaches `target` characters. Each block keeps its page range
    (`page` .. `page_end`) so citations still say "pp. 12–15"."""
    if not docs or "page" not in docs[0].metadata:
        return docs
    merged: list[Document] = []
    buf: list[Document] = []

    def flush():
        if not buf:
            return
        text = "\n\n".join(f"[Page {d.metadata['page']}]\n{d.page_content}" for d in buf) if len(buf) > 1 else buf[0].page_content
        meta = {**buf[0].metadata}
        if len(buf) > 1:
            meta["page_end"] = buf[-1].metadata["page"]
        merged.append(Document(text, metadata=meta))
        buf.clear()

    size = 0
    for d in docs:
        n = len(d.page_content)
        if buf and size + n > target:
            flush()
            size = 0
        buf.append(d)
        size += n
    flush()
    return merged


def chunk_documents(docs: list[Document]) -> list[Document]:
    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    if s.merge_short_pages:
        docs = merge_short_pages(docs, int(s.chunk_size * 0.9))  # room for the [Page n] markers
    chunks = splitter.split_documents(docs)
    counters: dict[str, int] = {}
    last_section: dict[str, str] = {}
    for c in chunks:
        src = c.metadata["source"]
        counters[src] = counters.get(src, 0) + 1
        c.metadata["chunk"] = counters[src]
        title = _section_title(c.page_content) or last_section.get(src)
        if title:
            c.metadata["section"] = title
            last_section[src] = title
    return chunks


def chunk_id(doc: Document) -> str:
    digest = hashlib.sha1(doc.page_content.encode()).hexdigest()[:10]
    return f"{_slug(doc.metadata['source'])}#{doc.metadata['chunk']:04d}-{digest}"


def full_text(docs: list[Document]) -> str:
    parts = []
    for d in docs:
        page = d.metadata.get("page")
        parts.append(f"[Page {page}]\n{d.page_content}" if page else d.page_content)
    return "\n\n".join(parts)


def summarize(text: str, name: str, llm=None) -> str | None:
    """One LLM call per file; failures are non-fatal (the summary is optional)."""
    try:
        if llm is None:
            from .llm import get_llm
            llm = get_llm()
        msg = llm.invoke([
            SystemMessage(prompts.SUMMARY_SYSTEM),
            HumanMessage(f"Document: {name}\n\n{text[:60000]}"),
        ])
        content = msg.content if isinstance(msg.content, str) else "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in msg.content)
        return content.strip() or None
    except Exception as e:  # noqa: BLE001
        print(f"  (summary skipped for {name}: {type(e).__name__})")
        return None


def delete_source(index, source: str, namespace: str) -> int:
    """Remove all vectors that came from `source` (serverless-friendly: list by ID prefix)."""
    removed = 0
    try:
        for id_batch in index.list(prefix=f"{_slug(source)}#", namespace=namespace):
            if id_batch:
                index.delete(ids=list(id_batch), namespace=namespace)
                removed += len(id_batch)
    except Exception as e:  # noqa: BLE001 — a brand-new namespace can 404 on list()
        if "404" not in str(e) and "not found" not in str(e).lower():
            raise
    return removed


def delete_document(name: str, namespace: str | None = None) -> int:
    """Remove a document from Pinecone and the local registry."""
    from .vectorstore import ensure_index

    ns = namespace or get_settings().pinecone_namespace
    removed = delete_source(ensure_index(), name, ns)
    DocumentRegistry().remove(ns, name)
    return removed


def ingest_paths(paths: list[Path], namespace: str | None = None, names: list[str] | None = None,
                 progress: Callable[[str], None] | None = None) -> dict:
    """Ingest files/folders. `names` lets the UI keep original upload filenames."""
    from .vectorstore import ensure_index, get_vectorstore

    s = get_settings()
    ns = namespace or s.pinecone_namespace
    say = progress or print

    files: list[tuple[Path, str]] = []
    for i, p in enumerate(paths):
        p = Path(p)
        if p.is_dir():
            files += [(f, f.name) for f in sorted(p.rglob("*")) if f.suffix.lower() in SUPPORTED]
        elif p.suffix.lower() in SUPPORTED:
            files.append((p, names[i] if names else p.name))
    if not files:
        raise DocuPilotError("No supported files found (.pdf, .md, .txt).")

    index = ensure_index()
    store = get_vectorstore(ns)
    registry = DocumentRegistry()
    report = {"namespace": ns, "files": [], "warnings": [], "errors": []}
    for path, name in files:
        try:
            docs, warnings = load_file(path, name)
        except DocuPilotError as e:
            report["errors"].append(str(e))
            say(f"  ✗ {e}")
            continue
        report["warnings"] += warnings
        chunks = chunk_documents(docs)
        removed = delete_source(index, name, ns)
        store.add_documents(chunks, ids=[chunk_id(c) for c in chunks], batch_size=64)
        text = full_text(docs)
        summary = summarize(text, name) if s.summarize_on_ingest else None
        registry.add(ns, name, text, pages=sum(1 for d in docs if "page" in d.metadata) or None,
                     chunks=len(chunks), summary=summary)
        registry.save_chunks(ns, name, chunks)
        report["files"].append({"file": name, "chunks": len(chunks), "replaced": removed})
        say(f"  ✓ {name}: {len(chunks)} chunks" + (f" (replaced {removed})" if removed else ""))
    return report
