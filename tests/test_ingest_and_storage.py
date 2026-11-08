"""Offline tests for ingestion helpers, the document registry and chat persistence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from pypdf import PdfWriter

from docupilot.chats import ChatStore, log_feedback
from docupilot.errors import DocuPilotError, friendly_error, is_transient
from docupilot.ingest import chunk_documents, chunk_id, full_text, load_file, summarize
from docupilot.registry import DocumentRegistry, make_document_loader

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"


def test_chunking_and_stable_ids():
    docs, warnings = load_file(SAMPLE / "leave_policy.md")
    chunks = chunk_documents(docs)
    assert len(chunks) > 1 and warnings == []
    assert all(c.metadata["source"] == "leave_policy.md" for c in chunks)
    assert all("section" in c.metadata for c in chunks)
    ids = [chunk_id(c) for c in chunks]
    assert len(set(ids)) == len(ids) and all(i.startswith("leave_policy.md#") for i in ids)
    assert ids == [chunk_id(c) for c in chunk_documents(load_file(SAMPLE / "leave_policy.md")[0])]


def test_scanned_pdf_is_rejected(tmp_path):
    w = PdfWriter()
    w.add_blank_page(width=595, height=842)
    w.add_blank_page(width=595, height=842)
    pdf = tmp_path / "scan.pdf"
    with pdf.open("wb") as f:
        w.write(f)
    with pytest.raises(DocuPilotError, match="scanned"):
        load_file(pdf)


def test_unsupported_and_empty_files(tmp_path):
    (tmp_path / "a.exe").write_text("x")
    (tmp_path / "broken.docx").write_text("not really a Word file")
    (tmp_path / "e.txt").write_text("   ")
    with pytest.raises(DocuPilotError, match="Unsupported"):
        load_file(tmp_path / "a.exe")
    with pytest.raises(DocuPilotError, match="Could not read"):
        load_file(tmp_path / "broken.docx")
    with pytest.raises(DocuPilotError, match="empty"):
        load_file(tmp_path / "e.txt")


def test_summary_failure_is_non_fatal():
    class Boom:
        def invoke(self, _):
            raise RuntimeError("down")
    assert summarize("text", "f.md", llm=Boom()) is None

    class Ok:
        def invoke(self, _):
            return AIMessage("A leave policy with 5 sections.")
    assert summarize("text", "f.md", llm=Ok()).startswith("A leave policy")


def test_registry_and_document_loader(tmp_path):
    reg = DocumentRegistry(tmp_path)
    docs, _ = load_file(SAMPLE / "leave_policy.md")
    reg.add("hr", "leave_policy.md", full_text(docs), pages=None, chunks=4, summary="Leave rules")
    reg.add("hr", "big.md", "B" * 5000, pages=3, chunks=9, summary="Big doc")
    assert list(reg.list("hr")) == ["big.md", "leave_policy.md"] and reg.list("other") == {}

    full = make_document_loader("hr", reg, char_limit=100_000)(["leave_policy.md"])
    assert len(full) == 1 and full[0].metadata["mode"] == "full_document" and "18 days" in full[0].page_content

    squeezed = make_document_loader("hr", reg, char_limit=1000)(None)
    assert {d.metadata["mode"] for d in squeezed} == {"summary+excerpt"}
    assert all(len(d.page_content) < 1000 for d in squeezed)

    reg.remove("hr", "big.md")
    assert list(reg.list("hr")) == ["leave_policy.md"] and reg.text("hr", "big.md") is None


def test_chat_persistence_survives_restart(tmp_path):
    from langchain_core.documents import Document

    from tests.test_graph import FakeLLM
    from docupilot.config import Settings
    from docupilot.graph import DocuPilot, build_graph

    db = tmp_path / "chats.sqlite"
    doc = Document("18 days of earned leave.", metadata={"source": "leave_policy.md"})

    def bot_with(store):
        g = build_graph(FakeLLM(), FakeLLM(), lambda q, k, s=None: [doc], checkpointer=store.checkpointer(),
                        settings=Settings())
        return DocuPilot(graph=g)

    store = ChatStore(db)
    bot_with(store).ask("How much leave?", thread_id="abc")
    store.touch("abc", "hr", "How much leave?")
    store.conn.close()

    store2 = ChatStore(db)  # "restart"
    assert store2.recent("hr")[0]["title"] == "How much leave?"
    assert [m["role"] for m in bot_with(store2).history("abc")] == ["user", "assistant"]
    store2.delete("abc")
    assert store2.recent("hr") == []


def test_feedback_log(tmp_path):
    p = tmp_path / "fb.jsonl"
    log_feedback({"question": "q", "rating": "up"}, path=p)
    log_feedback({"question": "q2", "rating": "down"}, path=p)
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    assert [r["rating"] for r in rows] == ["up", "down"] and all("ts" in r for r in rows)


@pytest.mark.parametrize("msg, transient", [
    ("429 RESOURCE_EXHAUSTED", True),
    ("503 UNAVAILABLE", True),
    ("429 insufficient_quota", False),
    ("404 NOT_FOUND", False),
    ("401 invalid api key", False),
])
def test_error_classification(msg, transient):
    assert is_transient(RuntimeError(msg)) is transient
    assert friendly_error(RuntimeError(msg))  # always produces a message


GEMINI_DAILY = ("429 RESOURCE_EXHAUSTED. You exceeded your current quota, please check your plan and billing details. "
                "Quota exceeded for metric: generate_content_free_tier_requests, limit: 20, model: gemini-3.8-flash "
                "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
GEMINI_MINUTE = ("429 RESOURCE_EXHAUSTED. You exceeded your current quota, please check your plan and billing details. "
                 "quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier. Please retry in 20s")
GEMINI_ZERO = "429 RESOURCE_EXHAUSTED. quota exceeded, limit: 0, model: gemini-3.8-pro"


def test_gemini_quota_errors_are_classified_correctly():
    from docupilot.errors import quota_kind
    assert quota_kind(RuntimeError(GEMINI_DAILY)) == "daily" and not is_transient(RuntimeError(GEMINI_DAILY))
    assert "today's free-tier quota" in friendly_error(RuntimeError(GEMINI_DAILY), "google_genai:gemini-3.8-flash")
    assert quota_kind(RuntimeError(GEMINI_MINUTE)) == "minute" and is_transient(RuntimeError(GEMINI_MINUTE))
    assert quota_kind(RuntimeError(GEMINI_ZERO)) == "no_free_tier"
    assert "no credit" not in friendly_error(RuntimeError(GEMINI_DAILY))  # the old, misleading message
