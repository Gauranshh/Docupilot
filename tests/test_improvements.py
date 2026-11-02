"""Tests for image reading, the answer cache, streaming, router learning and restore-from-Pinecone."""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from pypdf import PdfWriter

from docupilot.cache import AnswerCache, cache_key, normalize
from docupilot.config import Settings
from docupilot.graph import DocuPilot, build_graph
from docupilot.ingest import load_file
from docupilot.prompts import NO_ANSWER
from docupilot.router_model import forget_learned, learned_count, record_learned, train
from docupilot.vision import VisionQuotaExhausted, make_page_describer, render_page_png
from tests.test_graph import DOC, FakeLLM


def _blank_pdf(path, pages=3):
    w = PdfWriter()
    for _ in range(pages):
        w.add_blank_page(width=595, height=842)
    with open(path, "wb") as f:
        w.write(f)
    return path


# ---------- image reading ----------

def test_rendered_page_is_png(tmp_path):
    assert render_page_png(_blank_pdf(tmp_path / "a.pdf", 1), 1)[:4] == b"\x89PNG"


def test_scanned_pdf_is_read_when_image_reading_is_on(tmp_path):
    pdf = _blank_pdf(tmp_path / "scan.pdf")
    seen = []
    docs, warnings = load_file(pdf, describe_page=lambda png, page, src: seen.append(page) or f"Tree diagram {page}")
    assert seen == [1, 2, 3]
    assert all(d.page_content.startswith("[Image description] Tree diagram") and d.metadata["vision"] for d in docs)
    assert any("read 3 page image(s)" in w for w in warnings)


def test_blank_pages_are_skipped_and_cap_is_respected(tmp_path, monkeypatch):
    monkeypatch.setenv("VISION_MAX_PAGES", "2")
    pdf = _blank_pdf(tmp_path / "scan.pdf", 4)
    docs, warnings = load_file(pdf, describe_page=lambda png, page, src: "Blank page" if page == 1 else "Table of values")
    assert [d.metadata["page"] for d in docs] == [2]  # page 1 blank, pages 3-4 over the cap
    assert any("only the first 2" in w for w in warnings)


def test_quota_exhaustion_keeps_what_was_read(tmp_path):
    calls = []

    def describe(png, page, src):
        calls.append(page)
        if page == 2:
            raise VisionQuotaExhausted("daily")
        return "Chart"

    docs, warnings = load_file(_blank_pdf(tmp_path / "s.pdf"), describe_page=describe)
    assert calls == [1, 2] and [d.metadata["page"] for d in docs] == [1]
    assert any("quota ran out after reading 1" in w for w in warnings)


def test_describer_sends_the_image_and_maps_daily_quota():
    class LLM:
        def __init__(self, fail=None):
            self.fail, self.messages = fail, None

        def invoke(self, messages):
            self.messages = messages
            if self.fail:
                raise RuntimeError(self.fail)
            return AIMessage("A decision tree")

    llm = LLM()
    assert make_page_describer(llm)(b"\x89PNGdata", 7, "u5.pdf") == "A decision tree"
    blocks = llm.messages[0].content
    assert blocks[0]["type"] == "text" and "page 7" in blocks[0]["text"]
    assert blocks[1]["type"] == "image" and blocks[1]["mime_type"] == "image/png"
    with pytest.raises(VisionQuotaExhausted):
        make_page_describer(LLM("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"))(b"x", 1, "a")


# ---------- answer cache ----------

def test_normalize_and_key():
    assert normalize("  How MANY leaves?! ") == "how many leaves"
    k = cache_key("How many leaves?", "hb", "m", None, "sig1")
    assert k == cache_key("how many leaves", "hb", "m", None, "sig1")
    assert k != cache_key("how many leaves", "hb", "m", None, "sig2")
    assert k != cache_key("how many leaves", "hb", "m", ["a.pdf"], "sig1")


def test_cache_expires(tmp_path):
    c = AnswerCache(tmp_path / "c.sqlite", ttl_hours=1)
    c.put("k", "q", {"answer": "a"})
    assert c.get("k")["answer"] == "a"
    c.conn.execute("UPDATE answers SET created = ?", (time.time() - 7200,))
    assert c.get("k") is None


def _cached_bot(tmp_path, llm, signature):
    s = Settings(max_rewrites=2, max_generation_retries=1, keyword_routing=False)
    g = build_graph(llm, llm, lambda q, k, src=None: [DOC], settings=s)
    return DocuPilot(graph=g, settings=s, cache=AnswerCache(tmp_path / "c.sqlite"), kb_signature=lambda: signature[0])


def test_repeated_question_costs_no_ai_calls(tmp_path):
    llm, sig = FakeLLM(), ["v1"]
    bot = _cached_bot(tmp_path, llm, sig)
    first = bot.ask("How much earned leave?", thread_id="a")
    calls_after_first = len(llm.calls)
    second = bot.ask("how much earned leave", thread_id="b")
    assert not first["cached"] and second["cached"] and len(llm.calls) == calls_after_first
    assert second["answer"] == first["answer"] and second["sources"] == first["sources"]
    assert [m["role"] for m in bot.history("b")] == ["user", "assistant"]
    sig[0] = "v2"  # a document was added → cache no longer applies
    assert not bot.ask("How much earned leave?", thread_id="c")["cached"]


def test_follow_ups_and_refusals_are_not_cached(tmp_path):
    llm = FakeLLM()
    bot = _cached_bot(tmp_path, llm, ["v1"])
    bot.ask("Tell me about leave", thread_id="t")
    bot.ask("and for fathers?", thread_id="t")  # follow-up
    assert bot.cache.stats()["answers"] == 1
    refusing = _cached_bot(tmp_path / "r", FakeLLM(relevant=[]), ["v1"])
    assert refusing.ask("Stock options?", thread_id="x")["answer"] == NO_ANSWER
    assert refusing.cache.stats()["answers"] == 0


# ---------- streaming ----------

def test_answer_is_streamed_token_by_token():
    answer = "Employees get 18 days of earned leave [1]."
    llm = GenericFakeChatModel(messages=iter([AIMessage(answer)]))
    s = Settings(max_rewrites=2, max_generation_retries=1, keyword_routing=False, verify_answers=False)
    g = build_graph(llm, FakeLLM(), lambda q, k, src=None: [DOC], settings=s)
    seen = []
    res = DocuPilot(graph=g, settings=s).ask("How much earned leave?", on_token=seen.append)
    assert len(seen) > 3 and seen[-1] == answer and res["answer"] == answer
    assert all(seen[i] in seen[i + 1] for i in range(len(seen) - 1))  # text only grows


# ---------- router learning ----------

def test_learned_examples_are_recorded_and_forgotten(tmp_path):
    p = tmp_path / "learned.csv"
    record_learned("Explain gain ratio", "retrieve", p)
    record_learned("explain gain ratio", "retrieve", p)  # duplicate ignored
    record_learned("Revision notes for unit 2", "document", p)
    record_learned("bad", "nonsense-label", p)
    assert learned_count(p) == 2
    assert forget_learned("EXPLAIN GAIN RATIO", p) and learned_count(p) == 1


def test_llm_decisions_feed_learning_only_when_router_is_unsure():
    learned = []
    s = Settings(max_rewrites=2, max_generation_retries=1, keyword_routing=False)
    unsure = build_graph(FakeLLM(), FakeLLM(), lambda q, k, src=None: [DOC], settings=s,
                         classifier=lambda q: ("direct", 0.4), on_llm_route=lambda q, r: learned.append((q, r)))
    DocuPilot(graph=unsure, settings=s).ask("How much earned leave?")
    sure = build_graph(FakeLLM(), FakeLLM(), lambda q, k, src=None: [DOC], settings=s,
                       classifier=lambda q: ("retrieve", 0.99), on_llm_route=lambda q, r: learned.append((q, r)))
    DocuPilot(graph=sure, settings=s).ask("Hotel limit?")
    assert learned == [("How much earned leave?", "retrieve")]


def test_training_includes_learned_examples(tmp_path):
    p = tmp_path / "learned.csv"
    record_learned("give me flashcards for the whole chapter", "document", p)
    m = train(model_path=tmp_path / "r.joblib", learned=p)
    assert m["learned_examples"] == 1 and m["examples"] >= 191


# ---------- restore from Pinecone ----------

def test_restore_rebuilds_registry_from_pinecone(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    stored = {
        "u5#0002-x": {"text": "C4.5 by Quinlan", "source": "u5.pdf", "chunk": 2.0, "page": 18.0, "page_end": 22.0},
        "u5#0001-y": {"text": "Supervised learning", "source": "u5.pdf", "chunk": 1.0, "page": 1.0},
        "lv#0001-z": {"text": "18 days of earned leave", "source": "leave.md", "chunk": 1.0},
    }

    class Index:
        def list(self, namespace):
            yield list(stored)

        def fetch(self, ids, namespace):
            return SimpleNamespace(vectors={i: SimpleNamespace(metadata=dict(stored[i])) for i in ids})

    import docupilot.vectorstore as vs
    monkeypatch.setattr(vs, "ensure_index", lambda: Index())
    from docupilot.ingest import restore_from_pinecone
    from docupilot.registry import DocumentRegistry

    r = restore_from_pinecone("hb", progress=lambda m: None)
    assert r["vectors"] == 3 and {x["file"] for x in r["restored"]} == {"u5.pdf", "leave.md"}
    reg = DocumentRegistry(tmp_path)
    assert reg.list("hb")["u5.pdf"]["pages"] == 22 and reg.list("hb")["u5.pdf"]["chunks"] == 2
    chunks = [c for c in reg.load_chunks("hb") if c.metadata["source"] == "u5.pdf"]
    assert [c.metadata["chunk"] for c in chunks] == [1, 2] and isinstance(chunks[1].metadata["page"], int)
    assert reg.text("hb", "u5.pdf").startswith("Supervised learning")
    assert restore_from_pinecone("hb", progress=lambda m: None)["restored"] == []  # already listed → skipped
