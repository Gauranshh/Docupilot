"""Tests for the trained router, hybrid (BM25 + vector) retrieval and slide-page merging."""
from __future__ import annotations

from langchain_core.documents import Document

from docupilot.config import Settings
from docupilot.graph import _label, build_graph, DocuPilot
from docupilot.ingest import chunk_documents, merge_short_pages
from docupilot.keyword_search import BM25Index, reciprocal_rank_fusion, tokenize
from docupilot.registry import DocumentRegistry
from docupilot.router_model import TrainedRouter, train
from docupilot.vectorstore import make_hybrid_retriever
from tests.test_graph import DOC, FakeLLM


# ---------- trained router ----------

def test_router_trains_and_generalises(tmp_path):
    metrics = train(model_path=tmp_path / "router.joblib")
    assert metrics["examples"] >= 150
    assert metrics["holdout_accuracy"] >= 0.8
    assert metrics["confident_cv"]["0.7"]["accuracy"] >= 0.9
    model = TrainedRouter.load(tmp_path / "router.joblib")
    assert model.predict("summarize the pdf")[0] == "document"
    assert model.predict("thank you so much")[0] == "direct"
    assert model.predict("What is the hotel limit in Pune?")[0] == "retrieve"


def test_missing_model_is_not_an_error(tmp_path):
    assert TrainedRouter.load(tmp_path / "nope.joblib") is None


def _bot(llm, classifier, **settings):
    s = Settings(**{"max_rewrites": 2, "max_generation_retries": 1, "keyword_routing": False, **settings})
    g = build_graph(llm, llm, lambda q, k, s_=None: [DOC], classifier=classifier, settings=s)
    return DocuPilot(graph=g, settings=s)


def test_confident_classifier_skips_llm_router():
    llm = FakeLLM(route="direct")  # the LLM would say "direct"; the classifier wins
    res = _bot(llm, lambda q: ("retrieve", 0.95)).ask("How much earned leave?")
    assert "RouteDecision" not in llm.calls and res["route"] == "retrieve"
    assert any("trained router" in line for line in res["trace"])


def test_unsure_classifier_falls_back_to_llm_router():
    llm = FakeLLM(route="retrieve")
    res = _bot(llm, lambda q: ("direct", 0.41)).ask("How much earned leave?")
    assert "RouteDecision" in llm.calls and res["route"] == "retrieve"


# ---------- BM25 + fusion ----------

DOCS = [
    Document("C4.5 by Ross Quinlan is the best known decision tree system.", metadata={"source": "u5.pdf", "chunk": 1}),
    Document("Entropy measures impurity of a data set.", metadata={"source": "u5.pdf", "chunk": 2}),
    Document("Employees earn 18 days of earned leave.", metadata={"source": "leave.md", "chunk": 1}),
]


def test_tokenizer_keeps_codes_and_drops_stopwords():
    assert tokenize("What is C4.5 in section 3.4?") == ["c4.5", "section", "3.4"]


def test_bm25_ranks_exact_terms_and_filters_by_source():
    index = BM25Index(DOCS)
    top = index.search("Which system is C4.5?", k=3)
    assert top[0][0].metadata["chunk"] == 1 and top[0][0].metadata["source"] == "u5.pdf"
    assert index.search("earned leave", k=3, sources=["u5.pdf"]) == []
    assert index.search("zzz unknownword", k=3) == []


def test_rrf_merges_duplicates_even_when_pinecone_returns_floats():
    from_pinecone = Document("C4.5 ...", metadata={"source": "u5.pdf", "chunk": 1.0})
    fused = reciprocal_rank_fusion({"vector": [from_pinecone, DOCS[1]], "keyword": [DOCS[0]]})
    assert len(fused) == 2
    assert fused[0].metadata["found_by"] == ["vector", "keyword"]  # found by both → ranked first


def test_hybrid_retriever_finds_keyword_only_matches():
    class Index:
        def get(self):
            return BM25Index(DOCS)

    dense = lambda q, k, sources=None: [DOCS[1]]  # vector search misses the C4.5 chunk  # noqa: E731
    retrieve = make_hybrid_retriever(dense, Index(), Settings(fetch_k=5), reranker=lambda q, d, k: d[:k])
    found = retrieve("C4.5", 2)
    assert DOCS[0] in found and "keyword" in DOCS[0].metadata["found_by"]


def test_hybrid_can_be_switched_off():
    class Index:
        def get(self):
            raise AssertionError("keyword index must not be used")

    dense = lambda q, k, sources=None: [DOCS[1]]  # noqa: E731
    retrieve = make_hybrid_retriever(dense, Index(), Settings(hybrid_search=False), reranker=lambda q, d, k: d[:k])
    assert retrieve("C4.5", 2) == [DOCS[1]]


# ---------- slide decks ----------

def _pages(lengths):
    return [Document("x" * n, metadata={"source": "deck.pdf", "page": i}) for i, n in enumerate(lengths, 1)]


def test_short_pages_are_merged_with_page_ranges():
    merged = merge_short_pages(_pages([200, 200, 200, 200, 200, 2000, 100]), target=900)
    assert [(d.metadata["page"], d.metadata.get("page_end")) for d in merged] == [(1, 4), (5, None), (6, None), (7, None)]
    assert "[Page 2]" in merged[0].page_content


def test_merging_cuts_slide_deck_chunks():
    pages = [Document(f"Slide {i} about clustering and K-Means.", metadata={"source": "deck.pdf", "page": i})
             for i in range(1, 41)]
    assert len(chunk_documents(pages)) < 10  # 40 slides → a handful of context-rich chunks
    assert len(merge_short_pages(DOCS, 900)) == len(DOCS)  # non-PDF documents untouched


def test_labels_show_page_ranges_and_clean_numbers():
    assert _label(Document("t", metadata={"source": "a.pdf", "page": 3, "page_end": 5})) == "a.pdf · pp.3–5"
    assert _label(Document("t", metadata={"source": "a.pdf", "page": 19.0})) == "a.pdf · p.19"


def test_registry_stores_chunks_for_keyword_search(tmp_path):
    reg = DocumentRegistry(tmp_path)
    assert reg.load_chunks("ns") == [] and reg.chunks_signature("ns") == ()
    reg.save_chunks("ns", "u5.pdf", DOCS[:2])
    assert [d.page_content for d in reg.load_chunks("ns")] == [DOCS[0].page_content, DOCS[1].page_content]
    sig = reg.chunks_signature("ns")
    reg.remove("ns", "u5.pdf")
    assert reg.chunks_signature("ns") != sig and reg.load_chunks("ns") == []


def test_trace_reports_hybrid_mix():
    a = Document("C4.5 system", metadata={"source": "u5.pdf", "chunk": 1, "found_by": ["vector", "keyword"]})
    b = Document("entropy", metadata={"source": "u5.pdf", "chunk": 2, "found_by": ["keyword"]})
    s = Settings(max_rewrites=2, max_generation_retries=1, keyword_routing=False)
    g = build_graph(FakeLLM(relevant=(0, 1)), FakeLLM(relevant=(0, 1)), lambda q, k, s_=None: [a, b], settings=s)
    res = DocuPilot(graph=g, settings=s).ask("Which system is C4.5?")
    assert any("hybrid: 0 vector, 1 keyword, 1 both" in line for line in res["trace"])
