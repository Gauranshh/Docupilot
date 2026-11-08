"""The agentic RAG workflow, built as a LangGraph state machine.

    START
      │
  contextualize ──► route ──(direct)────► direct_answer ──────────────────────────► END
                      │ └──(document)──► load_documents ──► generate (whole-doc mode)
                  (retrieve)
                      ▼
       ┌──────► retrieve ──► grade_documents
       │      (filter+rerank)    │
       │      ┌──────────────────┼───────────────────┬──────────────┐
       │  (none relevant,    (relevant)        (none relevant,   (nothing left
       │   rewrites left)        │              web enabled)       to try)
       │      ▼                  ▼                   ▼                ▼
       └─ rewrite_query      generate ◄────── web_search          no_answer ──► END
                                 │
                          check_grounding ──(not grounded, retries left)──► generate
                                 │
                             finalize ──► END

Adaptive RAG (3-way routing), Corrective RAG (grade + rewrite + web fallback) and
Self-RAG (grounding check) in one graph. Conversation memory comes from the LangGraph
checkpointer, keyed by thread_id.
"""
from __future__ import annotations

import re
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from . import prompts
from .config import Settings, get_settings
from .errors import DocuPilotError, friendly_error, is_transient
from .keyword_search import as_int
from .schemas import GroundingGrade, RelevanceGrades, RewrittenQuery, RouteDecision, StandaloneQuestion
from .vectorstore import Retriever
from .web import WebSearch

HISTORY_WINDOW = 6  # messages of history used to resolve follow-ups
DocumentLoader = Callable[[list[str] | None], list[Document]]
DocumentLister = Callable[[], dict[str, dict]]
QuestionClassifier = Callable[[str], tuple[str, float]]  # question -> (route, confidence)


class RAGState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # persisted across turns
    question: str
    standalone_question: str
    search_query: str
    sources_filter: list[str] | None
    route: str
    documents: list[Document]
    rewrites: int
    generation_attempts: int
    answer: str
    sources: list[dict]
    grounded: bool | None
    unsupported: list[str]
    used_web: bool
    trace: list[str]


# ---------- helpers ----------

def _text(msg: BaseMessage) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def _label(doc: Document) -> str:
    m = doc.metadata
    parts = [str(m.get("title") or m.get("source", "unknown"))]
    if m.get("mode") == "full_document":
        parts.append("full document")
    elif m.get("mode") == "summary+excerpt":
        parts.append("summary + excerpt")
    if m.get("page"):
        first, last = as_int(m["page"]), as_int(m.get("page_end"))
        if m.get("unit") == "slide":
            parts.append(f"slides {first}–{last}" if last and last != first else f"slide {first}")
        else:
            parts.append(f"pp.{first}–{last}" if last and last != first else f"p.{first}")
    if m.get("section"):
        parts.append(f"§ {m['section']}")
    return " · ".join(parts)


def format_sources(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] ({_label(d)})\n{d.page_content}" for i, d in enumerate(docs, 1))


def cited_numbers(answer: str) -> set[int]:
    """Numbers used in citations like [1], [2][3] or [1, 4]."""
    nums: set[int] = set()
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer):
        nums.update(int(n) for n in re.split(r"\s*,\s*", group))
    return nums


def _source_cards(docs: list[Document], answer: str) -> list[dict]:
    cited = cited_numbers(answer)
    cards = [
        {
            "n": i,
            "label": _label(d),
            "source": d.metadata.get("source"),
            "page": as_int(d.metadata.get("page")),
            "found_by": d.metadata.get("found_by"),
            "score": d.metadata.get("rerank_score", d.metadata.get("score")),
            "origin": d.metadata.get("origin", "knowledge_base"),
            "cited": i in cited,
            "snippet": d.page_content[:400],
        }
        for i, d in enumerate(docs, 1)
    ]
    return sorted(cards, key=lambda c: (not c["cited"], c["n"]))


def _add_trace(state: RAGState, line: str) -> list[str]:
    return [*state.get("trace", []), line]


_DIRECT_RE = re.compile(
    r"^\s*(hi|hello|hey|hii+|yo|thanks|thank you|thx|ok|okay|good (morning|afternoon|evening)|bye)\b[\s!.?]*$", re.I)
_DOCUMENT_RE = re.compile(
    r"\b(summari[sz]e|summary|overview|outline|tl;?dr|gist|main points|key points|key takeaways|"
    r"what is (this|the) (pdf|document|file|doc) about|how many (questions|sections|pages|chapters|parts|tasks|items))\b",
    re.I)


def keyword_route(question: str) -> str | None:
    """Cheap pre-router for obvious cases; saves one LLM call per question. None = ask the LLM router."""
    if _DIRECT_RE.match(question):
        return "direct"
    if _DOCUMENT_RE.search(question):
        return "document"
    return None


def _retrying(runnable):
    """Retry rate limits / transient server errors with exponential backoff."""
    if hasattr(runnable, "with_retry"):
        return runnable.with_retry(retry_if_exception_type=is_transient, stop_after_attempt=4)
    return runnable


def _doc_listing(list_documents: DocumentLister | None) -> str:
    if not list_documents:
        return "(unknown)"
    docs = list_documents()
    if not docs:
        return "(none registered)"
    lines = []
    for name, meta in list(docs.items())[:30]:
        summary = (meta.get("summary") or "").replace("\n", " ")[:160]
        pages = f", {meta['pages']} pages" if meta.get("pages") else ""
        lines.append(f"- {name}{pages}: {summary}")
    return "\n".join(lines)


# ---------- graph ----------

def build_graph(
    llm: BaseChatModel,
    grader_llm: BaseChatModel,
    retriever: Retriever,
    web_search: WebSearch | None = None,
    document_loader: DocumentLoader | None = None,
    list_documents: DocumentLister | None = None,
    settings: Settings | None = None,
    checkpointer: Any | None = None,
    classifier: QuestionClassifier | None = None,
    on_llm_route: Callable[[str, str], None] | None = None,
):
    s = settings or get_settings()
    answer_llm = _retrying(llm)
    router = _retrying(grader_llm.with_structured_output(RouteDecision))
    condenser = _retrying(grader_llm.with_structured_output(StandaloneQuestion))
    grader = _retrying(grader_llm.with_structured_output(RelevanceGrades))
    rewriter = _retrying(grader_llm.with_structured_output(RewrittenQuery))
    verifier = _retrying(grader_llm.with_structured_output(GroundingGrade))

    def contextualize(state: RAGState) -> RAGState:
        q = state["question"]
        history = state.get("messages", [])[:-1][-HISTORY_WINDOW:]
        if not history:
            return {"standalone_question": q, "search_query": q, "trace": _add_trace(state, f"question: {q}")}
        out = condenser.invoke([SystemMessage(prompts.CONDENSE_SYSTEM), *history, HumanMessage(q)])
        return {
            "standalone_question": out.question,
            "search_query": out.question,
            "trace": _add_trace(state, f"follow-up resolved → {out.question}"),
        }

    def route(state: RAGState) -> RAGState:
        question = state["standalone_question"]
        shortcut = keyword_route(question) if s.keyword_routing else None
        predicted, confidence = classifier(question) if (classifier and not shortcut) else (None, 0.0)
        if shortcut:
            choice, note = shortcut, f"route: {shortcut} (keyword shortcut, no LLM call)"
        elif predicted and confidence >= s.router_min_confidence:
            choice, note = predicted, f"route: {predicted} (trained router, confidence {confidence:.2f}, no LLM call)"
        else:
            system = prompts.ROUTER_SYSTEM.format(documents=_doc_listing(list_documents))
            d = router.invoke([SystemMessage(system), HumanMessage(state["standalone_question"])])
            choice, note = d.route, f"route: {d.route} ({d.reason})"
            if on_llm_route:  # the trained router was unsure: keep the LLM's decision as a new example
                try:
                    on_llm_route(question, d.route)
                except Exception:  # noqa: BLE001 — learning must never break answering
                    pass
        if choice == "document" and (document_loader is None or not (list_documents and list_documents())):
            choice, note = "retrieve", note + " → no registered documents, using retrieval"
        return {"route": choice, "trace": _add_trace(state, note)}

    def load_documents(state: RAGState) -> RAGState:
        docs = document_loader(state.get("sources_filter")) if document_loader else []
        modes = {d.metadata.get("mode") for d in docs}
        return {"documents": docs,
                "trace": _add_trace(state, f"loaded {len(docs)} whole document(s) ({', '.join(filter(None, modes))})")}

    def after_load(state: RAGState) -> str:
        return "generate" if state.get("documents") else "retrieve"

    def retrieve(state: RAGState) -> RAGState:
        flt = state.get("sources_filter")
        docs = retriever(state["search_query"], s.top_k, flt)
        scope = f" in {flt}" if flt else ""
        how = [tuple(d.metadata.get("found_by") or ()) for d in docs]
        mix = ""
        if any("keyword" in h for h in how):
            both = sum(1 for h in how if len(h) > 1)
            kw = sum(1 for h in how if h == ("keyword",))
            mix = f" · hybrid: {len(docs) - both - kw} vector, {kw} keyword, {both} both"
        return {"documents": docs,
                "trace": _add_trace(state, f"retrieved {len(docs)} chunks for '{state['search_query']}'{scope}{mix}")}

    def grade_documents(state: RAGState) -> RAGState:
        docs = state.get("documents", [])
        if not docs:
            return {"documents": [], "trace": _add_trace(state, "graded: 0 relevant (nothing retrieved)")}
        listing = "\n\n".join(f"<chunk id={i}>\n{d.page_content}\n</chunk>" for i, d in enumerate(docs))
        grades = grader.invoke([
            SystemMessage(prompts.GRADER_SYSTEM),
            HumanMessage(f"Question: {state['standalone_question']}\n\n{listing}"),
        ])
        keep = [docs[i] for i in sorted(set(grades.relevant_ids)) if 0 <= i < len(docs)]
        return {"documents": keep, "trace": _add_trace(state, f"graded: {len(keep)}/{len(docs)} relevant")}

    def after_grading(state: RAGState) -> str:
        if state.get("documents"):
            return "generate"
        if state.get("rewrites", 0) < s.max_rewrites:
            return "rewrite_query"
        if web_search is not None and not state.get("used_web"):
            return "web_search"
        return "no_answer"

    def rewrite_query(state: RAGState) -> RAGState:
        out = rewriter.invoke([
            SystemMessage(prompts.REWRITE_SYSTEM),
            HumanMessage(f"Question: {state['standalone_question']}\nPrevious query: {state['search_query']}"),
        ])
        n = state.get("rewrites", 0) + 1
        return {"search_query": out.query, "rewrites": n, "trace": _add_trace(state, f"rewrite #{n}: {out.query}")}

    def web(state: RAGState) -> RAGState:
        docs = web_search(state["standalone_question"]) if web_search else []
        return {"documents": docs, "used_web": True, "trace": _add_trace(state, f"web search: {len(docs)} results")}

    def after_web(state: RAGState) -> str:
        return "generate" if state.get("documents") else "no_answer"

    def generate(state: RAGState) -> RAGState:
        docs = state["documents"]
        extra = ""
        if state.get("unsupported"):
            claims = "\n".join(f"  • {c}" for c in state["unsupported"])
            extra += prompts.STRICT_ADDENDUM.format(claims=claims)
        if state.get("used_web"):
            extra += prompts.WEB_ADDENDUM
        if state.get("route") == "document":
            extra += prompts.DOCUMENT_ADDENDUM
        msg = answer_llm.invoke([
            SystemMessage(prompts.ANSWER_SYSTEM.format(strict=extra)),
            HumanMessage(f"SOURCES:\n{format_sources(docs)}\n\nQUESTION: {state['standalone_question']}"),
        ])
        n = state.get("generation_attempts", 0) + 1
        return {"answer": _text(msg).strip(), "generation_attempts": n,
                "trace": _add_trace(state, f"generated answer (attempt {n})")}

    def check_grounding(state: RAGState) -> RAGState:
        g = verifier.invoke([
            SystemMessage(prompts.GROUNDING_SYSTEM),
            HumanMessage(f"SOURCES:\n{format_sources(state['documents'])}\n\nANSWER:\n{state['answer']}"),
        ])
        note = "grounded ✓" if g.grounded else f"NOT grounded: {g.unsupported_claims}"
        return {"grounded": g.grounded, "unsupported": g.unsupported_claims, "trace": _add_trace(state, note)}

    def after_generate(state: RAGState) -> str:
        if not s.verify_answers:
            return "finalize"
        if state.get("route") == "document" and not s.verify_document_answers:
            return "finalize"  # verifying would resend the whole document — too costly on free tiers
        return "check_grounding"

    def after_grounding(state: RAGState) -> str:
        if state.get("grounded") or state.get("generation_attempts", 0) > s.max_generation_retries:
            return "finalize"
        return "generate"

    def finalize(state: RAGState) -> RAGState:
        answer = state["answer"]
        if state.get("grounded") is False:
            answer += "\n\n> ⚠️ Parts of this answer could not be fully verified against the sources."
        return {"answer": answer, "sources": _source_cards(state["documents"], answer),
                "messages": [AIMessage(answer)]}

    def direct_answer(state: RAGState) -> RAGState:
        history = state.get("messages", [])[-HISTORY_WINDOW:]
        system = prompts.DIRECT_SYSTEM.format(documents=_doc_listing(list_documents))
        msg = answer_llm.invoke([SystemMessage(system), *history])
        answer = _text(msg).strip()
        return {"answer": answer, "sources": [], "grounded": None, "messages": [AIMessage(answer)]}

    def no_answer(state: RAGState) -> RAGState:
        return {
            "answer": prompts.NO_ANSWER,
            "sources": [],
            "grounded": None,
            "messages": [AIMessage(prompts.NO_ANSWER)],
            "trace": _add_trace(state, "no relevant sources → declined to answer"),
        }

    g = StateGraph(RAGState)
    for name, fn in [
        ("contextualize", contextualize), ("route", route), ("load_documents", load_documents),
        ("retrieve", retrieve), ("grade_documents", grade_documents), ("rewrite_query", rewrite_query),
        ("web_search", web), ("generate", generate), ("check_grounding", check_grounding),
        ("finalize", finalize), ("direct_answer", direct_answer), ("no_answer", no_answer),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "contextualize")
    g.add_edge("contextualize", "route")
    g.add_conditional_edges("route", lambda st: st["route"],
                            {"retrieve": "retrieve", "document": "load_documents", "direct": "direct_answer"})
    g.add_conditional_edges("load_documents", after_load, ["generate", "retrieve"])
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges("grade_documents", after_grading,
                            ["generate", "rewrite_query", "web_search", "no_answer"])
    g.add_edge("rewrite_query", "retrieve")
    g.add_conditional_edges("web_search", after_web, ["generate", "no_answer"])
    g.add_conditional_edges("generate", after_generate, ["check_grounding", "finalize"])
    g.add_conditional_edges("check_grounding", after_grounding, ["finalize", "generate"])
    g.add_edge("finalize", END)
    g.add_edge("direct_answer", END)
    g.add_edge("no_answer", END)

    return g.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


# ---------- convenience wrapper ----------

def _load_classifier(settings: Settings) -> QuestionClassifier | None:
    if not settings.trained_router:
        return None
    from .router_model import TrainedRouter

    model = TrainedRouter.load()
    return model.predict if model else None


class DocuPilot:
    """High-level API used by the CLI, the Streamlit app and the eval script."""

    def __init__(self, graph=None, namespace: str | None = None, llm_model: str | None = None,
                 checkpointer: Any | None = None, settings: Settings | None = None,
                 cache=None, kb_signature: Callable[[], str] | None = None):
        self.settings = settings or get_settings()
        self.model = llm_model or self.settings.llm_model
        self.namespace = namespace or self.settings.pinecone_namespace
        self.cache = cache
        self.kb_signature = kb_signature or (lambda: "")
        if graph is None:
            from .llm import get_grader_llm, get_llm
            from .registry import DocumentRegistry, make_document_loader
            from .router_model import record_learned
            from .vectorstore import make_pinecone_retriever
            from .web import make_tavily_search

            ns = self.namespace
            registry = DocumentRegistry()
            graph = build_graph(
                get_llm(self.model),
                get_grader_llm(llm_model),
                make_pinecone_retriever(ns),
                make_tavily_search(),
                document_loader=make_document_loader(ns, registry),
                list_documents=lambda: registry.list(ns),
                checkpointer=checkpointer,
                classifier=_load_classifier(self.settings),
                on_llm_route=record_learned if self.settings.router_learning else None,
            )
            if cache is None and self.settings.answer_cache:
                from .cache import AnswerCache
                self.cache = AnswerCache(self.settings.data_dir / "cache.sqlite", self.settings.cache_ttl_hours)
            if kb_signature is None:
                self.kb_signature = lambda: registry.signature(ns)
        self.graph = graph

    def _cache_key(self, question: str, sources: list[str] | None) -> str:
        from .cache import cache_key
        return cache_key(question, self.namespace, self.model, sources, self.kb_signature())

    def ask(self, question: str, thread_id: str = "default", sources: list[str] | None = None,
            on_step: Callable[[str], None] | None = None,
            on_token: Callable[[str], None] | None = None) -> dict:
        """Answer one question. `on_step(node)` reports progress; `on_token(text_so_far)` streams the answer."""
        question = (question or "").strip()
        result = {"answer": "", "sources": [], "grounded": None, "used_web": False,
                  "route": None, "trace": [], "error": None, "cached": False}
        if not question:
            result.update(answer="Please type a question.", error="empty question")
            return result
        if len(question) > self.settings.max_question_chars:
            msg = f"That question is too long (max {self.settings.max_question_chars} characters). Please shorten it."
            result.update(answer=msg, error="question too long")
            return result

        config = {"configurable": {"thread_id": thread_id}}
        first_turn = not self.graph.get_state(config).values.get("messages")
        key = self._cache_key(question, sources) if (self.cache is not None and first_turn) else None
        if key:
            hit = self.cache.get(key)
            if hit:
                try:  # keep the conversation history consistent, as if the graph had answered
                    self.graph.update_state(config, {"messages": [HumanMessage(question), AIMessage(hit["answer"])]},
                                            as_node="finalize")
                except Exception:  # noqa: BLE001
                    pass
                result.update(hit, cached=True,
                              trace=["answer reused from cache: same question, same documents (no AI calls)"])
                if on_token:
                    on_token(hit["answer"])
                return result

        turn_input: RAGState = {
            "messages": [HumanMessage(question)],
            "question": question,
            "sources_filter": list(sources) if sources else None,
            "documents": [],
            "rewrites": 0,
            "generation_attempts": 0,
            "unsupported": [],
            "used_web": False,
            "grounded": None,
            "answer": "",
            "sources": [],
            "trace": [],
        }
        streamed, step = "", None
        try:
            for mode, payload in self.graph.stream(turn_input, config, stream_mode=["updates", "messages"]):
                if mode == "updates" and on_step:
                    for node in payload:
                        on_step(node)
                elif mode == "messages" and on_token:
                    chunk, meta = payload
                    if meta.get("langgraph_node") not in STREAMED_NODES:
                        continue
                    if meta.get("langgraph_step") != step:  # a regenerated answer starts over
                        step, streamed = meta.get("langgraph_step"), ""
                    piece = _text(chunk) if hasattr(chunk, "content") else ""
                    if piece:
                        streamed += piece
                        on_token(streamed)
        except Exception as e:  # noqa: BLE001 — never show a raw traceback to the user
            result.update(answer=friendly_error(e, self.model), error=f"{type(e).__name__}: {e}")
            if not isinstance(e, DocuPilotError):
                # keep the conversation consistent: record that this turn failed
                try:
                    self.graph.update_state(config, {"messages": [AIMessage("(error — no answer)")]})
                except Exception:  # noqa: BLE001
                    pass
            return result

        final = self.graph.get_state(config).values
        result.update(
            answer=final.get("answer", ""),
            sources=final.get("sources", []),
            grounded=final.get("grounded"),
            used_web=final.get("used_web", False),
            route=final.get("route"),
            trace=final.get("trace", []),
        )
        if key and _cacheable(result):
            self.cache.put(key, question, result)
        return result

    def history(self, thread_id: str) -> list[dict]:
        """Messages of a saved conversation, for restoring the chat UI."""
        values = self.graph.get_state({"configurable": {"thread_id": thread_id}}).values
        out = []
        for m in values.get("messages", []):
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            out.append({"role": role, "content": _text(m)})
        return out


STREAMED_NODES = {"generate", "direct_answer"}


def _cacheable(result: dict) -> bool:
    """Only answers grounded in the documents are worth reusing."""
    return (not result["error"] and result["route"] in {"retrieve", "document"}
            and result["grounded"] is not False and not result["used_web"]
            and bool(result["sources"]) and result["answer"] != prompts.NO_ANSWER)
