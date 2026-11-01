"""Streamlit chat UI.   Run:  streamlit run app.py"""
from __future__ import annotations

import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

from docupilot.config import get_settings
from docupilot.errors import friendly_error

STEP_LABELS = {
    "contextualize": "Understanding the question",
    "route": "Choosing a strategy",
    "load_documents": "Reading the whole document",
    "retrieve": "Searching Pinecone & reranking",
    "grade_documents": "Grading retrieved chunks",
    "rewrite_query": "Rewriting the search query",
    "web_search": "Falling back to web search",
    "generate": "Writing the answer",
    "check_grounding": "Checking answer against sources",
    "finalize": "Done",
    "direct_answer": "Replying",
    "no_answer": "No relevant sources found",
}
ROUTE_BADGES = {"retrieve": "🔎 Passage search", "document": "📖 Whole-document", "direct": "💬 Direct"}

st.set_page_config(page_title="DocuPilot", page_icon="🧭", layout="wide")
settings = get_settings()


# ---------- cached resources ----------
@st.cache_resource
def chat_store():
    from docupilot.chats import ChatStore
    return ChatStore()


@st.cache_resource(show_spinner="Connecting to Pinecone…")
def load_bot(namespace: str, model: str):
    from docupilot.graph import DocuPilot
    return DocuPilot(namespace=namespace, llm_model=model, checkpointer=chat_store().checkpointer())


SKIP_MODEL_WORDS = ("tts", "image", "audio", "live", "embedding", "aqa", "imagen", "veo", "robotics",
                    "computer-use", "native")


def _usable(name: str, methods) -> bool:
    return "generateContent" in (methods or []) and not any(k in name.lower() for k in SKIP_MODEL_WORDS)


@st.cache_data(ttl=3600, show_spinner=False)
def _list_gemini_models(api_key: str) -> list[str]:
    """Chat-capable Gemini models for this API key. Raises on failure, so failures are not cached."""
    import json
    import urllib.request

    names, errors = [], []
    # 1) Plain REST call (standard library only)
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000"
        req = urllib.request.Request(url, headers={"x-goog-api-key": api_key})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode("utf-8"))
        for m in data.get("models", []):
            n = m["name"].removeprefix("models/")
            if _usable(n, m.get("supportedGenerationMethods")):
                names.append(n)
    except Exception as ex:  # noqa: BLE001
        errors.append(f"REST: {type(ex).__name__}: {ex}")
    # 2) google-genai SDK as a fallback
    if not names:
        try:
            from google import genai

            for m in genai.Client(api_key=api_key).models.list():
                n = m.name.removeprefix("models/")
                if _usable(n, getattr(m, "supported_actions", None)):
                    names.append(n)
        except Exception as ex:  # noqa: BLE001
            errors.append(f"SDK: {type(ex).__name__}: {ex}")
    if not names:
        raise RuntimeError(" | ".join(errors) or "no chat models returned")
    names = [f"google_genai:{n}" for n in sorted(set(names))]
    return sorted(names, key=lambda n: ("gemini" not in n, "preview" in n or "exp" in n, n))


def available_models() -> tuple[list[str], str | None]:
    """(models for the dropdown, reason it failed). Empty list -> fall back to a text box."""
    import os

    key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip().strip('"')
    if not settings.llm_model.startswith("google_genai:"):
        return [], None
    if not key:
        return [], "No GOOGLE_API_KEY / GEMINI_API_KEY found in .env"
    try:
        names = list(_list_gemini_models(key))
    except Exception as ex:  # noqa: BLE001
        return [], str(ex)[:400]
    if settings.llm_model not in names:
        names.insert(0, settings.llm_model)
    return names, None


def registry():
    from docupilot.registry import DocumentRegistry
    return DocumentRegistry()


def new_conversation():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []


st.session_state.setdefault("thread_id", str(uuid.uuid4()))
st.session_state.setdefault("history", [])


# ---------- sidebar ----------
with st.sidebar:
    st.title("🧭 DocuPilot")
    st.caption("Agentic RAG · Pinecone + LangGraph")

    namespace = st.text_input("Knowledge base (Pinecone namespace)", settings.pinecone_namespace,
                              help="Keep separate document collections apart, e.g. 'handbook', 'assignment'.")
    available, model_error = available_models()
    if available:
        default = settings.llm_model if settings.llm_model in available else available[0]
        model = st.selectbox("Chat model", available, index=available.index(default),
                             help="Text models your API key can use. If one says its quota is used up, pick another.")
    else:
        model = st.text_input("Chat model", settings.llm_model,
                              help="Any 'provider:model' string, e.g. google_genai:gemini-3.8-flash or openai:gpt-4.1-mini")
        if model_error:
            st.caption(f"⚠️ Couldn't load the model list: {model_error}")
    if st.button("🔄 Refresh model list", use_container_width=True):
        _list_gemini_models.clear()
        st.rerun()

    # --- documents ---
    docs = registry().list(namespace)
    st.subheader(f"Documents ({len(docs)})")
    with st.expander("➕ Add documents", expanded=not docs):
        uploads = st.file_uploader("PDF, Markdown or text", type=["pdf", "md", "txt"], accept_multiple_files=True)
        if st.button("Ingest", disabled=not uploads, use_container_width=True, type="primary"):
            from docupilot.ingest import ingest_paths
            with tempfile.TemporaryDirectory() as tmp, st.status("Ingesting…", expanded=True) as status:
                try:
                    paths, names = [], []
                    for f in uploads:
                        p = Path(tmp) / f.name
                        p.write_bytes(f.getbuffer())
                        paths.append(p)
                        names.append(f.name)
                    report = ingest_paths(paths, namespace=namespace, names=names, progress=st.write)
                    for w in report["warnings"]:
                        st.warning(w)
                    for e in report["errors"]:
                        st.error(e)
                    ok = bool(report["files"])
                    status.update(label="Ingestion complete" if ok else "Nothing ingested",
                                  state="complete" if ok else "error")
                except Exception as e:  # noqa: BLE001
                    st.error(friendly_error(e))
                    status.update(label="Ingestion failed", state="error")

    for name, meta in docs.items():
        with st.expander(f"📄 {name}"):
            pages = f"{meta['pages']} pages · " if meta.get("pages") else ""
            st.caption(f"{pages}{meta['chunks']} chunks · added {meta['ingested_at'][:10]}")
            if meta.get("summary"):
                st.write(meta["summary"])
            if st.button("Delete", key=f"del-{name}", use_container_width=True):
                try:
                    from docupilot.ingest import delete_document
                    n = delete_document(name, namespace)
                    st.toast(f"Deleted {name} ({n} vectors)")
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(friendly_error(e))

    scope = st.multiselect("Search only in", list(docs), placeholder="All documents",
                           help="Restrict answers to specific files.") if docs else []

    # --- conversations ---
    st.divider()
    st.subheader("Conversations")
    if st.button("✏️ New conversation", use_container_width=True):
        new_conversation()
        st.rerun()
    for t in chat_store().recent(namespace, limit=8):
        c1, c2 = st.columns([5, 1])
        current = t["id"] == st.session_state.thread_id
        if c1.button(("▶ " if current else "") + (t["title"] or "Untitled"), key=f"t-{t['id']}",
                     use_container_width=True):
            st.session_state.thread_id = t["id"]
            try:
                st.session_state.history = [{**m, "result": None}
                                            for m in load_bot(namespace, model).history(t["id"])]
            except Exception as e:  # noqa: BLE001
                st.error(friendly_error(e))
            st.rerun()
        if c2.button("🗑", key=f"x-{t['id']}"):
            chat_store().delete(t["id"])
            if current:
                new_conversation()
            st.rerun()

    st.divider()
    show_trace = st.toggle("Show agent trace", value=True)
    with st.expander("Index stats"):
        try:
            from docupilot.vectorstore import index_stats
            st.json(index_stats(namespace))
        except Exception as e:  # noqa: BLE001
            st.error(friendly_error(e))
    st.caption(f"Embeddings: `{settings.embedding_model}`  \n"
               f"Reranker: {'`' + settings.rerank_model + '`' if settings.rerank else 'off'}  \n"
               f"Web fallback: {'on' if settings.web_enabled else 'off'}")


# ---------- chat ----------
def save_feedback(idx: int):
    from docupilot.chats import log_feedback
    turn = st.session_state.history[idx]
    rating = st.session_state.get(f"fb-{st.session_state.thread_id}-{idx}")
    question = st.session_state.history[idx - 1]["content"] if idx > 0 else ""
    res = turn.get("result") or {}
    log_feedback({"thread_id": st.session_state.thread_id, "namespace": namespace, "model": model,
                  "question": question, "answer": turn["content"], "rating": "up" if rating == 1 else "down",
                  "route": res.get("route"), "sources": [s["source"] for s in res.get("sources", [])]})
    st.toast("Thanks for the feedback!")


def render_extras(res: dict | None, idx: int) -> None:
    if not res:
        return
    if res.get("route") in ROUTE_BADGES:
        badge = ROUTE_BADGES[res["route"]]
        if res.get("grounded") is True:
            badge += "  ·  ✅ verified against sources"
        if res.get("used_web"):
            badge += "  ·  🌐 web results"
        st.caption(badge)
    cards = res.get("sources") or []
    if cards:
        cited = [c for c in cards if c.get("cited")]
        other = [c for c in cards if not c.get("cited")]
        with st.expander(f"📚 Sources ({len(cited) or len(cards)} cited)"):
            for c in cited or cards:
                score = f" · relevance {c['score']}" if c.get("score") is not None else ""
                icon = "🌐" if c["origin"] == "web" else "📄"
                st.markdown(f"**[{c['n']}] {icon} {c['label']}**{score}")
                st.caption(c["snippet"] + ("…" if len(c["snippet"]) >= 400 else ""))
            if cited and other:
                st.caption("Also retrieved but not cited: " + ", ".join(f"[{c['n']}] {c['label']}" for c in other))
    if show_trace and res.get("trace"):
        with st.expander("🔎 Agent trace"):
            for line in res["trace"]:
                st.markdown(f"- {line}")
    if not res.get("error"):
        st.feedback("thumbs", key=f"fb-{st.session_state.thread_id}-{idx}", on_change=save_feedback, args=(idx,))


if not st.session_state.history:
    st.markdown("### Ask anything about your documents")
    st.caption("Specific facts → *How many days of earned leave do I get?*  ·  "
               "Whole document → *Summarise the travel policy* or *How many sections does the leave policy have?*")

for i, turn in enumerate(st.session_state.history):
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant":
            render_extras(turn.get("result"), i)

if question := st.chat_input("Ask a question…"):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            bot = load_bot(namespace, model)
            with st.status("Thinking…") as status:
                res = bot.ask(question, thread_id=st.session_state.thread_id, sources=scope or None,
                              on_step=lambda node: status.update(label=STEP_LABELS.get(node, node)))
                status.update(label="Answered" if not res["error"] else "Failed",
                              state="complete" if not res["error"] else "error")
        except Exception as e:  # noqa: BLE001 — e.g. bad keys while building the bot
            res = {"answer": friendly_error(e, model), "error": str(e), "sources": [], "trace": []}
        if res.get("error"):
            st.error(res["answer"])
            with st.expander("Technical details"):
                st.code(str(res["error"])[:3000])
        else:
            st.markdown(res["answer"])
            chat_store().touch(st.session_state.thread_id, namespace,
                               next(t["content"] for t in st.session_state.history if t["role"] == "user"))
        st.session_state.history.append({"role": "assistant", "content": res["answer"], "result": res})
        if len(st.session_state.history) == 2 and not res.get("error"):
            st.rerun()  # first answer of a new conversation → refresh so it appears in the sidebar list
        render_extras(res, len(st.session_state.history) - 1)

if st.session_state.history:
    transcript = "\n\n".join(f"**{t['role'].title()}:** {t['content']}" for t in st.session_state.history)
    st.download_button("⬇️ Export conversation (.md)", transcript,
                       file_name=f"docupilot-chat-{datetime.now():%Y%m%d-%H%M}.md", mime="text/markdown")
