"""Pinecone index management and the retriever used by the graph.

Retrieval = dense similarity search (FETCH_K candidates, optional metadata filter by file)
          → Pinecone hosted reranker (cross-encoder) → TOP_K best chunks.
"""
from __future__ import annotations

import time
from functools import lru_cache
from typing import Protocol

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from .config import get_settings
from .errors import DocuPilotError
from .llm import embedding_dimension, get_embeddings


class Retriever(Protocol):
    """Anything that turns a query into documents. Lets tests inject a fake."""

    def __call__(self, query: str, k: int, sources: list[str] | None = None) -> list[Document]: ...


@lru_cache(maxsize=1)
def get_pinecone() -> Pinecone:
    s = get_settings()
    if not s.pinecone_api_key:
        raise DocuPilotError("PINECONE_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return Pinecone(api_key=s.pinecone_api_key)


@lru_cache(maxsize=1)
def ensure_index():
    """Create the serverless index on first run, then return a handle to it."""
    s = get_settings()
    pc = get_pinecone()
    if not pc.has_index(s.pinecone_index):
        dim = embedding_dimension()
        print(f"Creating Pinecone index '{s.pinecone_index}' (dim={dim}, cosine, {s.pinecone_cloud}/{s.pinecone_region})")
        pc.create_index(
            name=s.pinecone_index,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=s.pinecone_cloud, region=s.pinecone_region),
        )
        while not pc.describe_index(s.pinecone_index).status["ready"]:
            time.sleep(1)
    else:
        existing = pc.describe_index(s.pinecone_index).dimension
        if existing != embedding_dimension():
            raise DocuPilotError(
                f"Index '{s.pinecone_index}' has dimension {existing} but the embedding model "
                f"'{s.embedding_model}' produces {embedding_dimension()}. Set a new PINECONE_INDEX name in .env."
            )
    return pc.Index(s.pinecone_index)


def get_vectorstore(namespace: str | None = None) -> PineconeVectorStore:
    s = get_settings()
    return PineconeVectorStore(
        index=ensure_index(),
        embedding=get_embeddings(),
        namespace=namespace or s.pinecone_namespace,
    )


def rerank(query: str, docs: list[Document], top_n: int) -> list[Document]:
    """Cross-encoder reranking with Pinecone Inference. Falls back to vector order on failure."""
    s = get_settings()
    if not docs:
        return docs
    try:
        res = get_pinecone().inference.rerank(
            model=s.rerank_model,
            query=query,
            documents=[{"id": str(i), "text": d.page_content} for i, d in enumerate(docs)],
            top_n=min(top_n, len(docs)),
            return_documents=False,
        )
        ranked = []
        for row in res.data:
            d = docs[int(row.index)]
            d.metadata["rerank_score"] = round(float(row.score), 4)
            ranked.append(d)
        return ranked
    except Exception as e:  # noqa: BLE001
        print(f"(rerank skipped: {type(e).__name__}: {str(e)[:120]})")
        return docs[:top_n]


def make_pinecone_retriever(namespace: str | None = None) -> Retriever:
    """Similarity search (+ optional source filter) → rerank; keeps scores for transparency."""
    store = get_vectorstore(namespace)
    s = get_settings()

    def retrieve(query: str, k: int, sources: list[str] | None = None) -> list[Document]:
        flt = {"source": {"$in": list(sources)}} if sources else None
        fetch = max(s.fetch_k, k) if s.rerank else k
        results = store.similarity_search_with_score(query, k=fetch, filter=flt)
        docs = []
        for doc, score in results:
            if score < s.min_score:
                continue
            doc.metadata["score"] = round(float(score), 4)
            docs.append(doc)
        return rerank(query, docs, k) if s.rerank else docs[:k]

    return retrieve


def index_stats(namespace: str | None = None) -> dict:
    s = get_settings()
    stats = ensure_index().describe_index_stats()
    ns = (stats.get("namespaces") or {}).get(namespace or s.pinecone_namespace, {})
    return {
        "index": s.pinecone_index,
        "namespace": namespace or s.pinecone_namespace,
        "vectors_in_namespace": ns.get("vector_count", 0),
        "total_vectors": stats.get("total_vector_count", 0),
        "dimension": stats.get("dimension"),
    }
