"""Keyword search (BM25) and Reciprocal Rank Fusion for hybrid retrieval.

Vector search matches MEANING ("leave for new fathers" ≈ "paternity leave") but can miss
EXACT terms: codes, names, numbers ("C4.5", "B134", "Section 3.4"). BM25 scores exact word
matches. DocuPilot runs both and fuses the two ranked lists with Reciprocal Rank Fusion,
which needs no score calibration between the two systems:

    RRF(d) = Σ_lists 1 / (k + rank_in_list(d)),  k = 60
"""
from __future__ import annotations

import math
import re
from collections import Counter

from langchain_core.documents import Document

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i in is it its me my of on or "
    "our so than that the their them there these this to was we what when where which who why "
    "will with you your".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")  # keeps "c4.5" and "3.4" whole


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def as_int(value):
    """Pinecone returns stored numbers as floats (2.0); local chunks keep ints (2)."""
    try:
        f = float(value)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return value


def doc_key(doc: Document) -> str:
    m = doc.metadata
    return f"{m.get('source')}#{as_int(m.get('chunk'))}"


class BM25Index:
    """Okapi BM25 over a list of chunks (k1 = 1.5, b = 0.75)."""

    def __init__(self, docs: list[Document], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.tokens = [tokenize(d.page_content) for d in docs]
        self.tf = [Counter(t) for t in self.tokens]
        self.lengths = [len(t) for t in self.tokens]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if docs else 0.0
        df: Counter = Counter()
        for t in self.tokens:
            df.update(set(t))
        n = len(docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def __len__(self) -> int:
        return len(self.docs)

    def search(self, query: str, k: int, sources: list[str] | None = None) -> list[tuple[Document, float]]:
        terms = [t for t in tokenize(query) if t in self.idf]
        if not terms:
            return []
        scored = []
        for i, doc in enumerate(self.docs):
            if sources and doc.metadata.get("source") not in sources:
                continue
            tf, length = self.tf[i], self.lengths[i]
            score = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                    score += self.idf[t] * f * (self.k1 + 1) / norm
            if score > 0:
                scored.append((doc, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


def reciprocal_rank_fusion(ranked_lists: dict[str, list[Document]], k: int = 60, limit: int | None = None) -> list[Document]:
    """Merge several ranked lists; duplicates (same file + chunk) are combined.

    Each returned document carries `metadata["found_by"]`, e.g. ["vector", "keyword"].
    """
    scores: dict[str, float] = {}
    first: dict[str, Document] = {}
    found_by: dict[str, list[str]] = {}
    for name, docs in ranked_lists.items():
        for rank, doc in enumerate(docs, 1):
            key = doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            first.setdefault(key, doc)
            found_by.setdefault(key, [])
            if name not in found_by[key]:
                found_by[key].append(name)
    order = sorted(scores, key=scores.get, reverse=True)
    out = []
    for key in order[:limit] if limit else order:
        doc = first[key]
        doc.metadata["found_by"] = found_by[key]
        doc.metadata["rrf_score"] = round(scores[key], 5)
        out.append(doc)
    return out
