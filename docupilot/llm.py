"""Model factories. Swap providers by changing LLM_MODEL / EMBEDDING_MODEL in .env."""
from __future__ import annotations

from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .config import get_settings


@lru_cache(maxsize=8)
def get_llm(model: str | None = None, temperature: float = 0.0) -> BaseChatModel:
    return init_chat_model(model or get_settings().llm_model, temperature=temperature)


def get_grader_llm(model: str | None = None) -> BaseChatModel:
    return get_llm(model or get_settings().grader_model)


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    return init_embeddings(get_settings().embedding_model)


@lru_cache(maxsize=1)
def embedding_dimension() -> int:
    """Discover the embedding size by embedding a probe string (works for any provider)."""
    return len(get_embeddings().embed_query("dimension probe"))
