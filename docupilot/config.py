"""Central configuration, read from environment variables / .env.

Provider auto-detection: if only GOOGLE_API_KEY is set, Gemini models are used by default;
otherwise OpenAI. Any value can be overridden explicitly in .env.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULTS = {
    "google": {
        "llm": "google_genai:gemini-3.8-flash",
        "embedding": "google_genai:models/gemini-embedding-001",
    },
    "openai": {
        "llm": "openai:gpt-4.1-mini",
        "embedding": "openai:text-embedding-3-small",
    },
}


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env(name: str, default: str) -> str:
    val = os.getenv(name)
    return val.strip() if val and val.strip() else default


def detect_provider() -> str:
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return "google"
    return "openai"


@dataclass(frozen=True)
class Settings:
    # Models — "provider:model" strings understood by LangChain's init_chat_model / init_embeddings.
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", DEFAULTS[detect_provider()]["llm"]))
    grader_model: str = field(default_factory=lambda: _env("GRADER_MODEL", _env("LLM_MODEL", DEFAULTS[detect_provider()]["llm"])))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", DEFAULTS[detect_provider()]["embedding"]))

    # Pinecone
    pinecone_api_key: str | None = field(default_factory=lambda: os.getenv("PINECONE_API_KEY"))
    pinecone_index: str = field(default_factory=lambda: _env("PINECONE_INDEX", "docupilot"))
    pinecone_namespace: str = field(default_factory=lambda: _env("PINECONE_NAMESPACE", "handbook"))
    pinecone_cloud: str = field(default_factory=lambda: _env("PINECONE_CLOUD", "aws"))
    pinecone_region: str = field(default_factory=lambda: _env("PINECONE_REGION", "us-east-1"))

    # Chunking & retrieval
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "150")))
    fetch_k: int = field(default_factory=lambda: int(_env("FETCH_K", "20")))      # candidates from Pinecone
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))           # kept after reranking
    min_score: float = field(default_factory=lambda: float(_env("MIN_SCORE", "0.0")))
    rerank: bool = field(default_factory=lambda: _bool("RERANK", True))
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL", "bge-reranker-v2-m3"))

    # Whole-document mode: max characters of full text sent to the LLM (~4 chars per token)
    full_doc_char_limit: int = field(default_factory=lambda: int(_env("FULL_DOC_CHAR_LIMIT", "300000")))
    summarize_on_ingest: bool = field(default_factory=lambda: _bool("SUMMARIZE_ON_INGEST", True))

    # Agent behaviour
    max_rewrites: int = field(default_factory=lambda: int(_env("MAX_REWRITES", "2")))
    max_generation_retries: int = field(default_factory=lambda: int(_env("MAX_GENERATION_RETRIES", "1")))
    max_question_chars: int = field(default_factory=lambda: int(_env("MAX_QUESTION_CHARS", "2000")))
    # Cost controls (important on free tiers with ~20 requests/day):
    keyword_routing: bool = field(default_factory=lambda: _bool("KEYWORD_ROUTING", True))   # skip router LLM for obvious cases
    verify_answers: bool = field(default_factory=lambda: _bool("VERIFY_ANSWERS", True))     # Self-RAG grounding check
    verify_document_answers: bool = field(default_factory=lambda: _bool("VERIFY_DOCUMENT_ANSWERS", False))

    # Optional web-search fallback (Tavily). Off automatically when no key is set.
    tavily_api_key: str | None = field(default_factory=lambda: os.getenv("TAVILY_API_KEY"))
    web_fallback: bool = field(default_factory=lambda: _bool("WEB_FALLBACK", True))

    # Local storage: document registry, full texts, chat history, feedback
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "storage")))

    @property
    def web_enabled(self) -> bool:
        return bool(self.web_fallback and self.tavily_api_key)


def get_settings() -> Settings:
    return Settings()
