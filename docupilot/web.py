"""Optional web-search fallback used when the knowledge base has nothing relevant."""
from __future__ import annotations

from typing import Callable

from langchain_core.documents import Document

from .config import get_settings

WebSearch = Callable[[str], list[Document]]


def make_tavily_search(max_results: int = 4) -> WebSearch | None:
    s = get_settings()
    if not s.web_enabled:
        return None
    from tavily import TavilyClient

    client = TavilyClient(api_key=s.tavily_api_key)

    def search(query: str) -> list[Document]:
        res = client.search(query, max_results=max_results)
        return [
            Document(
                page_content=r.get("content", ""),
                metadata={"source": r.get("url", "web"), "title": r.get("title"), "origin": "web"},
            )
            for r in res.get("results", [])
            if r.get("content")
        ]

    return search
