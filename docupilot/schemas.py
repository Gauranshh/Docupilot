"""Structured-output schemas used by the LLM 'judges' inside the graph."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Where to send the user's question."""

    route: Literal["retrieve", "document", "direct"] = Field(
        description="'retrieve' for specific facts found in a few passages; 'document' for questions "
        "about a whole document (summaries, counts, overviews, effort estimates); 'direct' only for "
        "greetings, thanks, or questions about the assistant itself."
    )
    reason: str = Field(description="One short sentence explaining the choice.")


class StandaloneQuestion(BaseModel):
    """A follow-up question rewritten so it makes sense without the chat history."""

    question: str


class RelevanceGrades(BaseModel):
    """Which retrieved chunks actually help answer the question."""

    relevant_ids: list[int] = Field(
        description="IDs of the chunks that contain information useful for answering. Empty if none."
    )


class RewrittenQuery(BaseModel):
    """A better search query for the vector store."""

    query: str


class GroundingGrade(BaseModel):
    """Whether an answer is fully supported by the provided sources."""

    grounded: bool = Field(description="True only if every factual claim is supported by the sources.")
    unsupported_claims: list[str] = Field(default_factory=list)
