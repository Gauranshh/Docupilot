"""Turn provider/SDK exceptions into messages a user can act on, and decide what to retry."""
from __future__ import annotations

import re

TRANSIENT_MARKERS = (
    "429", "rate limit", "ratelimit", "resource_exhausted", "resource exhausted", "too many requests",
    "503", "unavailable", "overloaded", "500 internal", "timeout", "timed out", "connection reset",
    "connection aborted", "temporarily",
)
# Account has no money/credit at all (OpenAI style). Never retry.
CREDIT_MARKERS = ("insufficient_quota", "credit_balance_exhausted", "no credits", "credits are depleted")


def _low(exc: BaseException) -> str:
    return f"{type(exc).__name__} {exc}".lower()


def quota_kind(exc: BaseException) -> str | None:
    """Classify quota errors: 'credit' (no money), 'daily', 'no_free_tier', 'minute' or None."""
    low = _low(exc)
    if any(m in low for m in CREDIT_MARKERS):
        return "credit"
    if not any(m in low for m in ("429", "resource_exhausted", "quota")):
        return None
    if re.search(r"limit:\s*0\b", low):
        return "no_free_tier"
    if "perday" in low or "per day" in low or "requests_per_day" in low:
        return "daily"
    return "minute"


def is_transient(exc: BaseException) -> bool:
    """Retry per-minute rate limits and server hiccups — never credit, daily-quota, auth or not-found errors."""
    kind = quota_kind(exc)
    if kind in {"credit", "daily", "no_free_tier"}:
        return False
    return kind == "minute" or any(m in _low(exc) for m in TRANSIENT_MARKERS)


class DocuPilotError(Exception):
    """Raised for problems the user can fix (bad input, missing keys, unreadable files)."""


def friendly_error(exc: BaseException, model: str | None = None) -> str:
    if isinstance(exc, DocuPilotError):
        return str(exc)
    low = _low(exc)
    m = f"`{model}`" if model else "the model"
    kind = quota_kind(exc)
    if kind == "credit":
        return ("💳 Your AI provider account has no credit left. Add billing credit, or switch to the free "
                "Gemini tier (set GOOGLE_API_KEY, remove OPENAI_API_KEY in .env), then restart the app.")
    if kind == "daily":
        return (f"📅 You've used today's free-tier quota for {m}. It resets at midnight Pacific time "
                "(12:30 pm IST in summer, 1:30 pm IST in winter). Meanwhile, switch the chat model in the sidebar to another model with free quota "
                "(see the limits for your project at https://aistudio.google.com/rate-limit), or enable billing.")
    if kind == "no_free_tier":
        return (f"🚫 {m} has no free-tier quota on your account (limit 0). Choose a different model in the sidebar "
                "or enable billing in Google AI Studio.")
    if "not_found" in low or "404" in low or "no longer available" in low or "model_not_found" in low:
        return (f"🔎 The model {m} isn't available to your account. Pick another model in the sidebar or update "
                "LLM_MODEL in .env (e.g. google_genai:gemini-3.8-flash).")
    if any(k in low for k in ("401", "403", "api key", "api_key", "unauthorized", "permission_denied")):
        return ("🔑 An API key is missing or was rejected. Check PINECONE_API_KEY and GOOGLE_API_KEY / OPENAI_API_KEY "
                "in .env (no quotes or spaces), then restart the app.")
    if is_transient(exc):
        return "⏳ The AI provider is rate-limiting requests (common on free tiers). Wait about a minute and ask again."
    if "pinecone" in low:
        return "🌲 Pinecone error: " + str(exc)[:300]
    return f"⚠️ Something went wrong ({type(exc).__name__}): {str(exc)[:300]}"
