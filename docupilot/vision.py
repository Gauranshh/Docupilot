"""Read pages whose content is a picture (diagrams, tables, worked examples, scans).

Text extraction returns almost nothing for such pages (49 of the 97 slides in our Unit 5
deck). When enabled, each low-text PDF page is rendered to an image and Gemini — which can
read images — writes a factual description of it. The description is indexed like normal
text, so those slides become searchable and citable.

It costs one AI call per page, once, at ingest; `VISION_MAX_PAGES` caps it per file.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Callable

from langchain_core.messages import HumanMessage

from . import prompts
from .errors import is_transient, quota_kind

PageDescriber = Callable[[bytes, int, str], str]  # (png bytes, page number, file name) -> description


class VisionQuotaExhausted(Exception):
    """The AI provider refused more calls today; stop describing and keep the text we have."""


def render_page_png(pdf_path: Path, page_number: int, scale: float = 1.6) -> bytes:
    """Render one 1-based page of a PDF to PNG bytes."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        image = pdf[page_number - 1].render(scale=scale).to_pil()
        if image.width > 1600:  # keep requests small: enough for slide text and diagrams
            image = image.resize((1600, int(image.height * 1600 / image.width)))
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    finally:
        pdf.close()


def make_page_describer(llm) -> PageDescriber:
    """Wrap a multimodal chat model (Gemini by default) as a page describer."""
    retrying = llm.with_retry(retry_if_exception_type=is_transient, stop_after_attempt=4) \
        if hasattr(llm, "with_retry") else llm

    def describe(png: bytes, page: int, source: str) -> str:
        message = HumanMessage(content=[
            {"type": "text", "text": prompts.VISION_PROMPT.format(page=page, source=source)},
            {"type": "image", "base64": base64.b64encode(png).decode(), "mime_type": "image/png"},
        ])
        try:
            out = retrying.invoke([message])
        except Exception as e:  # noqa: BLE001
            if quota_kind(e) in {"daily", "credit", "no_free_tier"}:
                raise VisionQuotaExhausted(str(e)) from e
            raise
        content = out.content if isinstance(out.content, str) else "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in out.content)
        return content.strip()

    return describe
