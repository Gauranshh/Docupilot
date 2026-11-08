"""Text extraction for Office and web formats: Word, PowerPoint, Excel, CSV and HTML.

Each loader returns LangChain Documents with the same metadata the rest of the pipeline uses:
- PowerPoint: one Document per slide (`page` = slide number, `unit` = "slide"), slide title as a
  heading, text boxes, tables and speaker notes. Slides that are mostly pictures can be read by
  the AI model, like low-text PDF pages.
- Word: one Document; headings become markdown headings (so citations show the section), tables
  become rows of `cell | cell`. Embedded pictures can be read by the AI model.
- Excel: one Document per sheet with a `## Sheet: <name>` heading and one line per row.
- CSV: rows as `cell | cell`. HTML: visible text with headings kept.
"""
from __future__ import annotations

import csv
import io
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from .config import get_settings
from .errors import DocuPilotError

if TYPE_CHECKING:
    from .vision import PageDescriber

OFFICE = {".docx", ".pptx", ".xlsx", ".csv", ".html", ".htm"}
LEGACY = {".doc": ".docx", ".ppt": ".pptx", ".xls": ".xlsx"}
MIN_IMAGE_SIDE = 120          # smaller pictures are usually icons or logos
MAX_SHEET_ROWS = 5000         # keep huge spreadsheets bounded


def _row(cells) -> str:
    vals = [str(c).strip() for c in cells if c is not None and str(c).strip()]
    return " | ".join(vals)


def _to_png(blob: bytes) -> bytes | None:
    """Convert an embedded picture to PNG; skip tiny icons and formats Pillow cannot open (e.g. WMF)."""
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(blob))
        if min(image.size) < MIN_IMAGE_SIDE:
            return None
        if image.width > 1600:
            image = image.resize((1600, int(image.height * 1600 / image.width)))
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return None


class _Reader:
    """Shared budget and messages for AI picture reading within one file."""

    def __init__(self, name: str, describe: "PageDescriber | None"):
        self.name, self.describe = name, describe
        self.left = get_settings().vision_max_pages
        self.read = 0
        self.stopped = False
        self.warnings: list[str] = []

    def __call__(self, png: bytes, where: int) -> str | None:
        from .vision import VisionQuotaExhausted

        if not self.describe or self.stopped:
            return None
        if self.left <= 0:
            self.stopped = True
            self.warnings.append(f"{self.name}: only the first {get_settings().vision_max_pages} "
                                 "pictures were read (VISION_MAX_PAGES).")
            return None
        self.left -= 1
        try:
            text = self.describe(png, where, self.name)
        except VisionQuotaExhausted:
            self.stopped = True
            self.warnings.append(f"{self.name}: the AI quota ran out after reading {self.read} picture(s); "
                                 "the rest were indexed as text only. Re-ingest later to finish.")
            return None
        except Exception as e:  # noqa: BLE001 — one bad picture must not stop the file
            self.warnings.append(f"{self.name}: could not read a picture ({type(e).__name__}).")
            return None
        if not text or text.lower().startswith("blank page"):
            return None
        self.read += 1
        return text

    def finish(self) -> list[str]:
        if self.read:
            self.warnings.append(f"{self.name}: read {self.read} picture(s) with the AI model.")
        return self.warnings


# ---------- PowerPoint ----------

def _shape_texts(shape) -> tuple[list[str], list]:
    """Text lines and picture shapes inside a shape (groups are walked recursively)."""
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    lines, pictures = [], []
    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        for s in shape.shapes:
            sub_lines, sub_pics = _shape_texts(s)
            lines += sub_lines
            pictures += sub_pics
        return lines, pictures
    if shape.shape_type == MSO_SHAPE_TYPE.PICTURE or hasattr(shape, "image"):
        try:
            shape.image.blob  # picture placeholders may be empty
            pictures.append(shape)
        except Exception:  # noqa: BLE001
            pass
    if getattr(shape, "has_table", False) and shape.has_table:
        for r in shape.table.rows:
            line = _row(c.text for c in r.cells)
            if line:
                lines.append(line)
    elif getattr(shape, "has_text_frame", False) and shape.has_text_frame:
        for p in shape.text_frame.paragraphs:
            t = "".join(r.text for r in p.runs).strip()
            if t:
                lines.append(("  " * p.level) + t)
    return lines, pictures


def load_pptx(path: Path, name: str, describe_page: "PageDescriber | None" = None) -> tuple[list[Document], list[str]]:
    from pptx import Presentation

    try:
        prs = Presentation(str(path))
    except Exception as e:  # noqa: BLE001
        raise DocuPilotError(f"Could not read '{name}': {e}") from e
    reader = _Reader(name, describe_page)
    min_chars = get_settings().vision_min_chars
    docs = []
    for n, slide in enumerate(prs.slides, 1):
        title = slide.shapes.title.text.strip() if slide.shapes.title is not None and slide.shapes.title.has_text_frame else ""
        lines, pictures = [], []
        for shape in slide.shapes:
            if title and slide.shapes.title is not None and shape.shape_id == slide.shapes.title.shape_id:
                continue
            sub_lines, sub_pics = _shape_texts(shape)
            lines += sub_lines
            pictures += sub_pics
        body = "\n".join(lines)
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip() if slide.notes_slide.notes_text_frame else ""
            if notes:
                body += f"\n\nSpeaker notes: {notes}"
        text = (f"## {title}\n" if title else "") + body.strip()
        meta = {"source": name, "page": n, "unit": "slide"}
        if len(body.strip()) < min_chars and pictures:
            # Mostly a picture: describe the largest one
            biggest = max(pictures, key=lambda p: (p.width or 0) * (p.height or 0))
            png = _to_png(biggest.image.blob)
            description = reader(png, n) if png else None
            if description:
                text = (text + "\n\n" if text else "") + f"[Image description] {description}"
                meta["vision"] = True
        docs.append(Document(text.strip(), metadata=meta))
    warnings = reader.finish()
    empty = [d.metadata["page"] for d in docs if len(d.page_content) < 25]
    if empty:
        shown = ", ".join(map(str, empty[:10])) + ("…" if len(empty) > 10 else "")
        hint = "" if describe_page else " Tick “Read pictures” to read picture slides."
        warnings.append(f"{name}: {len(empty)} slide(s) had little or no text (slides {shown}).{hint}")
    return docs, warnings


# ---------- Word ----------

def load_docx(path: Path, name: str, describe_page: "PageDescriber | None" = None) -> tuple[list[Document], list[str]]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        d = docx.Document(str(path))
    except Exception as e:  # noqa: BLE001
        raise DocuPilotError(f"Could not read '{name}': {e}") from e
    reader = _Reader(name, describe_page)
    parts: list[str] = []
    number = 0  # running counter for numbered lists
    blip = "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"
    embed = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
    for el in d.element.body.iterchildren():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(el, d)
            t = p.text.strip()
            style = (p.style.name if p.style is not None else "") or ""
            number = number + 1 if t and style.startswith("List Number") else 0
            if t:
                if style.startswith("List Number"):
                    parts.append(f"{number}. {t}")
                elif style.startswith("Heading") and style[-1:].isdigit():
                    parts.append("#" * min(int(style[-1]), 4) + " " + t)
                elif style == "Title":
                    parts.append("# " + t)
                elif style.startswith("List"):
                    parts.append("• " + t)
                else:
                    parts.append(t)
            if describe_page is not None:
                for b in el.iter(blip):
                    rel = d.part.related_parts.get(b.get(embed))
                    png = _to_png(rel.blob) if rel is not None else None
                    text = reader(png, reader.read + 1) if png else None
                    if text:
                        parts.append(f"[Image description] {text}")
        elif tag == "tbl":
            rows = [_row(c.text for c in r.cells) for r in Table(el, d).rows]
            # merged cells repeat their text; drop exact duplicate neighbours
            parts.append("\n".join(r for i, r in enumerate(rows) if r and (i == 0 or r != rows[i - 1])))
    text = "\n\n".join(x for x in parts if x)
    return [Document(text, metadata={"source": name})], reader.finish()


# ---------- Excel / CSV ----------

def load_xlsx(path: Path, name: str) -> tuple[list[Document], list[str]]:
    from openpyxl import load_workbook

    try:
        wb = load_workbook(str(path), read_only=True, data_only=True)
    except Exception as e:  # noqa: BLE001
        raise DocuPilotError(f"Could not read '{name}': {e}") from e
    docs, warnings = [], []
    for ws in wb.worksheets:
        rows = []
        for i, r in enumerate(ws.iter_rows(values_only=True)):
            if i >= MAX_SHEET_ROWS:
                warnings.append(f"{name}: sheet '{ws.title}' has more than {MAX_SHEET_ROWS} rows; the rest were skipped.")
                break
            line = _row(r)
            if line:
                rows.append(line)
        if rows:
            docs.append(Document(f"## Sheet: {ws.title}\n" + "\n".join(rows), metadata={"source": name}))
    wb.close()
    return docs, warnings


def load_csv(path: Path, name: str) -> tuple[list[Document], list[str]]:
    raw = path.read_text(encoding="utf-8-sig", errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(raw[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = [_row(r) for r in csv.reader(io.StringIO(raw), dialect)]
    rows = [r for r in rows if r][:MAX_SHEET_ROWS]
    return [Document("\n".join(rows), metadata={"source": name})], []


# ---------- HTML ----------

class _HTMLText(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer"}
    BLOCK = {"p", "div", "li", "tr", "br", "section", "article", "table", "ul", "ol", "blockquote", "pre"}

    def __init__(self):
        super().__init__()
        self.out: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in {"h1", "h2", "h3", "h4"}:
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in self.BLOCK:
            self.out.append("\n")
        elif tag in {"td", "th"}:
            self.out.append(" | ")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        elif tag in {"h1", "h2", "h3", "h4"} or tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def load_html(path: Path, name: str) -> tuple[list[Document], list[str]]:
    parser = _HTMLText()
    parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
    lines = [" ".join(line.split()).strip(" |") for line in "".join(parser.out).splitlines()]
    text = "\n".join(line for line in lines if line.strip())
    return [Document(text, metadata={"source": name})], []


def load_office(path: Path, name: str, describe_page: "PageDescriber | None" = None) -> tuple[list[Document], list[str]]:
    suffix = path.suffix.lower()
    if suffix == ".pptx":
        return load_pptx(path, name, describe_page)
    if suffix == ".docx":
        return load_docx(path, name, describe_page)
    if suffix == ".xlsx":
        return load_xlsx(path, name)
    if suffix == ".csv":
        return load_csv(path, name)
    return load_html(path, name)
