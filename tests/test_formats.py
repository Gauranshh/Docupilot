"""Word, PowerPoint, Excel, CSV and HTML files are read with the same metadata as PDFs."""
from __future__ import annotations

import io

import pytest
from langchain_core.documents import Document
from PIL import Image

from docupilot.errors import DocuPilotError
from docupilot.graph import _label
from docupilot.ingest import SUPPORTED, chunk_documents, load_file
from docupilot.vision import VisionQuotaExhausted


def _png(w=400, h=300, color=(30, 90, 160)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def _pptx(path, picture_slide=True):
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    s1 = prs.slides.add_slide(prs.slide_layouts[1])
    s1.shapes.title.text = "Decision trees"
    s1.placeholders[1].text = "C4.5 by Ross Quinlan is the best-known decision tree system.\nFinding the best tree is NP-hard."
    s1.notes_slide.notes_text_frame.text = "Mention information gain."
    s2 = prs.slides.add_slide(prs.slide_layouts[5])
    s2.shapes.title.text = "Loan data"
    rows = s2.shapes.add_table(3, 2, Inches(1), Inches(2), Inches(6), Inches(1.5)).table
    for r, (a, b) in enumerate([("Age", "Approved"), ("young", "no"), ("old", "yes")]):
        rows.cell(r, 0).text, rows.cell(r, 1).text = a, b
    if picture_slide:
        s3 = prs.slides.add_slide(prs.slide_layouts[6])  # blank layout, picture only
        s3.shapes.add_picture(io.BytesIO(_png()), Inches(1), Inches(1), Inches(5))
    prs.save(path)
    return path


def _docx(path, with_picture=False):
    import docx
    from docx.shared import Inches

    d = docx.Document()
    d.add_heading("Leave Policy", 1)
    d.add_paragraph("Employees get 18 days of earned leave per year.")
    d.add_heading("Sick leave", 2)
    d.add_paragraph("12 days of sick leave; a certificate is needed after 2 days.")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Leave type", "Days"
    t.cell(1, 0).text, t.cell(1, 1).text = "Casual", "8"
    if with_picture:
        d.add_picture(io.BytesIO(_png()), width=Inches(3))
    d.save(path)
    return path


def test_supported_formats():
    assert {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".html", ".md", ".txt"} <= SUPPORTED


def test_pptx_one_document_per_slide_with_title_table_and_notes(tmp_path):
    docs, warnings = load_file(_pptx(tmp_path / "u5.pptx"))
    assert [d.metadata["page"] for d in docs] == [1, 2]  # slide 3 is only a picture: no text
    assert all(d.metadata["unit"] == "slide" for d in docs)
    assert docs[0].page_content.startswith("## Decision trees")
    assert "C4.5" in docs[0].page_content and "Speaker notes: Mention information gain." in docs[0].page_content
    assert "young | no" in docs[1].page_content
    assert any("little or no text (slides 3)" in w for w in warnings)


def test_pptx_picture_slide_is_read_when_enabled(tmp_path):
    seen = []
    docs, warnings = load_file(_pptx(tmp_path / "u5.pptx"),
                               describe_page=lambda png, n, src: seen.append(n) or "Bar chart of accuracy by depth")
    assert seen == [3]
    assert docs[2].metadata["vision"] and "[Image description] Bar chart" in docs[2].page_content
    assert any("read 1 picture(s)" in w for w in warnings)


def test_slide_citations_say_slide(tmp_path):
    docs, _ = load_file(_pptx(tmp_path / "deck.pptx", picture_slide=False))
    chunks = chunk_documents(docs)  # two short slides are merged into one chunk
    assert len(chunks) == 1 and "[Slide 2]" in chunks[0].page_content
    assert _label(chunks[0]) == "deck.pptx · slides 1–2 · § Decision trees"
    assert _label(Document("x", metadata={"source": "a.pdf", "page": 4})) == "a.pdf · p.4"


def test_docx_headings_and_tables(tmp_path):
    docs, warnings = load_file(_docx(tmp_path / "leave.docx"))
    text = docs[0].page_content
    assert "# Leave Policy" in text and "## Sick leave" in text and "Casual | 8" in text
    chunks = chunk_documents(docs)
    assert chunks[0].metadata["section"] == "Leave Policy" and not warnings


def test_docx_pictures_are_read_and_quota_stops_cleanly(tmp_path):
    path = _docx(tmp_path / "p.docx", with_picture=True)
    docs, warnings = load_file(path, describe_page=lambda png, n, src: "Org chart: HR reports to the CEO")
    assert "[Image description] Org chart" in docs[0].page_content

    def out_of_quota(png, n, src):
        raise VisionQuotaExhausted("daily")

    docs, warnings = load_file(path, describe_page=out_of_quota)
    assert "Casual | 8" in docs[0].page_content and any("quota ran out" in w for w in warnings)


def test_xlsx_sheets(tmp_path):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Hotels"
    ws.append(["City", "Limit (INR)"])
    ws.append(["Mumbai", 9000])
    wb.create_sheet("Empty")
    wb.save(tmp_path / "rates.xlsx")
    docs, _ = load_file(tmp_path / "rates.xlsx")
    assert len(docs) == 1 and docs[0].page_content == "## Sheet: Hotels\nCity | Limit (INR)\nMumbai | 9000"


def test_csv_and_html(tmp_path):
    (tmp_path / "a.csv").write_text("city;limit\nMumbai;9000\n", encoding="utf-8")
    assert load_file(tmp_path / "a.csv")[0][0].page_content == "city | limit\nMumbai | 9000"
    (tmp_path / "p.html").write_text(
        "<html><head><title>x</title><style>p{}</style></head><body><nav>Menu</nav>"
        "<h1>Travel</h1><p>Hotel limit is 9000.</p><table><tr><td>Mumbai</td><td>9000</td></tr></table>"
        "<script>alert(1)</script></body></html>", encoding="utf-8")
    text = load_file(tmp_path / "p.html")[0][0].page_content
    assert text.startswith("# Travel") and "Hotel limit is 9000." in text and "Mumbai | 9000" in text
    assert "Menu" not in text and "alert" not in text


def test_old_office_formats_get_a_clear_message(tmp_path):
    (tmp_path / "old.ppt").write_bytes(b"\xd0\xcf\x11\xe0")
    with pytest.raises(DocuPilotError, match="Save As → .pptx"):
        load_file(tmp_path / "old.ppt")
