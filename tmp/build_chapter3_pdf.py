from __future__ import annotations

import html
import re
import subprocess
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "CHAPTER_3.md"
OUTPUT = ROOT / "verification" / "chapter_3.pdf"
TMP = ROOT / "tmp" / "pdfs"
DIAGRAMS = ROOT / "assets" / "chapter_3" / "diagrams"


def register_fonts() -> None:
    regular = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
    bold = "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"
    italic = "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf"
    pdfmetrics.registerFont(TTFont("TNR", regular))
    pdfmetrics.registerFont(TTFont("TNR-Bold", bold))
    pdfmetrics.registerFont(TTFont("TNR-Italic", italic))


def clean_inline(text: str) -> str:
    text = text.strip()
    text = re.sub(r"!\[[^]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("`", "")
    text = text.replace("**", "")
    text = text.replace("*", "")
    text = text.replace("—", "–")
    text = html.escape(text)
    return text


def make_diagram_png(stem: str, width: int) -> Path:
    target = TMP / f"{stem}.png"
    subprocess.run(
        ["rsvg-convert", "-w", str(width), str(DIAGRAMS / f"{stem}.svg"), "-o", str(target)],
        check=True,
    )
    return target


def page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("TNR", 9)
    canvas.setFillColor(colors.HexColor("#303030"))
    canvas.drawCentredString(A4[0] / 2, 12 * mm, str(doc.page))
    canvas.restoreState()


def build() -> None:
    register_fonts()
    TMP.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig1 = make_diagram_png("fig-01-observed-process", 1800)
    fig2 = make_diagram_png("fig-02-evidence-status", 1600)

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "BodyTNR",
        parent=styles["BodyText"],
        fontName="TNR",
        fontSize=12,
        leading=13.8,
        alignment=TA_JUSTIFY,
        spaceAfter=5.4,
        firstLineIndent=8 * mm,
    )
    body_noindent = ParagraphStyle("BodyNoIndent", parent=body, firstLineIndent=0)
    title = ParagraphStyle(
        "TitleTNR", parent=styles["Title"], fontName="TNR-Bold", fontSize=14,
        leading=16, alignment=TA_CENTER, spaceAfter=8,
    )
    abstract = ParagraphStyle(
        "Abstract", parent=body, fontSize=11.2, leading=12.9, firstLineIndent=0,
        spaceAfter=5,
    )
    h1 = ParagraphStyle(
        "H1", parent=styles["Heading1"], fontName="TNR-Bold", fontSize=13,
        leading=15, spaceBefore=9, spaceAfter=5,
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontName="TNR-Bold", fontSize=12,
        leading=14, spaceBefore=7, spaceAfter=4,
    )
    caption = ParagraphStyle(
        "Caption", parent=styles["BodyText"], fontName="TNR", fontSize=10,
        leading=11.5, alignment=TA_CENTER, spaceBefore=2, spaceAfter=6,
    )
    quote = ParagraphStyle(
        "Quote", parent=body_noindent, fontName="TNR-Italic", fontSize=11.2,
        leading=13, leftIndent=10 * mm, rightIndent=5 * mm, spaceBefore=3, spaceAfter=4,
    )
    math = ParagraphStyle(
        "Math", parent=body_noindent, fontName="TNR", fontSize=10.5,
        leading=12, alignment=TA_CENTER, spaceBefore=3, spaceAfter=4,
    )
    ref = ParagraphStyle(
        "Reference", parent=body_noindent, fontName="TNR", fontSize=11,
        leading=11.3, leftIndent=6 * mm, firstLineIndent=-6 * mm, spaceAfter=2.3,
    )
    list_style = ParagraphStyle(
        "List", parent=body, leftIndent=5 * mm, firstLineIndent=-5 * mm,
    )
    cell = ParagraphStyle(
        "Cell", parent=styles["BodyText"], fontName="TNR", fontSize=7.4,
        leading=8.5, alignment=TA_JUSTIFY,
    )
    cell_head = ParagraphStyle(
        "CellHead", parent=cell, fontName="TNR-Bold", alignment=TA_CENTER,
    )

    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4,
        rightMargin=20 * mm, leftMargin=20 * mm, topMargin=20 * mm, bottomMargin=20 * mm,
        title="Глава 3: прогнозирование временных рядов",
        author="",
    )
    text = SOURCE.read_text(encoding="utf-8")
    lines = text.splitlines()
    story = []
    buffer: list[str] = []
    in_refs = False
    title_done = False

    def flush(style=body):
        nonlocal buffer
        if buffer:
            value = " ".join(x.strip() for x in buffer).strip()
            if value:
                story.append(Paragraph(clean_inline(value), style))
            buffer = []

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line:
            flush(abstract if not title_done else body)
            i += 1
            continue
        if line.startswith("# "):
            flush()
            story.append(Paragraph(clean_inline(line[2:]), title))
            title_done = True
        elif line.startswith("## "):
            flush(abstract if not title_done else body)
            heading = clean_inline(line[3:])
            if heading == "Список литературы":
                in_refs = True
            story.append(Paragraph(heading, h1))
        elif line.startswith("### "):
            flush()
            story.append(Paragraph(clean_inline(line[4:]), h2))
        elif line.startswith("!["):
            flush()
            path_match = re.search(r"\(([^)]+)\)", line)
            if path_match:
                p = path_match.group(1)
                image_path = fig1 if "fig-01" in p else fig2
                probe = Image(str(image_path))
                target_width = doc.width * 0.90
                target_height = probe.imageHeight * target_width / probe.imageWidth
                image = Image(str(image_path), width=target_width, height=target_height)
                image.hAlign = "CENTER"
                story.append(image)
                story.append(Spacer(1, 2))
        elif line.startswith("*Рисунок ") and line.endswith("*"):
            flush()
            story.append(Paragraph(clean_inline(line[1:-1]), caption))
        elif line.startswith("**Таблица ") and line.endswith("**"):
            flush()
            story.append(Paragraph(clean_inline(line[2:-2]), caption))
        elif line.startswith("> "):
            flush()
            story.append(Paragraph(clean_inline(line[2:]), quote))
        elif line.startswith("$$"):
            flush()
            formula = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("$$"):
                formula.append(lines[i].strip())
                i += 1
            display = " ".join(formula)
            display = display.replace("\\hat y", "ŷ").replace("\\left", "").replace("\\right", "")
            display = display.replace("\\cdot", "·").replace("\\frac", "")
            story.append(Paragraph(clean_inline(display), math))
        elif line.startswith("|"):
            flush()
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            data_lines = [x for x in table_lines if not re.match(r"^\|[-| :]+\|$", x)]
            rows = []
            for row_index, entry in enumerate(data_lines):
                fields = [x.strip() for x in entry.strip("|").split("|")]
                style = cell_head if row_index == 0 else cell
                rows.append([Paragraph(clean_inline(x), style) for x in fields])
            if rows:
                cols = len(rows[0])
                widths = [doc.width / cols] * cols
                table = LongTable(rows, colWidths=widths, repeatRows=1, hAlign="CENTER")
                table.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#5A5A5A")),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EEF0")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]))
                story.append(table)
                story.append(Spacer(1, 4))
            continue
        elif re.match(r"^\d+\. ", line):
            flush()
            style = ref if in_refs else list_style
            story.append(Paragraph(clean_inline(line), style))
        elif line.startswith("**") and line.endswith("**"):
            flush()
            story.append(Paragraph(clean_inline(line[2:-2]), body_noindent))
        else:
            buffer.append(line)
        i += 1

    flush(ref if in_refs else body)
    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)


if __name__ == "__main__":
    build()
