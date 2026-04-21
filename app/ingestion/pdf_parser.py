"""PDF parsing utilities backed by PyMuPDF."""

from __future__ import annotations

import io
from pathlib import Path
import re

import fitz
import pytesseract
from PIL import Image

from app.config import settings


VALUE_PATTERN = re.compile(r"^(?:[<>]?\s*)?\d+(?:\.\d+)?$")
RANGE_PATTERN = re.compile(r"^(?:[<>]?\s*)?\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?$")
PAGE_PATTERN = re.compile(r"^Page \d+ of \d+$", re.IGNORECASE)
ASTERISK_PATTERN = re.compile(r"^\*.*\*$")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract searchable text from a PDF and OCR scanned pages when needed."""
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    extracted_pages: list[str] = []
    document = fitz.open(pdf_path)

    try:
        for page in document:
            blocks = page.get_text("blocks", sort=True)
            page_text = _extract_structured_page_text(blocks)
            if page_text:
                extracted_pages.append(page_text)
                continue

            # Fall back to OCR for scanned pages with no text layer.
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            ocr_text = pytesseract.image_to_string(image).strip()
            if ocr_text:
                extracted_pages.append(ocr_text)
    finally:
        document.close()

    return "\n\n".join(extracted_pages).strip()


def _extract_structured_page_text(blocks: list[tuple]) -> str:
    """Parse report-style PDF blocks into structured lab rows when possible."""
    block_texts = [
        block[4].strip()
        for block in blocks
        if len(block) >= 5 and block[4].strip()
    ]
    raw_text = "\n\n".join(block_texts).strip()

    start_index: int | None = None
    for index, block_text in enumerate(block_texts):
        lowered = block_text.lower()
        if "test name" in lowered and "results" in lowered and "bio. ref. interval" in lowered:
            start_index = index + 1
            break

    if start_index is None:
        return raw_text

    structured_rows: list[str] = []
    current_panel: str | None = None
    emitted_panel: str | None = None

    for block_text in block_texts[start_index:]:
        lines = [line.strip() for line in block_text.splitlines() if line.strip()]
        if not lines or _is_footer_or_noise(lines):
            continue

        row = _parse_test_row(lines)
        if row:
            if current_panel and current_panel != emitted_panel:
                structured_rows.append(f"Panel: {current_panel}")
                emitted_panel = current_panel
            structured_rows.append(row)
            continue

        if _is_section_heading(lines):
            heading = " ".join(lines)
            if "test report" in heading.lower():
                continue
            current_panel = heading
            continue

    if not structured_rows:
        return raw_text
    return "\n".join(structured_rows) + "\n\nRaw extracted text:\n" + raw_text


def _find_table_header_index(lines: list[str]) -> int | None:
    """Return the index of the test table header if present."""
    for index, line in enumerate(lines):
        lowered = line.lower()
        if "test name" in lowered and "results" in lowered and "bio. ref. interval" in lowered:
            return index
    return None


def _is_footer_or_noise(lines: list[str]) -> bool:
    """Identify page footers and other non-row noise."""
    joined = " ".join(lines)
    lowered = joined.lower()
    if PAGE_PATTERN.match(joined) or ASTERISK_PATTERN.match(joined):
        return True
    if lowered.startswith("note:"):
        return True
    return False


def _is_section_heading(lines: list[str]) -> bool:
    """Identify panel headers such as LIVER & KIDNEY FUNCTION TEST."""
    if len(lines) > 2:
        return False
    first = lines[0]
    if VALUE_PATTERN.match(first):
        return False
    if first.startswith("(") and first.endswith(")"):
        return False
    letters = [char for char in first if char.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    return uppercase_ratio > 0.85


def _parse_test_row(lines: list[str]) -> str | None:
    """Parse one lab result row from block lines."""
    if not lines:
        return None
    if not _is_result_token(lines[0]):
        return None
    if len(lines) < 2:
        return None

    result = lines[0]
    test_name = lines[1]
    method = ""
    reference_range = ""
    unit = ""

    for extra in lines[2:]:
        if PAGE_PATTERN.match(extra) or ASTERISK_PATTERN.match(extra):
            continue
        if extra.startswith("(") and extra.endswith(")"):
            method = f"{method} {extra}".strip()
        elif _is_reference_token(extra):
            reference_range = f"{reference_range} {extra}".strip()
        else:
            unit = f"{unit} {extra}".strip()

    parts = [f"Test: {test_name}", f"Result: {result}"]
    if method:
        parts.append(f"Method: {method}")
    if reference_range:
        parts.append(f"Reference Range: {reference_range}")
    if unit:
        parts.append(f"Unit: {unit}")
    return " | ".join(parts)


def _is_result_token(text: str) -> bool:
    """Identify a result token at the start of a row."""
    lowered = text.lower()
    return bool(VALUE_PATTERN.match(text)) or lowered in {
        "positive",
        "negative",
        "non reactive",
        "reactive",
        "present",
        "absent",
        "trace",
        "g1",
        "g2",
        "g3",
        "g4",
        "g5",
    }


def _is_reference_token(text: str) -> bool:
    """Identify reference range tokens such as 3.5 - 7.2 or <0.3."""
    normalized = text.replace(" ", "")
    if normalized.startswith((">", "<")):
        remainder = normalized[1:]
        return bool(remainder) and bool(re.fullmatch(r"\d+(?:\.\d+)?", remainder))
    return bool(RANGE_PATTERN.match(text))
