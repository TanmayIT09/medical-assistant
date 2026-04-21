"""Chunking utilities for retrieved document text."""

from __future__ import annotations

import re

from app.config import settings


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split text into overlapping chunks for embedding and retrieval."""
    normalized_text = _normalize_report_text(text)
    if not normalized_text:
        return []

    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    if overlap >= chunk_size:
        raise ValueError("chunk overlap must be smaller than the chunk size")

    paragraphs = [part.strip() for part in normalized_text.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        candidate = f"{current_chunk}\n\n{paragraph}".strip() if current_chunk else paragraph
        if len(candidate) <= chunk_size:
            current_chunk = candidate
            continue

        if current_chunk:
            chunks.append(current_chunk.strip())

        if len(paragraph) <= chunk_size:
            current_chunk = paragraph
            continue

        line_buffer = ""
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        for line in lines:
            candidate_line_chunk = f"{line_buffer}\n{line}".strip() if line_buffer else line
            if len(candidate_line_chunk) <= chunk_size:
                line_buffer = candidate_line_chunk
                continue

            if line_buffer:
                chunks.append(line_buffer.strip())
            line_buffer = line

        current_chunk = line_buffer.strip()

    if current_chunk:
        chunks.append(current_chunk.strip())

    return _add_overlap(chunks, overlap=overlap)


def _normalize_report_text(text: str) -> str:
    """Normalize report text while preserving useful line and table structure."""
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        collapsed = re.sub(r"[ \t]+", " ", raw_line).strip()
        if collapsed:
            cleaned_lines.append(collapsed)
            continue

        if cleaned_lines and cleaned_lines[-1] != "":
            cleaned_lines.append("")

    return "\n".join(cleaned_lines).strip()


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Add lightweight textual overlap so neighboring chunks retain context."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped_chunks: list[str] = []
    for index, chunk in enumerate(chunks):
        if index == 0:
            overlapped_chunks.append(chunk)
            continue

        previous_tail = chunks[index - 1][-overlap:].strip()
        overlapped_chunks.append(f"{previous_tail}\n{chunk}".strip())

    return overlapped_chunks
