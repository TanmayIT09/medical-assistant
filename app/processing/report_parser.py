"""Utilities for reconstructing structured lab-report rows from extracted text."""

from __future__ import annotations

import re


VALUE_PATTERN = re.compile(r"^(?:[<>]?\s*)?\d+(?:\.\d+)?$")
RANGE_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?")
UNIT_PATTERN = re.compile(r"^[A-Za-z%][A-Za-z0-9/%(). -]{0,24}$")
OBSERVATION_KEYWORDS = {
    "positive",
    "negative",
    "non reactive",
    "reactive",
    "present",
    "absent",
    "trace",
    "normal",
    "abnormal",
}


def build_structured_report_text(raw_text: str) -> str:
    """Create a structured lab summary to keep values attached to the correct tests."""
    if raw_text.lstrip().startswith("Panel:") or raw_text.lstrip().startswith("Structured lab observations:"):
        return raw_text

    entries = extract_lab_entries(raw_text)
    if not entries:
        return raw_text

    structured_lines = ["Structured lab observations:"]
    for entry in entries:
        line_parts = [f"Test: {entry['test_name']}"]
        if entry.get("result"):
            line_parts.append(f"Result: {entry['result']}")
        if entry.get("unit"):
            line_parts.append(f"Unit: {entry['unit']}")
        if entry.get("reference_range"):
            line_parts.append(f"Reference Range: {entry['reference_range']}")
        structured_lines.append(" | ".join(line_parts))

    return "\n".join(structured_lines) + "\n\nRaw extracted text:\n" + raw_text.strip()


def extract_lab_entries(raw_text: str) -> list[dict[str, str]]:
    """Heuristically rebuild lab rows where results and test names are split across lines."""
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    entries: list[dict[str, str]] = []
    pending_value: str | None = None
    current: dict[str, str] | None = None

    for line in lines:
        if _is_value_line(line):
            if current and current.get("test_name") and not current.get("result"):
                current["result"] = line
            else:
                if current and _has_meaningful_content(current):
                    entries.append(current)
                    current = None
                pending_value = line
            continue

        if _is_reference_range_line(line):
            if current is None:
                current = _new_entry()
            current["reference_range"] = _append_field(current.get("reference_range"), line)
            continue

        if _is_unit_line(line):
            if current is None:
                current = _new_entry()
            current["unit"] = _append_field(current.get("unit"), line)
            continue

        if current and current.get("test_name") and (current.get("reference_range") or current.get("unit")):
            entries.append(current)
            current = None

        if current is None:
            current = _new_entry()

        current["test_name"] = _append_field(current.get("test_name"), line)
        if pending_value and not current.get("result"):
            current["result"] = pending_value
            pending_value = None

    if current and _has_meaningful_content(current):
        entries.append(current)

    return [entry for entry in entries if entry.get("test_name")]


def _new_entry() -> dict[str, str]:
    """Create a new empty lab entry."""
    return {"test_name": "", "result": "", "unit": "", "reference_range": ""}


def _append_field(existing: str | None, value: str) -> str:
    """Append a line to an existing entry field."""
    if not existing:
        return value
    return f"{existing} {value}"


def _has_meaningful_content(entry: dict[str, str]) -> bool:
    """Check whether an entry has enough information to keep."""
    return any(entry.get(key) for key in ("test_name", "result", "unit", "reference_range"))


def _is_value_line(line: str) -> bool:
    """Identify single-value result lines such as 7.62 or Positive."""
    lowered = line.lower()
    return bool(VALUE_PATTERN.match(line)) or lowered in OBSERVATION_KEYWORDS


def _is_reference_range_line(line: str) -> bool:
    """Identify numeric reference ranges."""
    return bool(RANGE_PATTERN.search(line))


def _is_unit_line(line: str) -> bool:
    """Identify unit lines such as mg/dL or cells/hpf."""
    lowered = line.lower()
    if any(keyword in lowered for keyword in ("calculated", "method", "ratio", "panel")):
        return False
    return bool(UNIT_PATTERN.match(line)) and not _is_value_line(line) and not _is_reference_range_line(line)
