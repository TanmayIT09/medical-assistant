"""Helpers for storing small pieces of local application state."""

from __future__ import annotations

import json
from typing import Any

from app.config import settings


def set_active_report(source_name: str) -> None:
    """Persist the currently active uploaded report name."""
    settings.active_report_path.write_text(
        json.dumps({"source_name": source_name}, indent=2),
        encoding="utf-8",
    )


def get_active_report() -> str | None:
    """Return the currently active uploaded report name if available."""
    if not settings.active_report_path.exists():
        return None

    try:
        payload: dict[str, Any] = json.loads(settings.active_report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    source_name = payload.get("source_name")
    if isinstance(source_name, str) and source_name.strip():
        return source_name
    return None
