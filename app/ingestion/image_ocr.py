"""OCR helpers for uploaded image files."""

from __future__ import annotations

from pathlib import Path

import pytesseract
from PIL import Image

from app.config import settings


def extract_text_from_image(image_path: Path) -> str:
    """Extract text from an image using Tesseract OCR."""
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    with Image.open(image_path) as image:
        text = pytesseract.image_to_string(image)

    return text.strip()
