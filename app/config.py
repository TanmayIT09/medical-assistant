"""Application configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


@dataclass(frozen=True)
class Settings:
    """Centralized runtime settings for local development."""

    app_name: str = "medical-assistant"
    upload_dir: Path = DATA_DIR / "uploads"
    chroma_dir: Path = DATA_DIR / "chroma_db"
    state_dir: Path = DATA_DIR / "state"
    active_report_path: Path = DATA_DIR / "state" / "active_report.json"
    collection_name: str = "medical_documents"
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    embedding_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    max_context_chunks: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "4"))
    streamlit_api_url: str = os.getenv("STREAMLIT_API_URL", "http://localhost:8000")
    tesseract_cmd: str | None = os.getenv("TESSERACT_CMD")

    def ensure_directories(self) -> None:
        """Create required local directories if missing."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
