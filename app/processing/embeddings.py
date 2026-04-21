"""Embedding generation helpers backed by Ollama."""

from __future__ import annotations

from typing import Iterable

from ollama import Client

from app.config import settings


class OllamaEmbeddingService:
    """Small wrapper around the Ollama embeddings API."""

    def __init__(self) -> None:
        self._client = Client(host=settings.ollama_host)
        self._model = settings.embedding_model

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding vector for one text chunk."""
        response = self._client.embeddings(model=self._model, prompt=text)
        return response["embedding"]

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple chunks."""
        return [self.embed_text(text) for text in texts]
