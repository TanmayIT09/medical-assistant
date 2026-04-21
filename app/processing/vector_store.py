"""Persistent vector store helpers backed by ChromaDB."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import chromadb

from app.config import settings


class ChromaVectorStore:
    """Handles storage and retrieval of document chunks."""

    def __init__(self, persist_directory: Path | None = None) -> None:
        self._client = chromadb.PersistentClient(path=str(persist_directory or settings.chroma_dir))
        self._collection = self._client.get_or_create_collection(name=settings.collection_name)

    def _get_collection(self):
        """Return a fresh collection handle to avoid stale references after resets."""
        self._collection = self._client.get_or_create_collection(name=settings.collection_name)
        return self._collection

    def add_documents(
        self,
        chunks: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        source_name: str,
    ) -> int:
        """Store embedded chunks for later retrieval."""
        collection = self._get_collection()
        ids = [str(uuid4()) for _ in chunks]
        metadatas = [{"source": source_name, "chunk_index": index} for index, _ in enumerate(chunks)]
        collection.add(
            ids=ids,
            documents=list(chunks),
            embeddings=[list(vector) for vector in embeddings],
            metadatas=metadatas,
        )
        return len(ids)

    def query(
        self,
        query_embedding: Sequence[float],
        top_k: int | None = None,
        source_name: str | None = None,
    ) -> list[dict]:
        """Search similar chunks by vector similarity."""
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [list(query_embedding)],
            "n_results": top_k or settings.max_context_chunks,
        }
        if source_name:
            query_kwargs["where"] = {"source": source_name}

        collection = self._get_collection()
        results = collection.query(
            **query_kwargs,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        matches: list[dict] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            matches.append(
                {
                    "document": document,
                    "metadata": metadata or {},
                    "distance": distance,
                }
            )

        return matches

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self._get_collection().count()

    def reset(self) -> None:
        """Delete and recreate the collection to avoid stale retrieval state."""
        try:
            self._client.delete_collection(name=settings.collection_name)
        except Exception:
            # If the collection is already absent, just recreate it below.
            pass
        self._collection = self._client.get_or_create_collection(name=settings.collection_name)
