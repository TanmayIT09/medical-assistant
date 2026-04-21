"""Retrieval augmented generation pipeline for question answering."""

from __future__ import annotations

from ollama import Client

from app.config import settings
from app.llm.prompts import SYSTEM_PROMPT, build_guidance_prompt, build_user_prompt
from app.processing.embeddings import OllamaEmbeddingService
from app.processing.vector_store import ChromaVectorStore
from app.state import get_active_report


class MedicalRAGPipeline:
    """Coordinates retrieval from ChromaDB and answer generation via Ollama."""

    GUIDANCE_QUERY = (
        "Summarize the report and suggest safe diet options, foods to avoid, "
        "and general precautions based on the report findings."
    )

    def __init__(self) -> None:
        self._chat_client = Client(host=settings.ollama_host)
        self._embedding_service = OllamaEmbeddingService()
        self._vector_store = ChromaVectorStore()

    def _build_context(self, query_text: str) -> tuple[str, list[dict]]:
        """Retrieve the most relevant chunks for a given query."""
        query_embedding = self._embedding_service.embed_text(query_text)
        active_report = get_active_report()
        matches = self._vector_store.query(
            query_embedding=query_embedding,
            source_name=active_report,
        )
        context = "\n\n".join(
            (
                f"Source: {match['metadata'].get('source', 'unknown')}\n"
                f"Chunk: {match['metadata'].get('chunk_index', 'unknown')}\n"
                f"{match['document']}"
            )
            for match in matches
        ).strip()
        return context, matches

    def ask(self, question: str) -> dict:
        """Retrieve relevant context and generate a guarded answer."""
        context, matches = self._build_context(question)

        if not context:
            return {
                "answer": "I could not find any uploaded document context yet. Please upload a PDF or image first.",
                "sources": [],
            }

        response = self._chat_client.chat(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question=question, context=context)},
            ],
        )

        return {
            "answer": response["message"]["content"].strip(),
            "sources": [match["metadata"].get("source", "unknown") for match in matches],
        }

    def generate_guidance(self) -> dict:
        """Create structured diet and precaution guidance from uploaded report context."""
        context, matches = self._build_context(self.GUIDANCE_QUERY)
        if not context:
            return {
                "guidance": "I could not find any uploaded document context yet. Please upload a PDF or image first.",
                "sources": [],
            }

        response = self._chat_client.chat(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_guidance_prompt(context=context)},
            ],
        )

        return {
            "guidance": response["message"]["content"].strip(),
            "sources": [match["metadata"].get("source", "unknown") for match in matches],
        }
