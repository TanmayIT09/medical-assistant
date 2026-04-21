"""Question answering service."""

from __future__ import annotations

from app.llm.rag_pipeline import MedicalRAGPipeline


class QAService:
    """Thin service wrapper for the question-answering pipeline."""

    def __init__(self) -> None:
        self._pipeline = MedicalRAGPipeline()

    def answer_question(self, question: str) -> dict:
        """Generate an answer from indexed medical documents."""
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        return self._pipeline.ask(question=question.strip())

    def generate_report_guidance(self) -> dict:
        """Generate structured diet and precaution guidance from the report."""
        return self._pipeline.generate_guidance()
