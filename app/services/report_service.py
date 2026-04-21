"""Report ingestion service."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.image_ocr import extract_text_from_image
from app.ingestion.pdf_parser import extract_text_from_pdf
from app.processing.chunking import chunk_text
from app.processing.embeddings import OllamaEmbeddingService
from app.processing.report_parser import build_structured_report_text
from app.processing.vector_store import ChromaVectorStore
from app.state import set_active_report


class ReportService:
    """Handles saving uploads, extracting text, and indexing chunks."""

    SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    def __init__(self) -> None:
        self._embedding_service = OllamaEmbeddingService()
        self._vector_store = ChromaVectorStore()

    def ingest_file(self, file_path: Path) -> dict:
        """Extract text from a local file and index it in the vector store."""
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            extracted_text = extract_text_from_pdf(file_path)
        elif suffix in self.SUPPORTED_IMAGE_SUFFIXES:
            extracted_text = extract_text_from_image(file_path)
        else:
            raise ValueError("Unsupported file type. Please upload a PDF or image.")

        if not extracted_text:
            raise ValueError("No text could be extracted from the uploaded file.")

        structured_text = build_structured_report_text(extracted_text)
        chunks = chunk_text(structured_text)
        if not chunks:
            raise ValueError("No chunks were created from the extracted text.")

        # Keep retrieval focused on the latest uploaded report.
        self._vector_store.reset()
        embeddings = self._embedding_service.embed_texts(chunks)
        stored_count = self._vector_store.add_documents(
            chunks=chunks,
            embeddings=embeddings,
            source_name=file_path.name,
        )
        set_active_report(file_path.name)

        return {
            "filename": file_path.name,
            "characters": len(structured_text),
            "chunks_indexed": stored_count,
            "preview": structured_text[:1600],
        }
