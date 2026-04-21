"""FastAPI entry point for the medical assistant backend."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.services.qa_service import QAService
from app.services.report_service import ReportService


app = FastAPI(title="medical-assistant", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

report_service = ReportService()
qa_service = QAService()


@app.get("/health")
def health_check() -> dict:
    """Simple health endpoint for local verification."""
    return {"status": "ok"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> dict:
    """Upload a PDF or image and index it for later questions."""
    suffix = Path(file.filename or "").suffix.lower()
    allowed_suffixes = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    if suffix not in allowed_suffixes:
        raise HTTPException(status_code=400, detail="Only PDF and image files are supported.")

    destination = settings.upload_dir / Path(file.filename or "upload").name
    content = await file.read()
    destination.write_bytes(content)

    try:
        return report_service.ingest_file(destination)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {exc}") from exc


@app.get("/ask")
def ask_question(question: str = Query(..., min_length=3)) -> dict:
    """Ask a question about the uploaded medical document context."""
    try:
        return qa_service.answer_question(question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Question answering failed: {exc}") from exc


@app.get("/report-guidance")
def get_report_guidance() -> dict:
    """Generate structured diet and precaution guidance from the indexed report."""
    try:
        return qa_service.generate_report_guidance()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Guidance generation failed: {exc}") from exc
