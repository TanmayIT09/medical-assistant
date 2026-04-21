# medical-assistant

Production-ready local medical document assistant built with FastAPI, Streamlit, Ollama, ChromaDB, PyMuPDF, and Tesseract OCR.

## Features

- Upload PDF and image files
- Extract text from PDFs with PyMuPDF
- OCR scanned PDF pages and image uploads with Tesseract
- Chunk extracted text for retrieval
- Generate embeddings with Ollama using `nomic-embed-text`
- Store vectors in local ChromaDB persistence
- Ask questions through a guarded RAG pipeline using `llama3`
- Generate report-based diet guidance and precautions from uploaded reports
- Use a FastAPI backend with a Streamlit frontend

## Project Structure

```text
medical-assistant/
  app/
    main.py
    config.py
    ingestion/
      pdf_parser.py
      image_ocr.py
    processing/
      chunking.py
      embeddings.py
      vector_store.py
    llm/
      prompts.py
      rag_pipeline.py
    services/
      report_service.py
      qa_service.py
  ui/
    streamlit_app.py
  data/uploads/
  data/chroma_db/
  requirements.txt
  README.md
```

## Prerequisites

Install these local dependencies before running the project:

1. Python 3.11 or newer
2. [Ollama](https://ollama.com/) running locally
3. Tesseract OCR installed and available on your system path

Pull the Ollama models:

```bash
ollama pull nomic-embed-text
ollama pull llama3
```

If Tesseract is not on your `PATH`, set `TESSERACT_CMD` to the installed executable location.

## Setup

Create and activate a virtual environment, then install Python dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run The Backend

```bash
uvicorn app.main:app --reload
```

The API will start at `http://localhost:8000`.

Available endpoints:

- `POST /upload`
- `GET /ask?question=...`
- `GET /report-guidance`
- `GET /health`

## Run The Frontend

Open a new terminal and run:

```bash
streamlit run ui/streamlit_app.py
```

The Streamlit app will connect to `http://localhost:8000` by default.

If you want to point the frontend to a different backend URL, create `.streamlit/secrets.toml`:

```toml
api_base_url = "http://localhost:8000"
```

## Example Workflow

1. Start Ollama and ensure both models are available.
2. Start the FastAPI backend.
3. Start the Streamlit frontend.
4. Upload a PDF or image in the UI.
5. Click `Generate Report Guidance` for diet suggestions, foods to avoid, and precautions.
6. Optionally ask a custom question about the uploaded content.

## Safety Notes

- The assistant is designed for informational summarization only.
- It explicitly avoids diagnosis, prescriptions, and treatment plans.
- Users should consult a licensed medical professional for clinical decisions.

## Local Data

- Uploaded files are stored in `data/uploads/`
- ChromaDB persistence is stored in `data/chroma_db/`

## Notes

- Scanned PDFs are OCR'd page by page when no text layer is found.
- All storage and inference are local, provided your Ollama server is local.
