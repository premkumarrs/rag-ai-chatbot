# RAG AI Chatbot

A company-oriented Retrieval-Augmented Generation (RAG) chatbot that answers questions from internal documents. The system retrieves relevant knowledge-base content, grounds responses in that material, and falls back to a support message when the documents do not contain enough information.

## Current Features

- FastAPI backend
- Retrieval-Augmented Generation (RAG)
- PostgreSQL + pgvector vector storage
- Ollama / Qwen local LLM
- Nomic embeddings (`nomic-embed-text`)
- Multi-format ingestion: PDF, DOCX, TXT, Markdown, CSV, XLSX, PPTX, HTML, JSON
- OCR for scanned PDF pages and images (Tesseract)
- Structure-aware document chunking
- Document metadata (page, sheet, slide, section, etc.)
- SHA-256 content hashing and idempotent ingestion
- Similarity-threshold retrieval with grounded fallback when confidence is low
- Hybrid retrieval (vector + PostgreSQL keyword/full-text)
- Lightweight query normalization and deterministic reranking
- Retrieval confidence (HIGH / MEDIUM / LOW) with context assembly
- Streaming chat endpoint (`POST /chat/stream`)
- LLM provider abstraction (Ollama today; swappable for a future company GPU endpoint)

## Architecture

```
Documents
  → Parsing / OCR
  → Chunking
  → Embeddings
  → PostgreSQL + pgvector

User Question
  → Query normalization
  → Hybrid retrieval (vector + keyword)
  → Candidate fusion
  → Lightweight reranking
  → Retrieval confidence
  → Context assembly
  → Qwen LLM
  → Grounded Answer
```

The LLM layer uses a provider interface so the inference backend can later be switched to another approved local or company-hosted model endpoint without rewriting the RAG pipeline.

### Retrieval

- **Vector retrieval** — semantic search over pgvector embeddings
- **Keyword retrieval** — PostgreSQL full-text search plus exact technical-term matching
- **Hybrid fusion** — combines both result sets (Reciprocal Rank Fusion)
- **Lightweight reranking** — deterministic reordering of the small candidate set (no extra LLM call)
- **Retrieval confidence** — HIGH / MEDIUM / LOW based on candidate quality; LOW triggers grounded fallback
- **Grounded fallback** — answers only from retrieved company context; otherwise returns a support message

Query normalization is deterministic (informal wording, light typos). Technical identifiers such as `ADXL345`, `ESP32`, and `C-MAPSS` are preserved.

## Supported Documents

| Format | Extensions |
|---|---|
| PDF | `.pdf` |
| Word | `.docx` |
| Plain text | `.txt` |
| Markdown | `.md` |
| CSV | `.csv` |
| Excel | `.xlsx` |
| PowerPoint | `.pptx` |
| HTML | `.html`, `.htm` |
| JSON | `.json` |
| Images (OCR) | `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff` |

Place files directly under `data/` (subfolders are ignored). Ingestion runs offline before queries. The `data/` directory is gitignored—do not commit company documents.

## Setup

**Requirements:** Python 3.12, Docker, Ollama

1. **Create and activate a virtual environment**

```powershell
cd rag-ai-chatbot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. **Install dependencies**

```powershell
pip install -r requirements.txt
```

3. **Start PostgreSQL with pgvector**

```powershell
docker start rag-postgres
```

4. **Configure environment variables** (optional; defaults work for local development)

```powershell
$env:DATABASE_URL = "postgresql://USER:PASSWORD@localhost:5433/rag_ai_chatbot"
$env:OLLAMA_HOST = "http://localhost:11434"
```

5. **Pull Ollama models**

```powershell
ollama pull qwen3.5:4b
ollama pull nomic-embed-text
```

6. **Initialize the database and ingest documents**

```powershell
python -c "from app.database import init_db; init_db()"
python -m app.ingest
```

7. **Start the API**

```powershell
uvicorn app.main:app --reload
```

API docs: http://127.0.0.1:8000/docs

## OCR

- OCR uses **Tesseract** through `pytesseract` and **Pillow**.
- Normal PDF text extraction is attempted first.
- OCR runs only when extracted text is insufficient (scanned pages or image files).
- Tesseract must be installed separately on Windows:

```powershell
winget install UB-Mannheim.TesseractOCR
```

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service health check |
| POST | `/chat` | Question → grounded JSON answer |
| POST | `/chat/stream` | Question → streamed SSE response |

**Example — POST /chat**

```json
{
  "question": "What is covered under the product warranty?"
}
```

**Response**

```json
{
  "answer": "...",
  "sources": ["data/warranty.docx"],
  "fallback": false
}
```

When `fallback` is `true`, the knowledge base did not contain sufficient relevant content.

## Testing

```powershell
python -m unittest discover -s tests -p "test_*.py" -q
```

Tests cover document parsers, chunking, idempotent ingestion, OCR availability, hybrid retrieval (normalization, fusion, reranking, confidence, context assembly), LLM provider wiring, and RAG orchestration.

## Project Status

The current implementation includes the core RAG pipeline, multi-format ingestion with OCR, and Phase 3 intelligent hybrid retrieval (vector + keyword, fusion, lightweight reranking, retrieval confidence, and context assembly). Production load testing and advanced model-based rerankers are out of scope for now.
