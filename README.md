# AI Document Assistant — Local RAG Application

A document question-answering application built with **Python, FastAPI, Streamlit, ChromaDB, Sentence Transformers, and Ollama**.

Users can upload PDF documents, ask natural-language questions, and receive AI-generated answers supported by retrieved document passages and PDF page references.

## Features

- PDF upload and text extraction using PyMuPDF
- Page-aware text chunking
- Semantic embeddings using `all-MiniLM-L6-v2`
- Persistent vector storage using ChromaDB
- Semantic document retrieval
- Local answer generation using Ollama and Qwen 2.5 3B
- Streamlit chatbot interface
- Source filenames and page-number references
- Document-specific retrieval filtering
- Docker Compose deployment
- Automated Python tests with GitHub Actions
- Reproducible RAG evaluation script

## System Architecture

```text
User
  |
  v
Streamlit Frontend
  |
  v
FastAPI Backend
  |
  +--> PDF Extraction + Page-Aware Chunking
  |          |
  |          v
  |    Sentence Transformers
  |          |
  |          v
  |       ChromaDB
  |          |
  |          v
  +---- Semantic Retrieval
  |
  v
Ollama + Qwen 2.5 3B
  |
  v
Answer + Retrieved PDF Sources
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.12 |
| Backend | FastAPI |
| Frontend | Streamlit |
| PDF processing | PyMuPDF |
| Embeddings | Sentence Transformers |
| Vector database | ChromaDB |
| LLM inference | Ollama + Qwen 2.5 3B |
| Containerization | Docker Compose |
| Testing | pytest |
| Continuous integration | GitHub Actions |

## Running Locally with Docker

### Prerequisites

- Docker Desktop
- Ollama installed on the host computer
- Qwen 2.5 3B model downloaded

Download the model:

```bash
ollama pull qwen2.5:3b
```

Start Ollama if it is not already running.

Start the application:

```bash
docker compose up -d --build
```

Access the application:

- Streamlit: http://localhost:8501
- FastAPI documentation: http://localhost:8000/docs

The backend connects to Ollama through `host.docker.internal:11434`. Host accessibility may require additional configuration depending on the Docker and Ollama environment.

Stop the application:

```bash
docker compose down
```

ChromaDB data and the embedding-model cache are preserved in Docker volumes.

## Running Tests

Install the dependencies in a Python 3.12 environment:

```bash
pip install -r requirements.txt
```

Run the automated test suite:

```bash
python -m pytest -v
```

The project currently has **25 passing tests**, including a test verifying accurate storage of PDF page numbers.

GitHub Actions also runs automated tests on pushes and pull requests to the main branch.

## RAG Evaluation

The evaluation uses a two-page synthetic company-policy PDF and three questions:

1. Remote work allowance — expected answer: three days per week.
2. Annual leave allowance — expected answer: twenty days.
3. Annual revenue — not provided in the PDF; the assistant should abstain.

Generate the evaluation document:

```bash
python evaluation/create_sample_pdf.py
```

With the API and Ollama running, execute:

```bash
python evaluation/evaluate.py
```

### Measured Baseline

| Metric | Result |
|---|---|
| Answer checks | 3/3 (100%) |
| Expected-page retrieval | 2/2 (100%) |
| Unanswerable question handling | 1/1 passed |

**Evaluation limitations:** These results are from a small synthetic dataset, not a production benchmark. Answer checks use simple keyword matching, and retrieval is considered successful when the expected page appears anywhere in the returned source passages. No independent LLM-as-judge or human factuality assessment has been performed.

## Current Limitations

- Supports text-extractable PDFs; no OCR for scanned documents
- No user authentication or tenant-level document isolation
- Depends on a locally running Ollama instance
- Source references identify retrieved passages, but do not guarantee every generated statement is supported
- Evaluation dataset is intentionally small and illustrative
- Not hardened for public production deployment

## Potential Improvements

- Larger evaluation dataset with retrieval ranking metrics
- Stronger groundedness and hallucination evaluation
- Hybrid retrieval and reranking
- OCR support
- Authentication and secure multi-user document isolation
- Cloud deployment and monitoring

## Project Purpose

This project demonstrates practical experience building and testing a Retrieval-Augmented Generation (RAG) system, connecting machine learning components through backend APIs, packaging services with Docker, and integrating automated tests into a CI workflow.
