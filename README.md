
# AI Document Assistant

A production-style Retrieval-Augmented Generation (RAG)
application built using Python and FastAPI.

## Current Features

- PDF upload and text extraction using PyMuPDF
- Document chunking with overlap
- Local text embeddings using Sentence Transformers
- Persistent vector database using ChromaDB
- Semantic similarity search
- FastAPI REST endpoints
- Automated tests with Pytest

## API Endpoints

- GET / - Application information
- GET /health - Health check
- POST /upload - Upload, process and index a PDF
- POST /search - Search indexed document chunks

## Upcoming Features

- LLM-powered question answering
- Page-level source citations
- Document filtering
- RAG evaluation
- Docker deployment
- GitHub Actions CI/CD

## Planned Features
- PDF document upload
- Document processing and chunking
- Vector embeddings and semantic search
- ChromaDB vector database
- LLM-powered question answering
- Source citations
- Docker deployment
- GitHub Actions CI/CD

## Tech Stack
Python 3.12 | FastAPI | Pytest

## Getting Started

1. Create a Python 3.12 virtual environment.
2. Activate the virtual environment.
3. Install dependencies:

   pip install -r requirements.txt

4. Start the application:

   python -m uvicorn main:app --reload

5. Open http://127.0.0.1:8000/docs

## Run Tests

python -m pytest -v

## Status
Day 1 complete: Backend initialized and tests passing.

## Day 4 Features

- Local LLM integration with Ollama
- Qwen 2.5 3B for question answering
- Retrieval-Augmented Generation pipeline
- POST /ask endpoint
- AI-generated responses from retrieved document context
- Source chunk references
- Automated LLM tests using mocks

## RAG Architecture

PDF Upload
    ↓
Text Extraction
    ↓
Document Chunking
    ↓
Sentence Transformers Embeddings
    ↓
ChromaDB Vector Database
    ↓
Semantic Retrieval
    ↓
Ollama + Qwen 2.5 3B
    ↓
AI Answer + Retrieved Sources
