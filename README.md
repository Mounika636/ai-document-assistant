
# AI Document Assistant

A production-style Retrieval-Augmented Generation (RAG)
application built using Python and FastAPI.

## Current Features
- FastAPI backend
- Health-check endpoint
- Interactive Swagger API documentation
- Automated API tests using Pytest

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
