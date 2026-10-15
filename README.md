
# AI Document Assistant

A production-style Retrieval-Augmented Generation (RAG)
application built using Python and FastAPI.

## Current Features

- REST API using FastAPI
- PDF upload with file validation (10 MB limit)
- PDF text extraction using PyMuPDF
- Document chunking with configurable overlap
- Text and chunk previews
- Automated testing with Pytest (12 passing tests)
- Interactive Swagger API documentation

## Upcoming Features

- Semantic embeddings
- ChromaDB vector search
- LLM-powered question answering
- Source citations
- Docker containerization
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
