from io import BytesIO
import pymupdf
from fastapi.testclient import TestClient
from main import app
import pytest
import vector_store

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_vector_database(tmp_path, monkeypatch):
    """Use a temporary ChromaDB directory for each API test."""
    monkeypatch.setattr(
        vector_store,
        "DB_PATH",
        tmp_path / "chroma_db"
    )



def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == (
        "Welcome to AI Document Assistant"
    )


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"



def test_upload_valid_pdf():
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Hello RAG PDF")

    pdf_bytes = document.tobytes()
    document.close()

    response = client.post(
        "/upload",
        files={
            "file": (
                "sample.pdf",
                BytesIO(pdf_bytes),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["pages"] == 1
    assert response.json()["total_chunks"] >= 1
    assert "Hello RAG PDF" in response.json()["text_preview"]


def test_reject_non_pdf():
    response = client.post(
        "/upload",
        files={
            "file": (
                "sample.txt",
                b"Hello",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


def test_reject_invalid_pdf():
    response = client.post(
        "/upload",
        files={
            "file": (
                "fake.pdf",
                b"not a real pdf",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400


def test_reject_oversized_pdf():
    response = client.post(
        "/upload",
        files={
            "file": (
                "large.pdf",
                b"%PDF-" + b"x" * (10 * 1024 * 1024),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 413


    
def test_semantic_search_api():
    response = client.post(
        "/search",
        json={
            "query": "What are the technical skills?",
            "top_k": 3
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "What are the technical skills?"
    assert data["total_results"] == 0
    assert data["results"] == []

