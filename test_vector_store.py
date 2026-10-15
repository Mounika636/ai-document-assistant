
from vector_store import (
    get_collection,
    store_chunks,
    search_chunks,
)


def test_store_and_search(tmp_path):
    chunks = [
        "Employees may work remotely three days per week.",
        "The company provides health insurance benefits.",
        "Python is commonly used for machine learning.",
    ]

    count = store_chunks(
        document_id="test_document",
        chunks=chunks,
        db_path=tmp_path,
    )

    assert count == 3

    collection = get_collection(tmp_path)
    assert collection.count() == 3

    results = search_chunks(
        query="How many days can employees work from home?",
        top_k=2,
        db_path=tmp_path,
    )

    assert len(results) == 2
    assert results[0]["chunk_index"] == 0
    assert "remotely" in results[0]["text"]


def test_empty_database(tmp_path):
    results = search_chunks(
        query="What is machine learning?",
        db_path=tmp_path,
    )

    assert results == []


def test_empty_chunks(tmp_path):
    count = store_chunks(
        document_id="empty_document",
        chunks=[],
        db_path=tmp_path,
    )

    assert count == 0
