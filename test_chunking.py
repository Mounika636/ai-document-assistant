
import pytest

from chunking import chunk_text


def test_empty_text():
    assert chunk_text("") == []


def test_short_text():
    assert chunk_text("Hello RAG") == ["Hello RAG"]


def test_long_text():
    chunks = chunk_text("A" * 2500)

    assert len(chunks) == 3
    assert all(len(chunk) <= 1000 for chunk in chunks)


def test_chunk_overlap():
    text = "A" * 1000 + "B" * 500
    chunks = chunk_text(text)

    assert chunks[0][-200:] == chunks[1][:200]


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("Hello", chunk_size=0)


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("Hello", chunk_size=100, overlap=100)
