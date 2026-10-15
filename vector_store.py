
from pathlib import Path

import chromadb

from embeddings import generate_embeddings


DB_PATH = Path(__file__).parent / "chroma_db"


def get_collection(db_path=None):
    """Create or open the local ChromaDB collection."""
    path = db_path or DB_PATH

    client = chromadb.PersistentClient(path=str(path))

    return client.get_or_create_collection(
        name="document_chunks",
        metadata={"hnsw:space": "cosine"},
    )


def store_chunks(
    document_id: str,
    chunks: list[str],
    db_path=None,
):
    """Generate embeddings and store document chunks."""
    if not chunks:
        return 0

    collection = get_collection(db_path)
    embeddings = generate_embeddings(chunks)

    ids = [
        f"{document_id}_chunk_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {"document_id": document_id, "chunk_index": i}
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(chunks)


def search_chunks(
    query: str,
    top_k: int = 3,
    db_path=None,
):
    """Find the most relevant stored chunks."""
    collection = get_collection(db_path)

    if collection.count() == 0:
        return []

    query_embedding = generate_embeddings([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    matches = []

    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        matches.append({
            "text": text,
            "document_id": metadata["document_id"],
            "chunk_index": metadata["chunk_index"],
            "similarity": round(1 - distance, 4),
        })

    return matches
