
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
    page_numbers: list[int] | None = None,
    filename: str | None = None,
):
    """Store document chunks with page metadata."""

    if not chunks:
        return 0

    if page_numbers is not None and len(page_numbers) != len(chunks):
        raise ValueError("page_numbers must match chunks length")

    collection = get_collection(db_path)
    embeddings = generate_embeddings(chunks)

    ids = [
        f"{document_id}_chunk_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "document_id": document_id,
            "chunk_index": i,
            "page_number": (
                page_numbers[i]
                if page_numbers is not None
                else 0
            ),
            "filename": filename or "unknown.pdf",
        }
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
    document_id: str | None = None,
):
    """Find relevant chunks, optionally within one document."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    collection = get_collection(db_path)

    if collection.count() == 0:
        return []

    # Restrict search to the selected document.
    where_filter = (
        {"document_id": document_id}
        if document_id
        else None
    )

    if document_id:
        available = collection.get(
            where=where_filter,
            include=[],
        )

        if not available["ids"]:
            return []

        result_count = min(
            top_k,
            len(available["ids"]),
        )
    else:
        result_count = min(
            top_k,
            collection.count(),
        )

    query_embedding = generate_embeddings([query])[0]

    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": result_count,
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if where_filter:
        query_args["where"] = where_filter

    results = collection.query(**query_args)

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
            "page_number": metadata.get("page_number", 0),
            "filename": metadata.get("filename", "unknown.pdf"),
            "similarity": round(1 - distance, 4),
        })

    return matches
