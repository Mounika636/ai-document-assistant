from uuid import uuid4
from llm_service import generate_answer
from pydantic import BaseModel, Field
from vector_store import store_chunks, search_chunks
from io import BytesIO
from chunking import chunk_text
import pymupdf
from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI(
    title="AI Document Assistant",
    version="0.1.0"
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@app.get("/")
def home():
    return {
        "message": "Welcome to AI Document Assistant"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    contents = await file.read(MAX_FILE_SIZE + 1)

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PDF exceeds the 10 MB upload limit"
        )

    if not contents.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file"
        )

    try:
        with pymupdf.open(stream=contents, filetype="pdf") as document:
            pages = [
                page.get_text(sort=True)
                for page in document
            ]
            page_count = len(document)

    except (pymupdf.FileDataError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Unable to read PDF file"
        )

    
    text = "\n".join(pages).strip()

    if not text:
        raise HTTPException(
            status_code=422,
            detail="No extractable text found in PDF"
        )

    # Create chunks separately for each PDF page
    chunks = []
    page_numbers = []

    for page_number, page_text in enumerate(pages, start=1):
        page_chunks = chunk_text(page_text)

        chunks.extend(page_chunks)
        page_numbers.extend(
            [page_number] * len(page_chunks)
        )

    document_id = uuid4().hex

    store_chunks(
        document_id=document_id,
        chunks=chunks,
        page_numbers=page_numbers,
        filename=file.filename,
    )

    return {
        "document_id": document_id,
        "filename": file.filename,
        "pages": page_count,
        "characters": len(text),
        "total_chunks": len(chunks),
        "text_preview": text[:500],
        "chunk_previews": [chunk[:150] for chunk in chunks[:3]],
        "indexed": True
    }


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)


@app.post("/search")
def semantic_search(request: SearchRequest):
    matches = search_chunks(
        query=request.query,
        top_k=request.top_k,
    )

    return {
        "query": request.query,
        "total_results": len(matches),
        "results": matches,
    }




class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    document_id: str | None = None


@app.post("/ask")
def ask_question(request: AskRequest):
    """Answer questions using the selected PDF."""

    matches = search_chunks(
        query=request.question,
        top_k=request.top_k,
        document_id=request.document_id,
    )

    if not matches:
        return {
            "question": request.question,
            "answer": "No matching document content was found.",
            "sources": [],
        }

    context_parts = []

    for index, match in enumerate(matches, start=1):
        context_parts.append(
            f"[Source {index}]\n{match['text']}"
        )

    context = "\n\n".join(context_parts)

    answer = generate_answer(
        question=request.question,
        context=context,
    )

    sources = [
        {
            "source": index,
            "document_id": match["document_id"],
            "chunk_index": match["chunk_index"],
            "page_number": match.get("page_number", 0),
            "filename": match.get("filename", "unknown.pdf"),
            "similarity": match["similarity"],
            "text": match["text"],
        }
        for index, match in enumerate(matches, start=1)
    ]
    return {
        "question": request.question,
        "answer": answer,
        "sources": sources,
    }
