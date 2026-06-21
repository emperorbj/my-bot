import os
import shutil
import tempfile
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import rag
from schemas import (
    UploadResponse,
    AskRequest,
    AskResponse,
    SearchResponse,
    SearchResult,
    DocumentsListResponse,
    DocumentInfo,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_api")

app = FastAPI(
    title="PDF RAG API",
    description="Upload a PDF, then ask questions about it (hybrid dense+sparse retrieval over Qdrant).",
    version="1.0.0",
)

# Allow your frontend to call this API directly from the browser.
# Restrict allow_origins to your actual frontend URL(s) before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    # Make sure the collection + payload index exist before serving traffic.
    # This also patches collections created before the index fix was added.
    try:
        rag.create_collection_if_missing()
    except Exception:
        logger.exception(
            "Could not ensure Qdrant collection/index on startup "
            "(will retry on first upload)."
        )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save the upload to a temp file so PyPDFLoader (which needs a path) can read it.
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
    finally:
        file.file.close()

    try:
        document_id, num_pages, num_chunks = rag.index_pdf(
            file_path=tmp_path, filename=file.filename
        )
    except Exception as e:
        logger.exception("Failed to index uploaded PDF")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")
    finally:
        os.remove(tmp_path)

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        pages=num_pages,
        chunks=num_chunks,
    )


@app.post("/documents/ask", response_model=AskResponse)
def ask_document(payload: AskRequest):
    if not rag.document_exists(payload.document_id):
        raise HTTPException(status_code=404, detail="document_id not found.")

    try:
        answer = rag.ask_question(
            question=payload.question, document_id=payload.document_id
        )
    except Exception as e:
        logger.exception("Failed to answer question")
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {e}")

    return AskResponse(
        document_id=payload.document_id,
        question=payload.question,
        answer=answer,
    )


@app.get("/documents/{document_id}/search", response_model=SearchResponse)
def search_document(document_id: str, q: str):
    if not rag.document_exists(document_id):
        raise HTTPException(status_code=404, detail="document_id not found.")

    raw_results = rag.search(query=q, document_id=document_id)
    results = [SearchResult(**r) for r in raw_results]

    return SearchResponse(document_id=document_id, query=q, results=results)


@app.get("/documents", response_model=DocumentsListResponse)
def list_documents():
    docs = rag.list_documents()
    return DocumentsListResponse(documents=[DocumentInfo(**d) for d in docs])