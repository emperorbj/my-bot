from typing import List, Optional
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int


class AskRequest(BaseModel):
    document_id: str = Field(..., description="ID returned from /documents/upload")
    question: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    document_id: str
    question: str
    answer: str


class SearchResult(BaseModel):
    source: str
    page: Optional[int] = None
    snippet: str


class SearchResponse(BaseModel):
    document_id: str
    query: str
    results: List[SearchResult]


class DocumentInfo(BaseModel):
    document_id: str
    filename: Optional[str] = None


class DocumentsListResponse(BaseModel):
    documents: List[DocumentInfo]


class ErrorResponse(BaseModel):
    detail: str