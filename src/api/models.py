"""Pydantic models for API request/response."""

from pydantic import BaseModel, Field
from typing import Optional, List


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    query: str = Field(..., min_length=1, max_length=1000, description="User question")


class SourceInfo(BaseModel):
    """Source document information."""

    product: str
    question: str
    score: float


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    query: str
    answer: str
    sources: list[SourceInfo]
    latency_seconds: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    message: Optional[str] = None


class SourceListResponse(BaseModel):
    """Response for available sources."""

    sources: list[str]
    total_count: int


class AddDocumentRequest(BaseModel):
    """Request model for adding a new document to the knowledge base."""

    product: str = Field(..., min_length=1, max_length=200, description="Product or category name")
    question: str = Field(..., min_length=5, max_length=1000, description="The question this document answers")
    answer: str = Field(..., min_length=5, max_length=5000, description="The answer to the question")


class AddDocumentResponse(BaseModel):
    """Response after adding a new document."""

    success: bool
    message: str
    total_documents: int


class FAQEntry(BaseModel):
    """A single question/answer pair within a FAQ category."""

    question: str = Field(..., min_length=5, max_length=1000)
    answer: str = Field(..., min_length=5, max_length=5000)


class FAQCategory(BaseModel):
    """A category containing a list of FAQ entries."""

    category: str = Field(..., min_length=1, max_length=200, description="Product or category name")
    questions: List[FAQEntry] = Field(..., min_length=1)


class BulkFAQDocument(BaseModel):
    """Top-level structure for a bulk FAQ JSON upload."""

    categories: List[FAQCategory] = Field(..., min_length=1)


class BulkUploadResponse(BaseModel):
    """Response after a bulk FAQ upload."""

    success: bool
    message: str
    added_count: int
    total_documents: int
    errors: List[str] = Field(default_factory=list)
