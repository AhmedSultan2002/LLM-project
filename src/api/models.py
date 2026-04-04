"""Pydantic models for API request/response."""

from pydantic import BaseModel, Field
from typing import Optional


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
