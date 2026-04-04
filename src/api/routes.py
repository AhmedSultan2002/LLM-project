"""API routes for NUST Bank RAG service."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from .models import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    SourceListResponse,
    SourceInfo,
)
from .service import get_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        service = get_service()
        _ = service.retriever
        return HealthResponse(status="ok", message="Service is healthy")
    except Exception as e:
        return HealthResponse(status="error", message=str(e))


@router.post("/query", response_model=QueryResponse)
async def submit_query(request: QueryRequest):
    """Submit a customer query to the RAG pipeline."""
    try:
        service = get_service()
        result = service.query(request.query)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        sources = [
            SourceInfo(
                product=s["product"],
                question=s["question"],
                score=s["score"],
            )
            for s in result.get("sources", [])
        ]

        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            sources=sources,
            latency_seconds=result["latency_seconds"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources", response_model=SourceListResponse)
async def get_sources():
    """Get available document sources."""
    try:
        service = get_service()
        sources = service.get_sources()
        return SourceListResponse(sources=sources, total_count=len(sources))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
