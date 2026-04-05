"""API routes for NUST Bank RAG service."""

import json

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from .models import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    SourceListResponse,
    SourceInfo,
    AddDocumentRequest,
    AddDocumentResponse,
    BulkFAQDocument,
    BulkUploadResponse,
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


@router.post("/documents", response_model=AddDocumentResponse)
async def add_document(request: AddDocumentRequest):
    """Add a new Q&A document to the live knowledge base."""
    try:
        service = get_service()
        result = service.add_document(
            product=request.product,
            question=request.question,
            answer=request.answer,
        )
        return AddDocumentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/bulk", response_model=BulkUploadResponse)
async def bulk_add_documents(document: BulkFAQDocument):
    """
    Add multiple FAQ entries from a structured JSON body.

    Expected format:
    ```json
    {
      "categories": [
        {
          "category": "Product Name",
          "questions": [
            {"question": "...", "answer": "..."}
          ]
        }
      ]
    }
    ```
    """
    try:
        service = get_service()
        categories = [cat.model_dump() for cat in document.categories]
        result = service.add_documents_bulk(categories)
        return BulkUploadResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/bulk/upload", response_model=BulkUploadResponse)
async def bulk_upload_faq_file(file: UploadFile = File(...)):
    """
    Upload a JSON file containing FAQ entries and add them to the knowledge base.

    The file must follow the format:
    ```json
    {
      "categories": [
        {
          "category": "Product Name",
          "questions": [
            {"question": "...", "answer": "..."}
          ]
        }
      ]
    }
    ```
    """
    if not file.filename or not file.filename.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files are accepted.")

    try:
        raw = await file.read()
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    try:
        document = BulkFAQDocument(**data)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"File does not match the expected FAQ schema: {e}",
        )

    try:
        service = get_service()
        categories = [cat.model_dump() for cat in document.categories]
        result = service.add_documents_bulk(categories)
        return BulkUploadResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
