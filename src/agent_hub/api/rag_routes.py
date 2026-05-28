"""Knowledge-base RAG HTTP routes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from agent_hub.rag.pipeline import RAGPipeline


class RAGIngestRequest(BaseModel):
    user_id: str = Field(..., description="Tenant/user id")
    namespace: str = Field(..., description="Knowledge-base namespace")
    content: str = Field(..., description="Raw text, markdown, or PDF path")
    doc_type: str = Field(default="text", description="text, markdown, or pdf")
    document_id: str | None = Field(default=None, description="Stable document id")
    metadata: dict[str, Any] = Field(default_factory=dict)
    replace: bool = Field(default=True, description="Replace existing document chunks")


class RAGIngestResponse(BaseModel):
    user_id: str
    namespace: str
    document_id: str
    chunk_count: int
    status: str


class RAGRetrieveRequest(BaseModel):
    user_id: str = Field(..., description="Tenant/user id")
    query: str = Field(..., description="User query")
    namespace: str | None = Field(default=None, description="Optional namespace filter")
    top_k: int = Field(default=5, ge=1, le=50)


class RAGResultItem(BaseModel):
    content: str
    score: float
    metadata: dict[str, Any]
    chunk_id: int | None = None
    namespace: str | None = None
    document_id: str | None = None


class RAGRetrieveResponse(BaseModel):
    results: list[RAGResultItem]
    context: str


RAGProvider = RAGPipeline | Callable[[], RAGPipeline]


def build_rag_router(rag_provider: RAGProvider) -> APIRouter:
    router = APIRouter(prefix="/api/rag", tags=["rag"])

    def get_rag() -> RAGPipeline:
        if callable(rag_provider):
            return rag_provider()
        return rag_provider

    @router.post("/ingest", response_model=RAGIngestResponse)
    async def ingest(req: RAGIngestRequest) -> RAGIngestResponse:
        rag = get_rag()
        try:
            chunk_count = await rag.ingest(
                user_id=req.user_id,
                namespace=req.namespace,
                content=req.content,
                doc_type=req.doc_type,
                metadata=req.metadata,
                document_id=req.document_id,
                replace=req.replace,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"RAG ingest failed: {exc}") from exc

        return RAGIngestResponse(
            user_id=req.user_id,
            namespace=req.namespace,
            document_id=req.document_id or str(req.metadata.get("document_id") or req.namespace),
            chunk_count=chunk_count,
            status="ok",
        )

    @router.post("/retrieve", response_model=RAGRetrieveResponse)
    async def retrieve(req: RAGRetrieveRequest) -> RAGRetrieveResponse:
        rag = get_rag()
        try:
            results = await rag.retrieve(
                query=req.query,
                user_id=req.user_id,
                namespace=req.namespace,
                top_k=req.top_k,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"RAG retrieve failed: {exc}") from exc

        return RAGRetrieveResponse(
            results=[
                RAGResultItem(
                    content=result.content,
                    score=result.score,
                    metadata=result.metadata,
                    chunk_id=result.chunk_id,
                    namespace=result.namespace,
                    document_id=result.document_id,
                )
                for result in results
            ],
            context=rag.format_context(results),
        )

    @router.delete("/namespaces/{namespace}")
    async def delete_namespace(
        namespace: str,
        user_id: str = Query(..., description="Tenant/user id"),
    ) -> dict[str, str]:
        rag = get_rag()
        try:
            await rag.delete_namespace(user_id=user_id, namespace=namespace)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"RAG delete failed: {exc}") from exc
        return {"status": "ok", "user_id": user_id, "namespace": namespace}

    @router.delete("/documents/{namespace}/{document_id}")
    async def delete_document(
        namespace: str,
        document_id: str,
        user_id: str = Query(..., description="Tenant/user id"),
    ) -> dict[str, object]:
        rag = get_rag()
        try:
            deleted = await rag.delete_document(
                user_id=user_id,
                namespace=namespace,
                document_id=document_id,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"RAG delete failed: {exc}") from exc
        return {
            "status": "ok",
            "user_id": user_id,
            "namespace": namespace,
            "document_id": document_id,
            "deleted_chunks": deleted,
        }

    @router.get("/health")
    async def health() -> dict[str, Any]:
        rag = get_rag()
        return await rag.healthcheck()

    return router
