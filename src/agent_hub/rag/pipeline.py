"""Production RAG pipeline: ingest, hybrid retrieve, cache, and format context."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import structlog

from agent_hub.rag.chunker import DocumentChunker
from agent_hub.rag.embedder import Embedder
from agent_hub.rag.hybrid_ranker import rrf_merge
from agent_hub.rag.redis_store import RAGRedisStore
from agent_hub.rag.sparse_retriever import SparseRetriever
from agent_hub.rag.vector_store import ChunkDoc, SearchResult, VectorStore

if TYPE_CHECKING:
    from agent_hub.config.settings import Settings

logger = structlog.get_logger(__name__)


class RAGPipeline:
    """Enterprise knowledge-base RAG pipeline.

    PostgreSQL/pgvector is the canonical vector store. Redis is an optional hot
    layer for retrieval results and BM25 chunk snapshots. If Redis is down, the
    pipeline falls back to PostgreSQL and in-memory sparse indexes.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        redis_store: RAGRedisStore | None = None,
    ) -> None:
        self._settings = settings
        pg_dsn = settings.pg_dsn if settings else "postgresql://localhost:5432/agent_hub"
        chunk_size = settings.chunk_size if settings else 512
        chunk_overlap = settings.chunk_overlap if settings else 64
        self._top_k = settings.rag_top_k if settings else 5
        self._rrf_k = settings.rrf_k if settings else 60
        self._query_cache_enabled = bool(
            getattr(settings, "rag_query_cache_enabled", False)
        ) if settings else False
        self._sparse_cache_enabled = bool(
            getattr(settings, "rag_sparse_cache_enabled", False)
        ) if settings else False
        self._sparse_bootstrap_limit = int(
            getattr(settings, "rag_sparse_bootstrap_limit", 20_000)
        ) if settings else 20_000

        self._embedder = embedder or self._build_embedder(settings)
        dimension = self._resolve_dimension(settings, self._embedder)
        self._vector_store = vector_store or VectorStore(
            pg_dsn=pg_dsn,
            dimension=dimension,
            min_pool_size=int(getattr(settings, "pg_pool_min_size", 1)) if settings else 1,
            max_pool_size=int(getattr(settings, "pg_pool_max_size", 10)) if settings else 10,
        )
        self._chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._sparse = SparseRetriever()
        self._redis_store = redis_store or self._build_redis_store(settings)

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def vector_store(self) -> VectorStore:
        return self._vector_store

    @property
    def redis_store(self) -> RAGRedisStore | None:
        return self._redis_store

    async def ingest(
        self,
        user_id: str,
        namespace: str,
        content: str,
        doc_type: str = "text",
        metadata: dict[str, Any] | None = None,
        *,
        document_id: str | None = None,
        replace: bool = True,
    ) -> int:
        """Ingest a document into PostgreSQL vectors and sparse cache."""
        metadata = dict(metadata or {})
        document_id = document_id or str(metadata.get("document_id") or namespace)
        metadata.setdefault("source", namespace)
        metadata.setdefault("namespace", namespace)
        metadata.setdefault("document_id", document_id)
        metadata.setdefault("doc_type", doc_type)

        chunks = self._chunk_content(content, doc_type, metadata)
        if not chunks:
            logger.warning("rag_ingest_empty", user_id=user_id, namespace=namespace)
            return 0

        for chunk in chunks:
            chunk.document_id = document_id
            chunk.content_hash = self._hash_text(chunk.content)
            chunk.metadata.update(
                {
                    "namespace": namespace,
                    "document_id": document_id,
                    "chunk_index": chunk.chunk_index,
                    "content_hash": chunk.content_hash,
                }
            )

        embeddings = self._embedder.embed([chunk.content for chunk in chunks])
        await self._vector_store.ensure_schema()
        count = await self._vector_store.upsert(
            user_id,
            namespace,
            chunks,
            embeddings,
            document_id=document_id,
            doc_type=doc_type,
            metadata=metadata,
            replace=replace,
        )

        self._sparse.build_index(user_id, namespace, chunks)
        if self._redis_store is not None:
            await self._redis_store.invalidate_namespace(user_id=user_id, namespace=namespace)
            if self._sparse_cache_enabled:
                await self._redis_store.save_sparse_chunks(
                    user_id=user_id,
                    namespace=namespace,
                    chunks=chunks,
                )

        logger.info(
            "rag_ingest_complete",
            user_id=user_id,
            namespace=namespace,
            document_id=document_id,
            doc_type=doc_type,
            chunk_count=count,
        )
        return count

    async def retrieve(
        self,
        query: str,
        user_id: str,
        namespace: str | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Dense + sparse hybrid retrieval with RRF ranking."""
        top_k = top_k or self._top_k
        if not query.strip():
            return []

        await self._vector_store.ensure_schema()
        if self._redis_store is not None and self._query_cache_enabled:
            cached = await self._redis_store.get_retrieval_results(
                user_id=user_id,
                namespace=namespace,
                query=query,
                top_k=top_k,
            )
            if cached is not None:
                logger.info(
                    "rag_retrieve_cache_hit",
                    user_id=user_id,
                    namespace=namespace,
                    top_k=top_k,
                    result_count=len(cached),
                )
                return cached

        query_emb = self._embedder.embed_query(query)
        dense_results = await self._vector_store.search_dense_raw(
            query_emb,
            user_id,
            namespace,
            top_k=top_k * 2,
        )
        sparse_results = await self._retrieve_sparse(
            user_id=user_id,
            namespace=namespace,
            query=query,
            top_k=top_k * 2,
        )

        merged = rrf_merge(
            dense_results,
            sparse_results,
            k=self._rrf_k,
            top_k=top_k,
        )
        if self._redis_store is not None and self._query_cache_enabled:
            await self._redis_store.set_retrieval_results(
                user_id=user_id,
                namespace=namespace,
                query=query,
                top_k=top_k,
                results=merged,
            )

        logger.info(
            "rag_retrieve",
            user_id=user_id,
            namespace=namespace,
            query_preview=query[:50],
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            merged_count=len(merged),
        )
        return merged

    @staticmethod
    def format_context(results: list[SearchResult], max_tokens: int = 4000) -> str:
        """Format retrieval results as an LLM-ready context block."""
        if not results:
            return ""

        parts: list[str] = []
        total_chars = 0
        char_limit = max_tokens * 2
        for index, result in enumerate(results, start=1):
            source = result.metadata.get("source") or result.metadata.get("namespace") or "unknown"
            section = result.metadata.get("section", "")
            score = f"{result.score:.4f}"
            header = f"[来源 {index}: {source}; score={score}]"
            if section:
                header += f" ({section})"
            snippet = f"{header}\n{result.content}"
            if total_chars + len(snippet) > char_limit:
                break
            parts.append(snippet)
            total_chars += len(snippet)
        return "\n\n---\n\n".join(parts)

    async def delete_namespace(self, user_id: str, namespace: str) -> None:
        await self._vector_store.delete_by_namespace(user_id, namespace)
        self._sparse.remove_index(user_id, namespace)
        if self._redis_store is not None:
            await self._redis_store.invalidate_namespace(user_id=user_id, namespace=namespace)

    async def delete_document(self, user_id: str, namespace: str, document_id: str) -> int:
        deleted = await self._vector_store.delete_by_document(user_id, namespace, document_id)
        self._sparse.remove_index(user_id, namespace)
        if self._redis_store is not None:
            await self._redis_store.invalidate_namespace(user_id=user_id, namespace=namespace)
        return deleted

    async def healthcheck(self) -> dict[str, Any]:
        pg_status: dict[str, Any]
        try:
            pg_status = await self._vector_store.healthcheck()
        except Exception as exc:  # noqa: BLE001
            pg_status = {"ok": False, "error": str(exc)}

        redis_ok = None
        if self._redis_store is not None:
            redis_ok = await self._redis_store.ping()

        return {
            "postgres": pg_status,
            "redis": {"enabled": self._redis_store is not None, "ok": redis_ok},
            "embedding": {
                "provider": self._embedder.provider,
                "model": self._embedder.model_name,
                "dimension": self._safe_dimension(),
            },
            "retrieval": {
                "top_k": self._top_k,
                "rrf_k": self._rrf_k,
                "query_cache_enabled": self._query_cache_enabled,
                "sparse_cache_enabled": self._sparse_cache_enabled,
            },
        }

    async def close(self) -> None:
        await self._vector_store.close()
        if self._redis_store is not None:
            await self._redis_store.close()

    async def _retrieve_sparse(
        self,
        *,
        user_id: str,
        namespace: str | None,
        query: str,
        top_k: int,
    ) -> list[SearchResult]:
        if namespace is not None:
            await self._ensure_sparse_index(user_id, namespace)
            return self._sparse.search(user_id, namespace, query, top_k=top_k)

        namespaces = set(self._sparse.namespaces(user_id))
        try:
            namespaces.update(await self._vector_store.list_namespaces(user_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning("rag_sparse_namespace_load_failed", error=str(exc))
        for loaded_namespace in namespaces:
            await self._ensure_sparse_index(user_id, loaded_namespace)
        return self._sparse.search_all_namespaces(user_id, query, top_k=top_k)

    async def _ensure_sparse_index(self, user_id: str, namespace: str) -> None:
        if self._sparse.has_index(user_id, namespace):
            return

        chunks: list[ChunkDoc] | None = None
        if self._redis_store is not None and self._sparse_cache_enabled:
            chunks = await self._redis_store.load_sparse_chunks(
                user_id=user_id,
                namespace=namespace,
            )
        if chunks is None:
            try:
                chunks = await self._vector_store.list_chunks(
                    user_id,
                    namespace,
                    limit=self._sparse_bootstrap_limit,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("rag_sparse_chunk_load_failed", error=str(exc))
                chunks = []
            if not isinstance(chunks, list):
                chunks = []
            if self._redis_store is not None and self._sparse_cache_enabled and chunks:
                await self._redis_store.save_sparse_chunks(
                    user_id=user_id,
                    namespace=namespace,
                    chunks=chunks,
                )
        if chunks:
            self._sparse.build_index(user_id, namespace, chunks)

    def _chunk_content(
        self,
        content: str,
        doc_type: str,
        metadata: dict[str, Any],
    ) -> list[ChunkDoc]:
        if doc_type == "pdf":
            return self._chunker.chunk_pdf(content, metadata)
        if doc_type == "markdown":
            return self._chunker.chunk_markdown(content, metadata)
        return self._chunker.chunk_text(content, metadata)

    def _safe_dimension(self) -> int | None:
        try:
            return self._embedder.dimension
        except Exception:
            return getattr(self._settings, "embedding_dimension", None) if self._settings else None

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _build_embedder(settings: Settings | None) -> Embedder:
        if settings is None:
            return Embedder()
        return Embedder(
            model_name=settings.embedding_model,
            provider=getattr(settings, "embedding_provider", "local"),
            device=getattr(settings, "embedding_device", "") or None,
            api_key=(
                getattr(settings, "embedding_api_key", "")
                or getattr(settings, "llm_api_key", "")
            ),
            base_url=getattr(settings, "embedding_base_url", "") or None,
            dimension=getattr(settings, "embedding_dimension", 1024),
            batch_size=getattr(settings, "embedding_batch_size", 32),
            normalize=getattr(settings, "embedding_normalize", True),
            timeout=float(getattr(settings, "embedding_timeout_seconds", 30.0)),
        )

    @staticmethod
    def _build_redis_store(settings: Settings | None) -> RAGRedisStore | None:
        if settings is None:
            return None
        if not getattr(settings, "rag_redis_enabled", True):
            return None
        redis_url = getattr(settings, "redis_url", "")
        if not redis_url:
            return None
        return RAGRedisStore(
            redis_url,
            query_ttl=int(getattr(settings, "rag_query_cache_ttl", 300)),
            sparse_ttl=int(getattr(settings, "rag_sparse_cache_ttl", 86_400)),
        )

    @staticmethod
    def _resolve_dimension(settings: Settings | None, embedder: Embedder) -> int:
        if settings is not None:
            configured = getattr(settings, "embedding_dimension", None)
            if isinstance(configured, int) and configured > 0:
                return configured
        return embedder.dimension
