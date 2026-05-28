"""Redis-backed hot cache for RAG retrieval and sparse indexes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import TYPE_CHECKING

import structlog

from agent_hub.rag.vector_store import ChunkDoc, SearchResult

if TYPE_CHECKING:
    from redis.asyncio.client import Redis

logger = structlog.get_logger(__name__)


class RAGRedisStore:
    """Redis helper for production RAG.

    Redis is deliberately treated as a hot cache: PostgreSQL remains canonical.
    If Redis is unavailable, callers can continue with dense pgvector retrieval
    and rebuild sparse indexes from PostgreSQL.
    """

    def __init__(
        self,
        redis_url: str,
        *,
        query_ttl: int = 300,
        sparse_ttl: int = 86_400,
        key_prefix: str = "rag:v1",
    ) -> None:
        self._redis_url = redis_url
        self._query_ttl = query_ttl
        self._sparse_ttl = sparse_ttl
        self._key_prefix = key_prefix.rstrip(":")
        self._redis: Redis | None = None

    async def _get_redis(self) -> Redis:
        if self._redis is None:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=3,
            )
        return self._redis

    async def ping(self) -> bool:
        try:
            redis = await self._get_redis()
            return bool(await redis.ping())
        except Exception as exc:  # noqa: BLE001
            logger.warning("rag_redis_ping_failed", error=str(exc))
            return False

    async def get_retrieval_results(
        self,
        *,
        user_id: str,
        namespace: str | None,
        query: str,
        top_k: int,
    ) -> list[SearchResult] | None:
        try:
            redis = await self._get_redis()
            key = await self._retrieval_key(user_id, namespace, query, top_k)
            raw = await redis.get(key)
            if raw is None:
                return None
            payload = json.loads(raw)
            return [
                SearchResult(
                    content=item["content"],
                    score=float(item["score"]),
                    metadata=dict(item.get("metadata") or {}),
                    chunk_id=item.get("chunk_id"),
                    namespace=item.get("namespace"),
                    document_id=item.get("document_id"),
                )
                for item in payload.get("results", [])
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("rag_retrieval_cache_get_failed", error=str(exc))
            return None

    async def set_retrieval_results(
        self,
        *,
        user_id: str,
        namespace: str | None,
        query: str,
        top_k: int,
        results: list[SearchResult],
    ) -> None:
        if not results:
            return
        try:
            redis = await self._get_redis()
            key = await self._retrieval_key(user_id, namespace, query, top_k)
            payload = {
                "results": [
                    {
                        "content": r.content,
                        "score": r.score,
                        "metadata": r.metadata,
                        "chunk_id": r.chunk_id,
                        "namespace": r.namespace,
                        "document_id": r.document_id,
                    }
                    for r in results
                ]
            }
            await redis.setex(key, self._query_ttl, json.dumps(payload, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            logger.warning("rag_retrieval_cache_set_failed", error=str(exc))

    async def save_sparse_chunks(
        self,
        *,
        user_id: str,
        namespace: str,
        chunks: list[ChunkDoc],
    ) -> None:
        try:
            redis = await self._get_redis()
            key = self._sparse_key(user_id, namespace)
            payload = {
                "chunks": [
                    {
                        **asdict(chunk),
                        "metadata": chunk.metadata,
                    }
                    for chunk in chunks
                ]
            }
            await redis.setex(key, self._sparse_ttl, json.dumps(payload, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            logger.warning("rag_sparse_cache_save_failed", error=str(exc))

    async def load_sparse_chunks(
        self,
        *,
        user_id: str,
        namespace: str,
    ) -> list[ChunkDoc] | None:
        try:
            redis = await self._get_redis()
            raw = await redis.get(self._sparse_key(user_id, namespace))
            if raw is None:
                return None
            payload = json.loads(raw)
            return [
                ChunkDoc(
                    content=item["content"],
                    metadata=dict(item.get("metadata") or {}),
                    chunk_index=int(item.get("chunk_index", 0)),
                    document_id=item.get("document_id"),
                    content_hash=item.get("content_hash"),
                )
                for item in payload.get("chunks", [])
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("rag_sparse_cache_load_failed", error=str(exc))
            return None

    async def invalidate_namespace(self, *, user_id: str, namespace: str) -> None:
        """Invalidate query cache by bumping the namespace version."""
        try:
            redis = await self._get_redis()
            await redis.incr(self._namespace_version_key(user_id, namespace))
            await redis.incr(self._namespace_version_key(user_id, "*"))
            await redis.delete(self._sparse_key(user_id, namespace))
        except Exception as exc:  # noqa: BLE001
            logger.warning("rag_namespace_cache_invalidate_failed", error=str(exc))

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def _retrieval_key(
        self,
        user_id: str,
        namespace: str | None,
        query: str,
        top_k: int,
    ) -> str:
        version = await self._namespace_version(user_id, namespace)
        normalized = " ".join(query.strip().lower().split())
        digest = hashlib.sha256(
            f"{user_id}:{namespace or '*'}:{version}:{top_k}:{normalized}".encode("utf-8")
        ).hexdigest()
        return f"{self._key_prefix}:query:{digest}"

    async def _namespace_version(self, user_id: str, namespace: str | None) -> str:
        version_namespace = namespace or "*"
        try:
            redis = await self._get_redis()
            value = await redis.get(
                self._namespace_version_key(user_id, version_namespace)
            )
            return str(value or "0")
        except Exception:
            return "0"

    def _namespace_version_key(self, user_id: str, namespace: str) -> str:
        return f"{self._key_prefix}:nsver:{user_id}:{namespace}"

    def _sparse_key(self, user_id: str, namespace: str) -> str:
        digest = hashlib.sha256(f"{user_id}:{namespace}".encode("utf-8")).hexdigest()
        return f"{self._key_prefix}:sparse:{digest}"
