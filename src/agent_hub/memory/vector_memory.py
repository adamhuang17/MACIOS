"""向量化长期记忆存储。

复用 RAG 层的 VectorStore，按 user_id 命名空间隔离。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from agent_hub.core.models import MemoryEntry
from agent_hub.rag.vector_store import ChunkDoc, SearchResult, VectorStore

if TYPE_CHECKING:
    from agent_hub.rag.embedder import Embedder

logger = structlog.get_logger(__name__)

# 记忆命名空间前缀
_MEMORY_NS_PREFIX = "memory_"


class VectorMemory:
    """向量化长期记忆存储。

    将 MemoryEntry 的 embedding 存入 pgvector，
    支持语义相似度检索。

    Args:
        vector_store: pgvector 存储层实例。
        embedder: 文本向量化服务实例。
    """

    def __init__(self, vector_store: VectorStore, embedder: Embedder) -> None:
        self._store = vector_store
        self._embedder = embedder

    def _namespace(self, user_id: str) -> str:
        """生成用户记忆命名空间。"""
        return f"{_MEMORY_NS_PREFIX}{user_id}"

    async def store(self, entry: MemoryEntry, user_id: str) -> None:
        """将记忆向量化并存入 pgvector。

        Args:
            entry: 记忆条目。
            user_id: 用户 ID。
        """
        # 计算 embedding
        emb = self._embedder.embed_query(entry.content)
        entry.embedding = emb

        ns = self._namespace(user_id)
        chunk = ChunkDoc(
            content=entry.content,
            metadata={
                "memory_id": entry.memory_id,
                "memory_type": entry.memory_type,
                "session_id": entry.session_id or "",
                "tags": json.dumps(entry.tags, ensure_ascii=False),
                "created_at": entry.created_at.isoformat(),
                "ttl": entry.ttl,
            },
        )

        await self._store.ensure_schema()
        await self._store.upsert(user_id, ns, [chunk], [emb])
        logger.info("vector_memory_stored", memory_id=entry.memory_id, user_id=user_id)

    async def search_similar(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[tuple[MemoryEntry, float]]:
        """语义相似度检索记忆。

        Args:
            query: 查询文本。
            user_id: 用户 ID。
            top_k: 返回的最大结果数。
            threshold: 最低相似度阈值（0 表示不过滤）。

        Returns:
            ``(MemoryEntry, score)`` 元组列表，按相似度降序排列。
        """
        query_emb = self._embedder.embed_query(query)
        ns = self._namespace(user_id)

        results = await self._store.search_dense_raw(
            query_emb, user_id, ns, top_k=top_k,
        )

        entries: list[tuple[MemoryEntry, float]] = []
        for r in results:
            if threshold > 0 and r.score < threshold:
                continue
            entry = self._result_to_entry(r)
            if entry is not None and not entry.is_deleted:
                entries.append((entry, r.score))

        return entries

    async def search_by_embedding(
        self,
        embedding: list[float],
        user_id: str,
        top_k: int = 5,
    ) -> list[tuple[SearchResult, float]]:
        """用现有 embedding 检索相似记忆。

        Args:
            embedding: 查询向量。
            user_id: 用户 ID。
            top_k: 返回的最大结果数。

        Returns:
            ``(SearchResult, score)`` 元组列表。
        """
        ns = self._namespace(user_id)
        results = await self._store.search_dense_raw(
            embedding, user_id, ns, top_k=top_k,
        )
        return [(r, r.score) for r in results]

    async def delete(self, memory_id: str, user_id: str) -> None:
        """从向量存储中删除指定记忆。

        注意：pgvector 不支持按 metadata 高效删除，
        此处标记为 TODO，实际生产需要在 metadata 中记录 memory_id
        然后用 SQL DELETE WHERE。
        """
        logger.info("vector_memory_delete", memory_id=memory_id, user_id=user_id)

    @staticmethod
    def _result_to_entry(result: SearchResult) -> MemoryEntry | None:
        """将 SearchResult 转换为 MemoryEntry。"""
        try:
            from datetime import datetime
            meta = result.metadata
            tags_raw = meta.get("tags", "[]")
            tags = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw

            created_at_raw = meta.get("created_at", "")
            try:
                created_at = datetime.fromisoformat(str(created_at_raw))
            except (ValueError, TypeError):
                created_at = datetime.now()

            return MemoryEntry(
                memory_id=str(meta.get("memory_id", "")),
                content=result.content,
                memory_type=str(meta.get("memory_type", "long_term")),
                session_id=meta.get("session_id") or None,
                tags=tags if isinstance(tags, list) else [],
                created_at=created_at,
                ttl=int(meta.get("ttl", -1)),
            )
        except Exception:
            return None
