"""BM25 稀疏检索。

基于 rank_bm25 + jieba 分词，按 (user_id, namespace) 维护独立索引。
"""

from __future__ import annotations

from typing import Any

import jieba
import structlog
from rank_bm25 import BM25Okapi

from agent_hub.rag.vector_store import ChunkDoc, SearchResult

logger = structlog.get_logger(__name__)


class SparseRetriever:
    """BM25 稀疏检索器。

    为每个 ``(user_id, namespace)`` 组合维护一份独立的 BM25 索引。

    Example::

        retriever = SparseRetriever()
        retriever.build_index("u1", "docs", chunks)
        results = retriever.search("u1", "docs", "如何使用向量数据库", top_k=5)
    """

    def __init__(self) -> None:
        # key: (user_id, namespace) → (BM25Okapi, original_chunks)
        self._indices: dict[tuple[str, str], tuple[BM25Okapi, list[ChunkDoc]]] = {}

    def build_index(
        self,
        user_id: str,
        namespace: str,
        chunks: list[ChunkDoc],
    ) -> None:
        """为给定命名空间构建 BM25 索引。

        Args:
            user_id: 用户 ID。
            namespace: 命名空间。
            chunks: 用于构建索引的文档切块。
        """
        if not chunks:
            return

        tokenized = [list(jieba.cut(c.content)) for c in chunks]
        bm25 = BM25Okapi(tokenized)
        self._indices[(user_id, namespace)] = (bm25, list(chunks))
        logger.info(
            "sparse_index_built",
            user_id=user_id,
            namespace=namespace,
            doc_count=len(chunks),
        )

    def search(
        self,
        user_id: str,
        namespace: str,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """BM25 检索。

        Args:
            user_id: 用户 ID。
            namespace: 命名空间。
            query: 查询文本。
            top_k: 返回的最大结果数。

        Returns:
            按 BM25 分数降序排列的 SearchResult 列表。
        """
        key = (user_id, namespace)
        if key not in self._indices:
            return []

        bm25, chunks = self._indices[key]
        query_tokens = list(jieba.cut(query))
        scores = bm25.get_scores(query_tokens)

        # 取 Top-K（按分数降序）
        indexed_scores: list[tuple[int, float]] = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results: list[SearchResult] = []
        for idx, score in indexed_scores:
            if score <= 0:
                continue
            chunk = chunks[idx]
            results.append(SearchResult(
                content=chunk.content,
                score=float(score),
                metadata=chunk.metadata,
                chunk_id=idx,
            ))
        return results

    def search_all_namespaces(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """跨该用户所有命名空间检索。

        Args:
            user_id: 用户 ID。
            query: 查询文本。
            top_k: 返回的最大结果数。

        Returns:
            按 BM25 分数降序排列的合并结果。
        """
        all_results: list[SearchResult] = []
        for (uid, ns) in self._indices:
            if uid == user_id:
                all_results.extend(self.search(uid, ns, query, top_k))

        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:top_k]

    def has_index(self, user_id: str, namespace: str) -> bool:
        """判断是否已有索引。"""
        return (user_id, namespace) in self._indices

    def remove_index(self, user_id: str, namespace: str) -> None:
        """移除指定命名空间的索引。"""
        self._indices.pop((user_id, namespace), None)
