"""RAG Pipeline 主控。

串联 ETL 入库 + 混合检索 + 上下文拼装。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from agent_hub.rag.chunker import DocumentChunker
from agent_hub.rag.embedder import Embedder
from agent_hub.rag.hybrid_ranker import rrf_merge
from agent_hub.rag.sparse_retriever import SparseRetriever
from agent_hub.rag.vector_store import ChunkDoc, SearchResult, VectorStore

if TYPE_CHECKING:
    from agent_hub.config.settings import Settings

logger = structlog.get_logger(__name__)


class RAGPipeline:
    """RAG Pipeline：文档入库 + Dense/Sparse 混合检索 + 上下文拼装。

    Args:
        settings: 全局配置（用于 pg_dsn、embedding_model 等）。
        embedder: 可选注入的 Embedder 实例（测试时可 mock）。
        vector_store: 可选注入的 VectorStore 实例。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        # 配置
        pg_dsn = settings.pg_dsn if settings else "postgresql://localhost:5432/agent_hub"
        model_name = settings.embedding_model if settings else "BAAI/bge-large-zh-v1.5"
        chunk_size = settings.chunk_size if settings else 512
        chunk_overlap = settings.chunk_overlap if settings else 64
        self._top_k = settings.rag_top_k if settings else 5
        self._rrf_k = settings.rrf_k if settings else 60

        # 组件
        self._embedder = embedder or Embedder(model_name=model_name)
        self._vector_store = vector_store or VectorStore(
            pg_dsn=pg_dsn, dimension=self._embedder.dimension,
        )
        self._chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._sparse = SparseRetriever()

    @property
    def embedder(self) -> Embedder:
        """公开 Embedder 实例（供记忆层复用）。"""
        return self._embedder

    @property
    def vector_store(self) -> VectorStore:
        """公开 VectorStore 实例（供记忆层复用）。"""
        return self._vector_store

    # ── ETL 入库 ─────────────────────────────────────

    async def ingest(
        self,
        user_id: str,
        namespace: str,
        content: str,
        doc_type: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """文档入库：切块 → 向量化 → pgvector 写入 + BM25 索引构建。

        Args:
            user_id: 用户 ID。
            namespace: 命名空间/文档名。
            content: 文档内容（文本或文件路径）。
            doc_type: 文档类型：``"text"`` / ``"markdown"`` / ``"pdf"``。
            metadata: 附加元数据。

        Returns:
            入库的切块数量。
        """
        metadata = metadata or {}

        # 1. 切块
        if doc_type == "pdf":
            chunks = self._chunker.chunk_pdf(content, metadata)
        elif doc_type == "markdown":
            chunks = self._chunker.chunk_markdown(content, metadata)
        else:
            chunks = self._chunker.chunk_text(content, metadata)

        if not chunks:
            logger.warning("rag_ingest_empty", user_id=user_id, namespace=namespace)
            return 0

        # 2. 向量化
        texts = [c.content for c in chunks]
        embeddings = self._embedder.embed(texts)

        # 3. pgvector 写入
        await self._vector_store.ensure_schema()
        count = await self._vector_store.upsert(user_id, namespace, chunks, embeddings)

        # 4. BM25 索引构建
        self._sparse.build_index(user_id, namespace, chunks)

        logger.info(
            "rag_ingest_complete",
            user_id=user_id,
            namespace=namespace,
            doc_type=doc_type,
            chunk_count=count,
        )
        return count

    # ── 混合检索 ─────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        user_id: str,
        namespace: str | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Dense + Sparse 混合检索，RRF 融合排序。

        Args:
            query: 用户查询文本。
            user_id: 用户 ID。
            namespace: 命名空间（None 表示搜索该用户所有空间）。
            top_k: 返回的最大结果数。

        Returns:
            按 RRF 分数降序排列的 SearchResult 列表。
        """
        top_k = top_k or self._top_k

        # Dense 检索
        query_emb = self._embedder.embed_query(query)
        dense_results = await self._vector_store.search_dense_raw(
            query_emb, user_id, namespace, top_k=top_k * 2,
        )

        # Sparse 检索
        if namespace is not None:
            sparse_results = self._sparse.search(
                user_id, namespace, query, top_k=top_k * 2,
            )
        else:
            sparse_results = self._sparse.search_all_namespaces(
                user_id, query, top_k=top_k * 2,
            )

        # RRF 融合
        merged = rrf_merge(
            dense_results,
            sparse_results,
            k=self._rrf_k,
            top_k=top_k,
        )

        logger.info(
            "rag_retrieve",
            query_preview=query[:50],
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            merged_count=len(merged),
        )
        return merged

    # ── 上下文拼装 ───────────────────────────────────

    @staticmethod
    def format_context(results: list[SearchResult], max_tokens: int = 4000) -> str:
        """将检索结果拼装为 LLM 上下文字符串。

        Args:
            results: 按相关度排序的检索结果。
            max_tokens: 最大 token 数估算（按字符 / 2 估算）。

        Returns:
            格式化的上下文字符串。
        """
        if not results:
            return ""

        parts: list[str] = []
        total_chars = 0
        char_limit = max_tokens * 2  # token ≈ 字符 / 2

        for i, r in enumerate(results, start=1):
            source = r.metadata.get("source", r.metadata.get("namespace", "未知"))
            section = r.metadata.get("section", "")
            header = f"[来源 {i}: {source}]"
            if section:
                header += f" ({section})"

            snippet = f"{header}\n{r.content}"
            snippet_len = len(snippet)

            if total_chars + snippet_len > char_limit:
                break

            parts.append(snippet)
            total_chars += snippet_len

        return "\n\n---\n\n".join(parts)

    # ── 清理 ─────────────────────────────────────────

    async def delete_namespace(self, user_id: str, namespace: str) -> None:
        """删除指定命名空间的所有数据。"""
        await self._vector_store.delete_by_namespace(user_id, namespace)
        self._sparse.remove_index(user_id, namespace)

    async def close(self) -> None:
        """关闭资源。"""
        await self._vector_store.close()
