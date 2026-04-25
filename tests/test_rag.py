"""RAG 模块单元测试。

覆盖 Embedder、VectorStore、SparseRetriever、RRF 融合、
DocumentChunker、RAGPipeline 端到端。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_hub.rag.chunker import DocumentChunker
from agent_hub.rag.embedder import Embedder
from agent_hub.rag.hybrid_ranker import rrf_merge
from agent_hub.rag.sparse_retriever import SparseRetriever
from agent_hub.rag.vector_store import ChunkDoc, SearchResult, VectorStore

# ═══════════════════════════════════════════════════════
# Embedder 测试
# ═══════════════════════════════════════════════════════


class TestEmbedder:
    """Embedder 单元测试（mock SentenceTransformer）。"""

    def _make_embedder(self) -> Embedder:
        embedder = Embedder.__new__(Embedder)
        embedder._model_name = "test-model"
        embedder._device = "cpu"
        embedder._dimension = 4
        # mock model
        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        mock_model.get_sentence_embedding_dimension.return_value = 4
        embedder._model = mock_model
        return embedder

    def test_embed_batch(self) -> None:
        embedder = self._make_embedder()
        result = embedder.embed(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 4

    def test_embed_query(self) -> None:
        embedder = self._make_embedder()
        import numpy as np
        embedder._model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
        result = embedder.embed_query("hello")
        assert len(result) == 4
        assert isinstance(result, list)

    def test_dimension_property(self) -> None:
        embedder = self._make_embedder()
        assert embedder.dimension == 4

    def test_detect_device_cpu(self) -> None:
        device = Embedder._detect_device()
        assert device in ("cpu", "cuda")


# ═══════════════════════════════════════════════════════
# SparseRetriever 测试
# ═══════════════════════════════════════════════════════


class TestSparseRetriever:
    """BM25 稀疏检索测试（真实 BM25 实例）。"""

    def _build_retriever(self) -> SparseRetriever:
        retriever = SparseRetriever()
        chunks = [
            ChunkDoc(content="Python 是一种面向对象的编程语言", metadata={"source": "doc1"}),
            ChunkDoc(content="向量数据库用于存储和检索高维向量", metadata={"source": "doc2"}),
            ChunkDoc(content="机器学习是人工智能的一个分支", metadata={"source": "doc3"}),
            ChunkDoc(content="PostgreSQL 是一个强大的开源关系数据库", metadata={"source": "doc4"}),
            ChunkDoc(content="pgvector 为 PostgreSQL 添加了向量相似度搜索能力", metadata={"source": "doc5"}),
        ]
        retriever.build_index("u1", "test_ns", chunks)
        return retriever

    def test_build_index(self) -> None:
        retriever = self._build_retriever()
        assert retriever.has_index("u1", "test_ns")
        assert not retriever.has_index("u1", "nonexistent")

    def test_search_returns_results(self) -> None:
        retriever = self._build_retriever()
        results = retriever.search("u1", "test_ns", "向量数据库", top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        # 向量数据库相关内容应排在前面
        assert "向量" in results[0].content

    def test_search_empty_namespace(self) -> None:
        retriever = SparseRetriever()
        results = retriever.search("u1", "empty", "任意查询")
        assert results == []

    def test_search_all_namespaces(self) -> None:
        retriever = self._build_retriever()
        # 添加第二个命名空间
        retriever.build_index("u1", "ns2", [
            ChunkDoc(content="深度学习使用神经网络", metadata={"source": "doc6"}),
        ])
        results = retriever.search_all_namespaces("u1", "学习", top_k=5)
        assert len(results) > 0

    def test_remove_index(self) -> None:
        retriever = self._build_retriever()
        retriever.remove_index("u1", "test_ns")
        assert not retriever.has_index("u1", "test_ns")


# ═══════════════════════════════════════════════════════
# RRF 融合测试
# ═══════════════════════════════════════════════════════


class TestRRFMerge:
    """RRF (Reciprocal Rank Fusion) 融合排序测试。"""

    def test_basic_merge(self) -> None:
        dense = [
            SearchResult(content="A", score=0.9, chunk_id=1),
            SearchResult(content="B", score=0.8, chunk_id=2),
            SearchResult(content="C", score=0.7, chunk_id=3),
        ]
        sparse = [
            SearchResult(content="B", score=5.0, chunk_id=2),
            SearchResult(content="D", score=4.0, chunk_id=4),
            SearchResult(content="A", score=3.0, chunk_id=1),
        ]

        merged = rrf_merge(dense, sparse, k=60)

        # A 和 B 同时出现在两路，RRF 分数更高
        keys = [f"chunk_{r.chunk_id}" for r in merged]
        assert len(merged) == 4
        # A: 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226
        # B: 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252
        # B 应排最前
        assert merged[0].chunk_id == 2  # B

    def test_empty_dense(self) -> None:
        sparse = [SearchResult(content="X", score=1.0, chunk_id=10)]
        merged = rrf_merge([], sparse, k=60)
        assert len(merged) == 1
        assert merged[0].content == "X"

    def test_empty_sparse(self) -> None:
        dense = [SearchResult(content="Y", score=0.9, chunk_id=20)]
        merged = rrf_merge(dense, [], k=60)
        assert len(merged) == 1
        assert merged[0].content == "Y"

    def test_both_empty(self) -> None:
        merged = rrf_merge([], [], k=60)
        assert merged == []

    def test_top_k_limit(self) -> None:
        dense = [
            SearchResult(content=f"D{i}", score=1.0 - i * 0.1, chunk_id=i)
            for i in range(10)
        ]
        merged = rrf_merge(dense, [], k=60, top_k=3)
        assert len(merged) == 3

    def test_rrf_scores_are_positive(self) -> None:
        dense = [SearchResult(content="A", score=0.5, chunk_id=1)]
        sparse = [SearchResult(content="B", score=2.0, chunk_id=2)]
        merged = rrf_merge(dense, sparse, k=60)
        assert all(r.score > 0 for r in merged)

    def test_rrf_formula_accuracy(self) -> None:
        """验证 RRF 公式精确性。"""
        dense = [SearchResult(content="A", score=0.9, chunk_id=1)]
        sparse = [SearchResult(content="A", score=5.0, chunk_id=1)]
        merged = rrf_merge(dense, sparse, k=60)
        # A 在两路都排第 1：1/(60+1) + 1/(60+1) = 2/61
        expected = 2.0 / 61.0
        assert abs(merged[0].score - expected) < 1e-10


# ═══════════════════════════════════════════════════════
# DocumentChunker 测试
# ═══════════════════════════════════════════════════════


class TestDocumentChunker:
    """文档切块测试。"""

    def test_chunk_text_basic(self) -> None:
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 1
        assert all(isinstance(c, ChunkDoc) for c in chunks)

    def test_chunk_text_empty(self) -> None:
        chunker = DocumentChunker()
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   ") == []

    def test_chunk_text_preserves_content(self) -> None:
        chunker = DocumentChunker(chunk_size=2000, chunk_overlap=50)
        text = "这是一段测试内容，应该完整保留。"
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_chunk_text_large_chunk_splits(self) -> None:
        chunker = DocumentChunker(chunk_size=20, chunk_overlap=5)
        # 生成一段较长文本
        text = "这是一个测试。" * 50
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1

    def test_chunk_markdown_splits_by_heading(self) -> None:
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        md = """# 标题一

这是第一章内容。

## 标题二

这是第二章内容，比较详细。

### 标题三

这是第三节内容。
"""
        chunks = chunker.chunk_markdown(md)
        assert len(chunks) >= 1
        # 检查 section 元数据
        sections = [c.metadata.get("section", "") for c in chunks]
        assert any("标题" in s for s in sections)

    def test_chunk_markdown_empty(self) -> None:
        chunker = DocumentChunker()
        assert chunker.chunk_markdown("") == []

    def test_chunk_indices_sequential(self) -> None:
        chunker = DocumentChunker(chunk_size=30, chunk_overlap=5)
        text = "\n\n".join([f"段落{i}的内容是一些测试文本。" for i in range(10)])
        chunks = chunker.chunk_text(text)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_metadata_passed_through(self) -> None:
        chunker = DocumentChunker()
        meta = {"source": "test.txt", "author": "tester"}
        chunks = chunker.chunk_text("一些内容", meta)
        assert chunks[0].metadata["source"] == "test.txt"
        assert chunks[0].metadata["author"] == "tester"


# ═══════════════════════════════════════════════════════
# RAGPipeline 端到端测试（mock 组件）
# ═══════════════════════════════════════════════════════


class TestRAGPipeline:
    """RAGPipeline 端到端测试。"""

    def _make_pipeline(self):
        """创建带 mock 组件的 RAGPipeline。"""
        from agent_hub.rag.pipeline import RAGPipeline

        # mock embedder
        mock_embedder = MagicMock(spec=Embedder)
        mock_embedder.dimension = 4
        mock_embedder.embed.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]

        # mock vector_store
        mock_vs = AsyncMock(spec=VectorStore)
        mock_vs.upsert.return_value = 1
        mock_vs.search_dense_raw.return_value = [
            SearchResult(content="向量检索结果", score=0.9, metadata={"source": "vec"}, chunk_id=1),
        ]

        pipeline = RAGPipeline(
            settings=None,
            embedder=mock_embedder,
            vector_store=mock_vs,
        )
        return pipeline

    @pytest.mark.asyncio
    async def test_ingest_text(self) -> None:
        pipeline = self._make_pipeline()
        count = await pipeline.ingest("u1", "test_ns", "测试文档内容", doc_type="text")
        assert count >= 1
        pipeline._vector_store.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_markdown(self) -> None:
        pipeline = self._make_pipeline()
        md = "# 标题\n\n内容段落"
        count = await pipeline.ingest("u1", "md_ns", md, doc_type="markdown")
        assert count >= 1

    @pytest.mark.asyncio
    async def test_ingest_empty(self) -> None:
        pipeline = self._make_pipeline()
        count = await pipeline.ingest("u1", "ns", "", doc_type="text")
        assert count == 0

    @pytest.mark.asyncio
    async def test_retrieve_hybrid(self) -> None:
        pipeline = self._make_pipeline()

        # 构建 BM25 索引
        pipeline._sparse.build_index("u1", "ns", [
            ChunkDoc(content="向量检索结果", metadata={"source": "vec"}),
            ChunkDoc(content="其他内容", metadata={"source": "other"}),
        ])

        results = await pipeline.retrieve("向量检索", "u1", "ns", top_k=5)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_retrieve_empty(self) -> None:
        pipeline = self._make_pipeline()
        pipeline._vector_store.search_dense_raw.return_value = []
        results = await pipeline.retrieve("不存在的内容", "u1", "empty_ns")
        assert isinstance(results, list)

    def test_format_context(self) -> None:
        from agent_hub.rag.pipeline import RAGPipeline
        results = [
            SearchResult(
                content="这是第一段检索到的内容",
                score=0.9,
                metadata={"source": "doc1.md"},
            ),
            SearchResult(
                content="这是第二段检索到的内容",
                score=0.8,
                metadata={"source": "doc2.md", "section": "概述"},
            ),
        ]
        ctx = RAGPipeline.format_context(results)
        assert "doc1.md" in ctx
        assert "doc2.md" in ctx
        assert "概述" in ctx
        assert "---" in ctx

    def test_format_context_empty(self) -> None:
        from agent_hub.rag.pipeline import RAGPipeline
        assert RAGPipeline.format_context([]) == ""


# ═══════════════════════════════════════════════════════
# RetrievalAgent 集成测试（mock RAGPipeline）
# ═══════════════════════════════════════════════════════


class TestRetrievalAgentIntegration:
    """RetrievalAgent 与 RAGPipeline 集成测试。"""

    @pytest.mark.asyncio
    async def test_retrieval_success(self) -> None:
        from agent_hub.agents.retrieval_agent import RetrievalAgent
        from agent_hub.core.models import SubTask

        mock_rag = MagicMock()
        mock_rag.retrieve = AsyncMock(return_value=[
            SearchResult(content="检索结果内容", score=0.9, metadata={"source": "test.md"}),
        ])
        mock_rag.format_context.return_value = "[来源 1: test.md]\n检索结果内容"

        agent = RetrievalAgent(rag_pipeline=mock_rag, top_k=5)
        subtask = SubTask(
            subtask_id="st1",
            description="什么是向量数据库",
            required_agents=["retrieval_agent"],
        )
        result = await agent.run(subtask, "sess1", "u1")
        assert result.success
        assert "检索结果" in result.output

    @pytest.mark.asyncio
    async def test_retrieval_empty(self) -> None:
        from agent_hub.agents.retrieval_agent import RetrievalAgent
        from agent_hub.core.models import SubTask

        mock_rag = MagicMock()
        mock_rag.retrieve = AsyncMock(return_value=[])

        agent = RetrievalAgent(rag_pipeline=mock_rag)
        subtask = SubTask(
            subtask_id="st2",
            description="不存在的问题",
            required_agents=["retrieval_agent"],
        )
        result = await agent.run(subtask, "sess1", "u1")
        assert result.success
        assert "未在知识库中找到" in result.output

    @pytest.mark.asyncio
    async def test_retrieval_error_fallback(self) -> None:
        from agent_hub.agents.retrieval_agent import RetrievalAgent
        from agent_hub.core.models import SubTask

        mock_rag = AsyncMock()
        mock_rag.retrieve.side_effect = ConnectionError("pg down")

        agent = RetrievalAgent(rag_pipeline=mock_rag)
        subtask = SubTask(
            subtask_id="st3",
            description="测试查询",
            required_agents=["retrieval_agent"],
        )
        result = await agent.run(subtask, "sess1", "u1")
        assert result.success
        assert "暂不可用" in result.output
