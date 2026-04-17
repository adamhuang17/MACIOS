"""RAG 知识检索模块。

提供 Dense (pgvector) + Sparse (BM25) 混合检索，
RRF 融合排序，文档语义切块，Query 重写，相关性评估，
以及统一的 RAG Pipeline 主控。
"""

from agent_hub.rag.chunker import DocumentChunker
from agent_hub.rag.embedder import Embedder
from agent_hub.rag.hybrid_ranker import rrf_merge
from agent_hub.rag.pipeline import RAGPipeline
from agent_hub.rag.query_rewriter import QueryRewriter
from agent_hub.rag.relevance_evaluator import RelevanceEvaluator
from agent_hub.rag.sparse_retriever import SparseRetriever
from agent_hub.rag.vector_store import ChunkDoc, SearchResult, VectorStore

__all__ = [
    "ChunkDoc",
    "DocumentChunker",
    "Embedder",
    "QueryRewriter",
    "RAGPipeline",
    "RelevanceEvaluator",
    "SearchResult",
    "SparseRetriever",
    "VectorStore",
    "rrf_merge",
]
