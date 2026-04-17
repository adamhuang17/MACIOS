"""RAG 评测引擎。

支持 Precision@K, Recall@K, MRR 等标准 IR 指标。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_hub.eval.dataset import QAPair
from agent_hub.rag.vector_store import SearchResult


@dataclass
class EvalResult:
    """单条评测结果。"""

    question: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    metadata: dict = field(default_factory=dict)


@dataclass
class AggregateResult:
    """汇总评测指标。"""

    num_queries: int
    avg_precision_at_k: float
    avg_recall_at_k: float
    mrr: float  # Mean Reciprocal Rank
    per_query: list[EvalResult] = field(default_factory=list)


class RAGEvaluator:
    """RAG 检索质量评测器。

    Args:
        top_k: 评估 top-K 检索结果。
    """

    def __init__(self, top_k: int = 5) -> None:
        self._top_k = top_k

    def evaluate_single(
        self,
        qa: QAPair,
        results: list[SearchResult],
    ) -> EvalResult:
        """评测单条 QA 对。"""
        retrieved_ids = [
            r.metadata.get("doc_id", r.chunk_id)
            for r in results[:self._top_k]
        ]
        relevant_set = set(qa.relevant_doc_ids)

        precision = self._precision_at_k(retrieved_ids, relevant_set)
        recall = self._recall_at_k(retrieved_ids, relevant_set)
        rr = self._reciprocal_rank(retrieved_ids, relevant_set)

        return EvalResult(
            question=qa.question,
            retrieved_ids=retrieved_ids,
            relevant_ids=qa.relevant_doc_ids,
            precision_at_k=precision,
            recall_at_k=recall,
            reciprocal_rank=rr,
        )

    def aggregate(self, results: list[EvalResult]) -> AggregateResult:
        """汇总多条评测结果。"""
        if not results:
            return AggregateResult(
                num_queries=0,
                avg_precision_at_k=0.0,
                avg_recall_at_k=0.0,
                mrr=0.0,
            )

        n = len(results)
        return AggregateResult(
            num_queries=n,
            avg_precision_at_k=sum(r.precision_at_k for r in results) / n,
            avg_recall_at_k=sum(r.recall_at_k for r in results) / n,
            mrr=sum(r.reciprocal_rank for r in results) / n,
            per_query=results,
        )

    # ── 指标计算 ─────────────────────────────────────

    @staticmethod
    def _precision_at_k(retrieved: list[str], relevant: set[str]) -> float:
        """Precision@K: 检索结果中相关文档的比例。"""
        if not retrieved:
            return 0.0
        hits = sum(1 for rid in retrieved if rid in relevant)
        return hits / len(retrieved)

    @staticmethod
    def _recall_at_k(retrieved: list[str], relevant: set[str]) -> float:
        """Recall@K: 相关文档被检索到的比例。"""
        if not relevant:
            return 1.0  # 无相关文档视为全召回
        hits = sum(1 for rid in retrieved if rid in relevant)
        return hits / len(relevant)

    @staticmethod
    def _reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
        """Reciprocal Rank: 第一个相关文档的排名倒数。"""
        for i, rid in enumerate(retrieved, start=1):
            if rid in relevant:
                return 1.0 / i
        return 0.0
