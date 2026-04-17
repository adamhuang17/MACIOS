"""RRF (Reciprocal Rank Fusion) 融合排序。

将 Dense 和 Sparse 检索双路结果合并，按 RRF 分数重新排序。

公式：``RRF(d) = Σ_{r ∈ R} 1 / (k + rank_r(d))``

其中 ``k`` 为平滑常数（默认 60），``rank_r(d)`` 为文档 ``d`` 在结果集 ``r``
中的排名（从 1 开始）。
"""

from __future__ import annotations

from agent_hub.rag.vector_store import SearchResult


def rrf_merge(
    dense_results: list[SearchResult],
    sparse_results: list[SearchResult],
    k: int = 60,
    top_k: int | None = None,
) -> list[SearchResult]:
    """RRF 融合 Dense + Sparse 检索结果。

    Args:
        dense_results: Dense（向量）检索结果列表，已按相似度降序排列。
        sparse_results: Sparse（BM25）检索结果列表，已按 BM25 分数降序排列。
        k: 平滑常数，默认 60。
        top_k: 返回的最大结果数；``None`` 表示返回全部。

    Returns:
        按 RRF 分数降序排列的合并结果。
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, SearchResult] = {}

    # Dense 贡献
    for rank, result in enumerate(dense_results, start=1):
        doc_key = _doc_key(result)
        rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + 1.0 / (k + rank)
        if doc_key not in content_map:
            content_map[doc_key] = result

    # Sparse 贡献
    for rank, result in enumerate(sparse_results, start=1):
        doc_key = _doc_key(result)
        rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + 1.0 / (k + rank)
        if doc_key not in content_map:
            content_map[doc_key] = result

    # 按 RRF 分数降序排列
    sorted_keys = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)

    merged: list[SearchResult] = []
    for doc_key in sorted_keys:
        original = content_map[doc_key]
        merged.append(SearchResult(
            content=original.content,
            score=rrf_scores[doc_key],
            metadata=original.metadata,
            chunk_id=original.chunk_id,
        ))

    if top_k is not None:
        return merged[:top_k]
    return merged


def _doc_key(result: SearchResult) -> str:
    """生成文档唯一标识。

    优先用 chunk_id，fallback 到 content hash。
    """
    if result.chunk_id is not None:
        return f"chunk_{result.chunk_id}"
    return f"hash_{hash(result.content)}"
