#!/usr/bin/env python3
"""三级缓存命中率与延迟基准测试。

用法::

    python scripts/bench_cache.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent_hub.core.cache import CacheLayer, CacheStats


# ── 测试 query 池 ──────────────────────────────────────

_QUERY_POOL = [
    "什么是机器学习？",
    "Python 的列表推导式怎么用？",
    "解释一下 Docker 的工作原理",
    "数据库索引是如何工作的？",
    "什么是 RESTful API？",
    "Git rebase 和 merge 的区别",
    "如何优化 SQL 查询性能？",
    "什么是微服务架构？",
    "Redis 的持久化机制是什么？",
    "解释一下 Transformer 架构",
    "HTTP 和 HTTPS 有什么区别？",
    "什么是 CI/CD？",
    "Kubernetes 的核心概念有哪些？",
    "如何设计一个消息队列系统？",
    "什么是 CAP 定理？",
    "解释一下 OAuth 2.0 的工作流程",
    "什么是事件驱动架构？",
    "如何实现分布式事务？",
    "什么是 GraphQL？",
    "解释一下 WebSocket 的工作原理",
]


async def bench_cache_l1() -> None:
    """L1 精确匹配延迟基准。"""
    mock_redis = AsyncMock()
    cache = CacheLayer(ttl=300)
    cache._redis = mock_redis

    # 预写入
    mock_redis.setex = AsyncMock(return_value=True)
    user_id = "bench_user"
    for q in _QUERY_POOL:
        await cache.set(user_id, q, f"回答: {q}")

    # 测试命中延迟
    mock_redis.get = AsyncMock(return_value="缓存的回答")

    latencies: list[float] = []
    for _ in range(500):
        q = random.choice(_QUERY_POOL)
        start = time.perf_counter()
        await cache.get(user_id, q)
        elapsed_us = (time.perf_counter() - start) * 1_000_000
        latencies.append(elapsed_us)

    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    avg = statistics.mean(latencies)

    print("\n" + "=" * 60)
    print(f"L1 精确匹配延迟（mock Redis, 500 次）")
    print("=" * 60)
    print(f"  平均延迟: {avg:.1f} μs ({avg/1000:.3f} ms)")
    print(f"  P50:      {p50:.1f} μs ({p50/1000:.3f} ms)")
    print(f"  P95:      {p95:.1f} μs ({p95/1000:.3f} ms)")
    print(f"  P99:      {p99:.1f} μs ({p99/1000:.3f} ms)")


async def bench_cache_l2() -> None:
    """L2 语义匹配延迟基准。"""
    import numpy as np

    # 模拟 embedder
    mock_embedder = MagicMock()
    dim = 8
    # 为每个 query 生成固定 embedding
    embeddings = {q: np.random.randn(dim).tolist() for q in _QUERY_POOL}
    mock_embedder.embed_query = lambda q: embeddings.get(q, np.random.randn(dim).tolist())

    mock_redis = AsyncMock()
    cache = CacheLayer(embedder=mock_embedder, ttl=300, semantic_threshold=0.95)
    cache._redis = mock_redis

    # 预写入 L2 缓存
    user_id = "bench_user"
    for q in _QUERY_POOL:
        emb = embeddings[q]
        entry_key = f"cache:l2:{user_id}:{hashlib.sha256(f'{user_id}:{q.strip().lower()}'.encode()).hexdigest()[:16]}"
        await mock_redis.sadd(f"cache:l2_index:{user_id}", entry_key)
        mock_redis.get.side_effect = None

    # 测试 L2 查找延迟（模拟 L1 miss, L2 entry 存在）
    call_count = 0

    async def mock_get(key):
        nonlocal call_count
        call_count += 1
        if key.startswith("cache:l1:"):
            return None  # L1 miss
        # L2 entry
        q_idx = call_count % len(_QUERY_POOL)
        q = _QUERY_POOL[q_idx]
        return json.dumps({
            "embedding": embeddings[q],
            "response": f"回答: {q}",
            "query": q,
        })

    mock_redis.get = mock_get
    mock_redis.smembers = AsyncMock(
        return_value={f"cache:l2:{user_id}:{hashlib.sha256(f'{user_id}:{q.strip().lower()}'.encode()).hexdigest()[:16]}" for q in _QUERY_POOL}
    )

    latencies: list[float] = []
    for i in range(200):
        q = _QUERY_POOL[i % len(_QUERY_POOL)]
        start = time.perf_counter()
        await cache.get(user_id, q)
        elapsed_us = (time.perf_counter() - start) * 1_000_000
        latencies.append(elapsed_us)

    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    avg = statistics.mean(latencies)

    print("\n" + "=" * 60)
    print(f"L2 语义匹配延迟（mock Redis + mock Embedder, 200 次）")
    print("=" * 60)
    print(f"  平均延迟: {avg:.1f} μs ({avg/1000:.3f} ms)")
    print(f"  P50:      {p50:.1f} μs ({p50/1000:.3f} ms)")
    print(f"  P95:      {p95:.1f} μs ({p95/1000:.3f} ms)")
    print(f"  P99:      {p99:.1f} μs ({p99/1000:.3f} ms)")


def bench_cache_stats() -> None:
    """模拟缓存命中率统计。"""
    stats = CacheStats(l1_hits=0, l2_hits=0, misses=0)

    # 模拟请求分布: 20% L1 命中, 15% L2 命中, 65% miss
    random.seed(42)
    n_requests = 1000
    for _ in range(n_requests):
        r = random.random()
        if r < 0.20:
            stats.l1_hits += 1
        elif r < 0.35:
            stats.l2_hits += 1
        else:
            stats.misses += 1

    print("\n" + "=" * 60)
    print(f"缓存命中率模拟（{n_requests} 次请求）")
    print("=" * 60)
    print(f"  L1 命中: {stats.l1_hits} ({stats.l1_hits/n_requests:.1%})")
    print(f"  L2 命中: {stats.l2_hits} ({stats.l2_hits/n_requests:.1%})")
    print(f"  未命中:  {stats.misses} ({stats.misses/n_requests:.1%})")
    print(f"  总命中率: {stats.hit_rate:.1%}")
    print(f"  ⚠ 注意: 这是模拟数据，实际命中率需在真实环境下测量")


async def main() -> None:
    print("=" * 60)
    print("三级缓存基准测试")
    print("=" * 60)
    await bench_cache_l1()
    await bench_cache_l2()
    bench_cache_stats()


if __name__ == "__main__":
    asyncio.run(main())
