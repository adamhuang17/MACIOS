"""三级缓存层测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_hub.core.cache import CacheHit, CacheLayer, CacheStats


class TestCacheStats:
    """缓存统计测试。"""

    def test_empty_stats(self) -> None:
        stats = CacheStats()
        assert stats.total == 0
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self) -> None:
        stats = CacheStats(l1_hits=3, l2_hits=2, misses=5)
        assert stats.total == 10
        assert stats.hit_rate == 0.5

    def test_all_hits(self) -> None:
        stats = CacheStats(l1_hits=10, l2_hits=0, misses=0)
        assert stats.hit_rate == 1.0


class TestCacheLayerL1:
    """L1 精确缓存测试。"""

    @pytest.mark.asyncio
    async def test_l1_miss(self) -> None:
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        mock_redis.smembers.return_value = set()

        cache = CacheLayer()
        cache._redis = mock_redis

        result = await cache.get("u1", "测试查询")
        assert result is None
        assert cache.stats.misses == 1

    @pytest.mark.asyncio
    async def test_l1_hit(self) -> None:
        mock_redis = AsyncMock()
        mock_redis.get.return_value = "缓存的回答"

        cache = CacheLayer()
        cache._redis = mock_redis

        result = await cache.get("u1", "测试查询")
        assert result is not None
        assert result.response == "缓存的回答"
        assert result.level == "L1"
        assert cache.stats.l1_hits == 1

    @pytest.mark.asyncio
    async def test_set_and_get(self) -> None:
        mock_redis = AsyncMock()
        # set 调用
        mock_redis.setex.return_value = True
        # get 后返回缓存值
        mock_redis.get.return_value = "回答"

        cache = CacheLayer(ttl=60)
        cache._redis = mock_redis

        await cache.set("u1", "query", "回答")
        mock_redis.setex.assert_called()

    @pytest.mark.asyncio
    async def test_invalidate(self) -> None:
        mock_redis = AsyncMock()
        cache = CacheLayer()
        cache._redis = mock_redis

        await cache.invalidate("u1", "query")
        mock_redis.delete.assert_called_once()


class TestCacheLayerL2:
    """L2 语义缓存测试。"""

    @pytest.mark.asyncio
    async def test_l2_hit(self) -> None:
        import json

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [1.0, 0.0, 0.0]

        mock_redis = AsyncMock()
        # L1 miss
        mock_redis.get.side_effect = [
            None,  # L1 miss
            json.dumps({  # L2 entry
                "embedding": [1.0, 0.0, 0.0],
                "response": "语义缓存命中",
                "query": "similar query",
            }),
        ]
        mock_redis.smembers.return_value = {"cache:l2:u1:abc123"}

        cache = CacheLayer(embedder=mock_embedder, semantic_threshold=0.9)
        cache._redis = mock_redis

        result = await cache.get("u1", "查询")
        assert result is not None
        assert result.level == "L2"
        assert result.response == "语义缓存命中"

    @pytest.mark.asyncio
    async def test_l2_below_threshold(self) -> None:
        import json

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [1.0, 0.0, 0.0]

        mock_redis = AsyncMock()
        # L1 miss, L2 entry with perpendicular vector
        mock_redis.get.side_effect = [
            None,
            json.dumps({
                "embedding": [0.0, 1.0, 0.0],
                "response": "不相关",
                "query": "different query",
            }),
        ]
        mock_redis.smembers.return_value = {"cache:l2:u1:xyz"}

        cache = CacheLayer(embedder=mock_embedder, semantic_threshold=0.9)
        cache._redis = mock_redis

        result = await cache.get("u1", "查询")
        assert result is None
        assert cache.stats.misses == 1


class TestCosineSimlarity:
    """余弦相似度工具函数测试。"""

    def test_identical_vectors(self) -> None:
        sim = CacheLayer._cosine_similarity([1, 0, 0], [1, 0, 0])
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self) -> None:
        sim = CacheLayer._cosine_similarity([1, 0, 0], [0, 1, 0])
        assert abs(sim) < 1e-6

    def test_opposite_vectors(self) -> None:
        sim = CacheLayer._cosine_similarity([1, 0], [-1, 0])
        assert abs(sim - (-1.0)) < 1e-6

    def test_zero_vector(self) -> None:
        sim = CacheLayer._cosine_similarity([0, 0], [1, 0])
        assert sim == 0.0

    def test_different_lengths(self) -> None:
        sim = CacheLayer._cosine_similarity([1, 0], [1, 0, 0])
        assert sim == 0.0
