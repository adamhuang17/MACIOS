"""三级缓存层。

L1: 精确匹配 — sha256(user_id + normalize(query)) → Redis string
L2: 语义匹配 — query embedding 余弦相似度 > threshold → Redis
L3: 穿透 — 返回 None
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from agent_hub.rag.embedder import Embedder

logger = structlog.get_logger(__name__)


@dataclass
class CacheHit:
    """缓存命中结果。"""

    response: str
    level: str  # "L1" | "L2"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStats:
    """缓存命中率统计。"""

    l1_hits: int = 0
    l2_hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.l1_hits + self.l2_hits + self.misses

    @property
    def hit_rate(self) -> float:
        return (self.l1_hits + self.l2_hits) / max(self.total, 1)


class CacheLayer:
    """三级缓存：精确匹配 → 语义匹配 → 穿透。

    Args:
        redis_url: Redis 连接地址。
        embedder: Embedder 实例（L2 语义缓存用）。
        ttl: L1 缓存 TTL（秒）。
        semantic_threshold: L2 语义缓存相似度阈值。
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        embedder: Embedder | None = None,
        ttl: int = 300,
        semantic_threshold: float = 0.95,
    ) -> None:
        self._ttl = ttl
        self._semantic_threshold = semantic_threshold
        self._embedder = embedder
        self._stats = CacheStats()

        # 懒加载 Redis 连接
        self._redis = None
        self._redis_url = redis_url

    async def _get_redis(self):
        """懒加载 Redis 异步连接。"""
        if self._redis is None:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self._redis_url, decode_responses=True,
            )
        return self._redis

    @property
    def stats(self) -> CacheStats:
        return self._stats

    # ── 查询 ─────────────────────────────────────────

    async def get(self, user_id: str, query: str) -> CacheHit | None:
        """三级缓存查询。

        Args:
            user_id: 用户 ID。
            query: 用户查询。

        Returns:
            缓存命中结果，或 None（穿透）。
        """
        # L1: 精确匹配
        l1_hit = await self._get_l1(user_id, query)
        if l1_hit is not None:
            self._stats.l1_hits += 1
            logger.debug("cache_l1_hit", user_id=user_id)
            return CacheHit(response=l1_hit, level="L1")

        # L2: 语义匹配
        if self._embedder is not None:
            l2_hit = await self._get_l2(user_id, query)
            if l2_hit is not None:
                self._stats.l2_hits += 1
                logger.debug("cache_l2_hit", user_id=user_id)
                return CacheHit(response=l2_hit, level="L2")

        # L3: 穿透
        self._stats.misses += 1
        return None

    # ── 写入 ─────────────────────────────────────────

    async def set(
        self,
        user_id: str,
        query: str,
        response: str,
        embedding: list[float] | None = None,
    ) -> None:
        """写入三级缓存。

        Args:
            user_id: 用户 ID。
            query: 用户查询。
            response: LLM 响应。
            embedding: 查询向量（可选，用于 L2 缓存）。
        """
        try:
            r = await self._get_redis()
            # L1: 精确缓存
            l1_key = self._l1_key(user_id, query)
            await r.setex(l1_key, self._ttl, response)

            # L2: 语义缓存（存储 embedding + response）
            if embedding is not None:
                l2_key = self._l2_entry_key(user_id, query)
                entry = json.dumps({
                    "embedding": embedding,
                    "response": response,
                    "query": query,
                })
                await r.setex(l2_key, self._ttl * 2, entry)

                # 维护 L2 索引集合
                index_key = self._l2_index_key(user_id)
                await r.sadd(index_key, l2_key)
                await r.expire(index_key, self._ttl * 3)

        except Exception as exc:
            logger.warning("cache_set_failed", error=str(exc))

    # ── 失效 ─────────────────────────────────────────

    async def invalidate(self, user_id: str, query: str) -> None:
        """使指定缓存失效。"""
        try:
            r = await self._get_redis()
            l1_key = self._l1_key(user_id, query)
            l2_key = self._l2_entry_key(user_id, query)
            await r.delete(l1_key, l2_key)
        except Exception as exc:
            logger.warning("cache_invalidate_failed", error=str(exc))

    # ── 关闭 ─────────────────────────────────────────

    async def close(self) -> None:
        """关闭 Redis 连接。"""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    # ── L1 实现 ──────────────────────────────────────

    async def _get_l1(self, user_id: str, query: str) -> str | None:
        """L1 精确匹配。"""
        try:
            r = await self._get_redis()
            key = self._l1_key(user_id, query)
            return await r.get(key)
        except Exception:
            return None

    # ── L2 实现 ──────────────────────────────────────

    async def _get_l2(self, user_id: str, query: str) -> str | None:
        """L2 语义匹配：遍历该用户的缓存 embedding，找余弦相似度 > threshold 的。"""
        try:
            r = await self._get_redis()
            index_key = self._l2_index_key(user_id)
            entry_keys = await r.smembers(index_key)

            if not entry_keys:
                return None

            query_emb = self._embedder.embed_query(query)

            best_score = 0.0
            best_response = None

            for ek in entry_keys:
                raw = await r.get(ek)
                if raw is None:
                    # 已过期，从索引中移除
                    await r.srem(index_key, ek)
                    continue
                entry = json.loads(raw)
                cached_emb = entry["embedding"]
                sim = self._cosine_similarity(query_emb, cached_emb)
                if sim > best_score:
                    best_score = sim
                    best_response = entry["response"]

            if best_score >= self._semantic_threshold and best_response is not None:
                return best_response

        except Exception as exc:
            logger.warning("cache_l2_failed", error=str(exc))
        return None

    # ── Key 构建 ─────────────────────────────────────

    @staticmethod
    def _l1_key(user_id: str, query: str) -> str:
        normalized = query.strip().lower()
        h = hashlib.sha256(f"{user_id}:{normalized}".encode()).hexdigest()
        return f"cache:l1:{h}"

    @staticmethod
    def _l2_entry_key(user_id: str, query: str) -> str:
        normalized = query.strip().lower()
        h = hashlib.sha256(f"{user_id}:{normalized}".encode()).hexdigest()[:16]
        return f"cache:l2:{user_id}:{h}"

    @staticmethod
    def _l2_index_key(user_id: str) -> str:
        return f"cache:l2_index:{user_id}"

    # ── 工具函数 ─────────────────────────────────────

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """计算余弦相似度。"""
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
