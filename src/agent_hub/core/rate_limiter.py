"""LLM 并发限流器。

基于 asyncio.Semaphore 控制 LLM 调用并发数，
支持 asyncio.wait_for 超时降级。
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class RateLimiter:
    """异步并发限流器。

    Args:
        max_concurrent: 最大并发请求数。
        default_timeout: 默认超时时间（秒），0 表示不限时。
    """

    def __init__(
        self,
        max_concurrent: int = 5,
        default_timeout: float = 30.0,
    ) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._default_timeout = default_timeout
        self._max_concurrent = max_concurrent
        # 统计
        self._total_calls: int = 0
        self._timeouts: int = 0
        self._rejected: int = 0

    @property
    def total_calls(self) -> int:
        return self._total_calls

    @property
    def timeouts(self) -> int:
        return self._timeouts

    async def call(
        self,
        coro: Awaitable[T],
        timeout: float | None = None,
    ) -> T:
        """在限流和超时保护下执行异步协程。

        Args:
            coro: 要执行的协程。
            timeout: 超时秒数（None 使用默认值，0 不限时）。

        Returns:
            协程返回值。

        Raises:
            asyncio.TimeoutError: 超时。
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        self._total_calls += 1

        async with self._semaphore:
            try:
                if effective_timeout > 0:
                    return await asyncio.wait_for(coro, timeout=effective_timeout)
                return await coro
            except asyncio.TimeoutError:
                self._timeouts += 1
                logger.warning(
                    "rate_limiter_timeout",
                    timeout=effective_timeout,
                    total_timeouts=self._timeouts,
                )
                raise

    async def call_with_fallback(
        self,
        coro: Awaitable[T],
        fallback: T,
        timeout: float | None = None,
    ) -> T:
        """在限流下执行协程，超时或异常时返回兜底值。

        Args:
            coro: 要执行的协程。
            fallback: 兜底返回值。
            timeout: 超时秒数。

        Returns:
            协程返回值，或 fallback。
        """
        try:
            return await self.call(coro, timeout)
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("rate_limiter_fallback", error=str(exc))
            return fallback
