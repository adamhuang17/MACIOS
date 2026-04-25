"""SSE 流式输出 + 流式 Pipeline + 并发限流测试。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_hub.api.streaming import sse_generator
from agent_hub.core.rate_limiter import RateLimiter

# ═══════════════════════════════════════════════════════
# RateLimiter 测试
# ═══════════════════════════════════════════════════════


class TestRateLimiter:
    """并发限流器测试。"""

    @pytest.mark.asyncio
    async def test_basic_call(self) -> None:
        limiter = RateLimiter(max_concurrent=5, default_timeout=10)

        async def coro() -> int:
            return 42

        result = await limiter.call(coro())
        assert result == 42
        assert limiter.total_calls == 1

    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        limiter = RateLimiter(max_concurrent=5, default_timeout=0.1)

        async def slow_coro() -> str:
            await asyncio.sleep(10)
            return "done"

        with pytest.raises(asyncio.TimeoutError):
            await limiter.call(slow_coro())

        assert limiter.timeouts == 1

    @pytest.mark.asyncio
    async def test_fallback_on_timeout(self) -> None:
        limiter = RateLimiter(max_concurrent=5, default_timeout=0.1)

        async def slow_coro() -> str:
            await asyncio.sleep(10)
            return "done"

        result = await limiter.call_with_fallback(
            slow_coro(), fallback="兜底值", timeout=0.1,
        )
        assert result == "兜底值"

    @pytest.mark.asyncio
    async def test_concurrency_limit(self) -> None:
        limiter = RateLimiter(max_concurrent=2, default_timeout=5)
        concurrent_count = 0
        max_concurrent = 0

        async def tracked_coro() -> str:
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return "ok"

        tasks = [limiter.call(tracked_coro()) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        assert all(r == "ok" for r in results)
        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_no_timeout(self) -> None:
        limiter = RateLimiter(max_concurrent=5, default_timeout=0)

        async def coro() -> str:
            return "no timeout"

        result = await limiter.call(coro())
        assert result == "no timeout"


# ═══════════════════════════════════════════════════════
# SSE Generator 测试
# ═══════════════════════════════════════════════════════


class TestSSEGenerator:
    """SSE 事件生成器测试。"""

    @pytest.mark.asyncio
    async def test_non_streaming_fallback(self) -> None:
        """无 run_stream 时降级到标准 run。"""
        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock()
        mock_output = MagicMock()
        mock_output.response = "标准回答"
        mock_pipeline.run.return_value = mock_output
        # 没有 run_stream 属性
        del mock_pipeline.run_stream

        mock_input = MagicMock()
        mock_input.trace_id = "t1"

        events = []
        async for event in sse_generator(mock_pipeline, mock_input):
            events.append(json.loads(event))

        assert events[0]["type"] == "status"
        assert any(e["type"] == "token" and e["content"] == "标准回答" for e in events)
        assert events[-1]["type"] == "done"

    @pytest.mark.asyncio
    async def test_streaming_mode(self) -> None:
        """有 run_stream 时逐 chunk 推送。"""
        mock_pipeline = MagicMock()

        async def mock_stream(_input: object) -> AsyncIterator[dict[str, str]]:
            yield {"type": "token", "content": "Hello "}
            yield {"type": "token", "content": "World"}

        mock_pipeline.run_stream = mock_stream

        mock_input = MagicMock()
        mock_input.trace_id = "t1"

        events = []
        async for event in sse_generator(mock_pipeline, mock_input):
            events.append(json.loads(event))

        assert events[0]["type"] == "status"
        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert "Hello " in tokens
        assert "World" in tokens
        assert events[-1]["type"] == "done"

    @pytest.mark.asyncio
    async def test_error_handling(self) -> None:
        """异常时推送 error 事件。"""
        mock_pipeline = MagicMock()

        async def mock_stream(_input: object) -> AsyncIterator[dict[str, str]]:
            raise RuntimeError("LLM 炸了")
            yield {"type": "token", "content": ""}

        mock_pipeline.run_stream = mock_stream

        mock_input = MagicMock()
        mock_input.trace_id = "t1"

        events = []
        async for event in sse_generator(mock_pipeline, mock_input):
            events.append(json.loads(event))

        assert any(e["type"] == "error" for e in events)
        assert events[-1]["type"] == "done"
