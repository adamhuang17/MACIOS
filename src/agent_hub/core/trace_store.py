"""轻量 Trace 存储。

提供按 ``trace_id`` 聚合 SpanContext 的内存存储，用于 ``/trace/{trace_id}``
接口的查询。生产环境可替换为 OTLP / Jaeger 后端，但骨架阶段优先保持
零外部依赖、零配置可用。

设计原则：
- 进程内存储 + 环形上限，避免无限增长导致 OOM；
- 写入容错（异常静默），不影响主流程；
- 读取按时间戳排序，便于前端绘制时间轴。
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Protocol

import structlog

logger = structlog.get_logger(__name__)


class TraceStoreProtocol(Protocol):
    """Trace 存储接口（便于将来替换为 SQLite/OTLP 后端）。"""

    def record_span(self, span: dict[str, object]) -> None: ...

    def get_trace(self, trace_id: str) -> list[dict[str, object]] | None: ...


class InMemoryTraceStore:
    """进程内 Trace 存储，使用 OrderedDict 实现 LRU 上限。

    Args:
        max_traces: 最多保留的 trace 数量；超出按 FIFO 淘汰最旧的。
        max_spans_per_trace: 单个 trace 内最多保留的 span 数量。
    """

    def __init__(
        self,
        max_traces: int = 1000,
        max_spans_per_trace: int = 200,
    ) -> None:
        self._max_traces = max_traces
        self._max_spans_per_trace = max_spans_per_trace
        self._traces: OrderedDict[str, list[dict[str, object]]] = OrderedDict()
        self._lock = threading.Lock()

    def record_span(self, span: dict[str, object]) -> None:
        """记录单个 Span。

        重复 ``trace_id`` 会追加到同一列表；超出上限自动截断。
        """
        trace_id = str(span.get("trace_id") or "")
        if not trace_id:
            return
        with self._lock:
            spans = self._traces.get(trace_id)
            if spans is None:
                spans = []
                self._traces[trace_id] = spans
                # LRU 淘汰
                while len(self._traces) > self._max_traces:
                    self._traces.popitem(last=False)
            else:
                # 移到末尾保持 LRU 语义
                self._traces.move_to_end(trace_id)
            spans.append(span)
            if len(spans) > self._max_spans_per_trace:
                del spans[: len(spans) - self._max_spans_per_trace]

    def get_trace(self, trace_id: str) -> list[dict[str, object]] | None:
        """按 ``trace_id`` 取出所有 Span，按 ``start_ms`` 升序排序。"""
        with self._lock:
            spans = self._traces.get(trace_id)
            if spans is None:
                return None
            # 浅拷贝避免外部并发修改
            return sorted(
                list(spans),
                key=lambda s: int(s.get("start_ms", 0) or 0),
            )

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()


# ── 全局单例 ───────────────────────────────────────────

_global_store: TraceStoreProtocol | None = None


def set_trace_store(store: TraceStoreProtocol | None) -> None:
    """注册全局 TraceStore；传入 ``None`` 表示禁用。"""
    global _global_store  # noqa: PLW0603
    _global_store = store


def get_trace_store() -> TraceStoreProtocol | None:
    """获取全局 TraceStore。"""
    return _global_store


__all__ = [
    "InMemoryTraceStore",
    "TraceStoreProtocol",
    "get_trace_store",
    "set_trace_store",
]
