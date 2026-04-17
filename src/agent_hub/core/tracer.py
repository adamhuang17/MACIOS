"""全链路追踪。

基于 ContextVar 的轻量 Span 追踪，支持 OpenTelemetry 可选集成。
当 ``opentelemetry`` 未安装时自动降级为纯内存版本，仅通过
structlog 输出 JSON Lines 日志。

Example::

    from agent_hub.core.tracer import SpanContext, get_current_trace_id

    with SpanContext("pipeline.run", trace_id="abc-123") as span:
        span.add_event("step_done", {"key": "value"})
    print(span.to_dict())
"""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

# ── OpenTelemetry 可选加载 ────────────────────────────

_HAS_OTEL = False
_otel_tracer: Any = None

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )

    _HAS_OTEL = True
except ImportError:  # pragma: no cover — OTel 不是硬依赖
    pass

# ── 全局状态 ──────────────────────────────────────────

_current_trace_id: ContextVar[str] = ContextVar("trace_id", default="")


def init_tracer(service_name: str = "agent-hub") -> Any:
    """初始化全局追踪器。

    优先使用 OTLP exporter（Jaeger/Zipkin），不可用时降级为
    ConsoleSpanExporter；OTel SDK 未安装时仅使用 structlog。

    Args:
        service_name: 服务名称，用于 OTel resource 标识。

    Returns:
        OpenTelemetry ``Tracer`` 实例（无 OTel 时返回 ``None``）。
    """
    global _otel_tracer  # noqa: PLW0603

    if _HAS_OTEL:
        import os

        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        # 优先 OTLP exporter（仅在非测试 / 未禁用时启用）
        otlp_ok = False
        if not os.environ.get("OTEL_SDK_DISABLED"):
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
                logger.info("tracer_initialized", backend="otlp", service=service_name)
                otlp_ok = True
            except ImportError:
                pass

        if not otlp_ok:
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
            logger.info("tracer_initialized", backend="console", service=service_name)

        otel_trace.set_tracer_provider(provider)
        _otel_tracer = otel_trace.get_tracer(service_name)
        return _otel_tracer

    logger.info("tracer_initialized", backend="structlog_only", service=service_name)
    return None


def get_tracer() -> Any:
    """获取全局 OTel Tracer（未初始化时自动调用 ``init_tracer``）。"""
    if _otel_tracer is None and _HAS_OTEL:
        init_tracer()
    return _otel_tracer


def get_current_trace_id() -> str:
    """获取当前上下文的 ``trace_id``。"""
    return _current_trace_id.get()


# ── SpanContext — 轻量 Span 上下文管理器 ──────────────


class SpanContext:
    """简化的 Span 上下文管理器。

    在 ``with`` 块内自动完成：

    1. 设置当前 ``trace_id`` 到 ContextVar
    2. 记录 ``span_start`` / ``span_end`` structlog 日志
    3. 计算持续时间（毫秒）
    4. 捕获异常状态但 **不吞异常**

    若 OpenTelemetry 可用，还会同步创建 OTel Span。

    Args:
        name: Span 名称（如 ``"pipeline.run"``）。
        trace_id: 链路追踪 ID；为 ``None`` 时自动生成 UUID。

    Example::

        with SpanContext("agent.llm", trace_id="t-001") as span:
            span.add_event("prompt_built")
            # ... 业务逻辑
        print(span.duration_ms)
    """

    def __init__(self, name: str, trace_id: Optional[str] = None) -> None:
        self.name = name
        self.trace_id = trace_id or str(uuid.uuid4())
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.events: list[dict[str, Any]] = []
        self._otel_span: Any = None

    @property
    def duration_ms(self) -> int:
        """Span 持续时间（毫秒）。"""
        if self.end_time <= 0:
            return int((time.monotonic() - self.start_time) * 1000)
        return int((self.end_time - self.start_time) * 1000)

    def __enter__(self) -> SpanContext:
        self.start_time = time.monotonic()
        _current_trace_id.set(self.trace_id)
        logger.debug("span_start", span_name=self.name, trace_id=self.trace_id)

        # OTel span（可选）
        if _HAS_OTEL and _otel_tracer is not None:
            self._otel_span = _otel_tracer.start_span(self.name)

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        self.end_time = time.monotonic()
        status = "ok" if exc_type is None else "error"

        logger.info(
            "span_end",
            span_name=self.name,
            trace_id=self.trace_id,
            duration_ms=self.duration_ms,
            status=status,
            error=str(exc_val) if exc_val else None,
        )

        if self._otel_span is not None:
            if exc_val:
                self._otel_span.set_status(
                    otel_trace.StatusCode.ERROR,  # type: ignore[name-defined]
                    str(exc_val),
                )
                self._otel_span.record_exception(exc_val)
            self._otel_span.end()

        return False  # 不吞异常

    # ── 事件 & 序列化 ────────────────────────────────

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """在 Span 内添加事件。

        Args:
            name: 事件名称。
            attributes: 事件属性键值对。
        """
        event = {
            "name": name,
            "attributes": attributes or {},
            "timestamp": time.time(),
        }
        self.events.append(event)

        if self._otel_span is not None:
            self._otel_span.add_event(name, attributes=attributes or {})

    def to_dict(self) -> dict[str, Any]:
        """导出为字典（JSON Lines 兼容）。

        Returns:
            包含 trace_id、时间戳、持续时间和事件的字典。
        """
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "start_ms": int(self.start_time * 1000),
            "end_ms": int(self.end_time * 1000),
            "duration_ms": self.duration_ms,
            "events": self.events,
        }
