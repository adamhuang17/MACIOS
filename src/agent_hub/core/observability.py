"""OTel 可观测性初始化。

集中配置 TracerProvider + OTLP exporter + Metrics。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from agent_hub.config.settings import Settings

logger = structlog.get_logger(__name__)

_metrics: dict[str, Any] = {}


def init_observability(settings: Settings | None = None) -> dict[str, Any]:
    """初始化 OTel TracerProvider + OTLP exporter + Metrics。

    Args:
        settings: 全局配置（可选，用于读取 OTLP endpoint 等）。

    Returns:
        包含 tracer / meter 等对象的字典。
    """
    result: dict[str, Any] = {}

    try:
        from opentelemetry import metrics as otel_metrics
        from opentelemetry import trace as otel_trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": "agent-hub"})

        # Tracer
        provider = TracerProvider(resource=resource)
        otlp_exporter = OTLPSpanExporter()  # 默认 localhost:4317
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        otel_trace.set_tracer_provider(provider)
        result["tracer"] = otel_trace.get_tracer("agent-hub")

        # Metrics
        meter_provider = MeterProvider(resource=resource)
        otel_metrics.set_meter_provider(meter_provider)
        meter = otel_metrics.get_meter("agent-hub")
        result["meter"] = meter

        # 常用指标
        result["request_counter"] = meter.create_counter(
            "agent_hub.requests",
            description="Total requests processed",
        )
        result["latency_histogram"] = meter.create_histogram(
            "agent_hub.latency_ms",
            description="Request latency in milliseconds",
            unit="ms",
        )
        result["cache_hit_counter"] = meter.create_counter(
            "agent_hub.cache_hits",
            description="Cache hit count by level",
        )

        global _metrics  # noqa: PLW0603
        _metrics = result

        logger.info("observability_initialized", backend="otlp")

    except ImportError:
        logger.info("observability_fallback", reason="opentelemetry not installed")
    except Exception as exc:
        logger.warning("observability_init_failed", error=str(exc))

    return result


def get_metrics() -> dict[str, Any]:
    """获取全局 metrics 字典。"""
    return _metrics


def record_request(intent: str, duration_ms: int, status: str) -> None:
    """记录一次请求的指标。"""
    if not _metrics:
        return
    try:
        if "request_counter" in _metrics:
            _metrics["request_counter"].add(1, {"intent": intent, "status": status})
        if "latency_histogram" in _metrics:
            _metrics["latency_histogram"].record(duration_ms, {"intent": intent})
    except Exception:
        pass


def record_cache_hit(level: str) -> None:
    """记录缓存命中。"""
    if not _metrics:
        return
    try:
        if "cache_hit_counter" in _metrics:
            _metrics["cache_hit_counter"].add(1, {"level": level})
    except Exception:
        pass
