"""跨层事件 envelope 契约。

定义统一的领域事件信封（DomainEvent），用于跨进程事件传递。

设计原则：
- DomainEvent 是 envelope，不是 Pilot 的 ExecutionEvent
- Pilot 的事件审计索引（workspace_id、stream_sequence、task_id 等）
  通过 stream_id / stream_sequence 字段保留，装进 envelope 而不替换它
- payload 由各业务层自行序列化，contracts 不约束其内容
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DomainEvent(BaseModel):
    """通用领域事件信封。

    任何业务层都可以把自己的事件装进这个 envelope 发布；
    消费方通过 event_type + aggregate_type 做路由，
    通过 stream_id + stream_sequence 做审计和幂等。
    """

    model_config = ConfigDict(extra="forbid")

    # ── 事件标识 ──
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    schema_version: int = 1

    # ── 聚合根标识 ──
    aggregate_type: str
    aggregate_id: str

    # ── 流式审计索引（Pilot 用 workspace_id + stream_sequence） ──
    workspace_id: str | None = None
    stream_id: str | None = None
    stream_sequence: int | None = None

    # ── 追踪关联 ──
    trace_id: str | None = None
    correlation_id: str | None = None
    causation_id: str | None = None

    # ── 内容 ──
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


__all__ = [
    "DomainEvent",
]
