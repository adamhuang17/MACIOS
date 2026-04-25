"""ExecutionEvent: append-only 审计事实。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from agent_hub.pilot.domain.enums import ActorType, EventLevel, EventType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ExecutionEvent(BaseModel):
    """Pilot 执行事件。

    ``sequence`` 由 EventStore 在同一个 ``workspace_id`` 内单调分配，
    调用方构造时不应自行赋值。``payload`` 必须可 JSON 序列化。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:16]}")
    workspace_id: str
    sequence: int = 0
    trace_id: Optional[str] = None
    task_id: Optional[str] = None
    plan_id: Optional[str] = None
    step_id: Optional[str] = None
    approval_id: Optional[str] = None
    artifact_id: Optional[str] = None
    type: EventType
    level: EventLevel = EventLevel.INFO
    actor_type: ActorType = ActorType.SYSTEM
    actor_id: Optional[str] = None
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow)
