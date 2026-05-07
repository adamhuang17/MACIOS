"""任务进度与活动心跳事件。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

from agent_hub.pilot.domain.enums import ActorType, EventLevel, EventType
from agent_hub.pilot.domain.events import ExecutionEvent
from agent_hub.pilot.domain.models import PlanStep, Task
from agent_hub.pilot.services.event_publisher import PilotEventPublisher


class TaskProgressReporter:
    """向统一事件流写入 task.progress 与周期性 heartbeat。"""

    def __init__(
        self,
        publisher: PilotEventPublisher,
        *,
        interval_seconds: float = 5.0,
    ) -> None:
        self._publisher = publisher
        self._interval_seconds = max(1.0, float(interval_seconds))

    async def emit(
        self,
        task: Task,
        message: str,
        *,
        phase: str,
        plan_id: str | None = None,
        step_id: str | None = None,
        level: EventLevel = EventLevel.INFO,
        payload: dict[str, Any] | None = None,
    ) -> ExecutionEvent:
        event_payload = {
            "phase": phase,
            "kind": "milestone",
            **(payload or {}),
        }
        return await self._publisher.record(ExecutionEvent(
            workspace_id=task.workspace_id,
            trace_id=task.trace_id,
            task_id=task.task_id,
            plan_id=plan_id or task.plan_id,
            step_id=step_id,
            type=EventType.TASK_PROGRESS,
            level=level,
            actor_type=ActorType.AGENT,
            actor_id="agent-hub",
            message=message,
            payload=event_payload,
        ))

    @asynccontextmanager
    async def heartbeat(
        self,
        task: Task,
        message: str,
        *,
        phase: str,
        plan_id: str | None = None,
        step: PlanStep | None = None,
        payload: dict[str, Any] | None = None,
    ) -> AsyncIterator[None]:
        base_payload = dict(payload or {})
        step_id = step.step_id if step is not None else None
        await self.emit(
            task,
            message,
            phase=phase,
            plan_id=plan_id or (step.plan_id if step is not None else None),
            step_id=step_id,
            payload={"kind": "milestone", **base_payload},
        )

        stop = asyncio.Event()
        started = time.monotonic()
        interval = self._interval_seconds

        async def _loop() -> None:
            heartbeat_count = 0
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=interval)
                except TimeoutError:
                    heartbeat_count += 1
                    await self.emit(
                        task,
                        f"Agent 仍在工作，最近活动 {int(interval)} 秒前",
                        phase=phase,
                        plan_id=plan_id or (
                            step.plan_id if step is not None else None
                        ),
                        step_id=step_id,
                        payload={
                            "kind": "heartbeat",
                            "heartbeat_count": heartbeat_count,
                            "interval_seconds": interval,
                            "elapsed_seconds": int(time.monotonic() - started),
                            **base_payload,
                        },
                    )
                else:
                    break

        heartbeat_task = asyncio.create_task(_loop())
        try:
            yield
        finally:
            stop.set()
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task


__all__ = ["TaskProgressReporter"]