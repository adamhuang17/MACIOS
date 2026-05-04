"""PilotEventPublisher：先持久化事件，再向 EventBus fan-out。

任何服务层的事件发布都必须经过本类，保证发布出去的事件已经被
EventStore 分配真实的 ``sequence``，便于 SSE 续传与审计。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_hub.pilot.domain.events import ExecutionEvent
    from agent_hub.pilot.events.bus import EventBus
    from agent_hub.pilot.events.store import EventStore


class PilotEventPublisher:
    def __init__(
        self,
        store: EventStore,
        bus: EventBus | None = None,
    ) -> None:
        self._store = store
        self._bus = bus

    async def record(self, event: ExecutionEvent) -> ExecutionEvent:
        stored = await self._store.append_event(event)
        if self._bus is not None:
            await self._bus.publish(stored)
        return stored


__all__ = ["PilotEventPublisher"]
