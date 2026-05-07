"""PilotEventPublisher：先持久化事件，再向 EventBus fan-out。

任何服务层的事件发布都必须经过本类，保证发布出去的事件已经被
EventStore 分配真实的 ``sequence``，便于 SSE 续传与审计。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from agent_hub.pilot.domain.events import ExecutionEvent
    from agent_hub.pilot.events.bus import EventBus
    from agent_hub.pilot.events.store import EventStore

EventHandler = Callable[["ExecutionEvent"], Awaitable[None] | None]

logger = structlog.get_logger(__name__)


class PilotEventPublisher:
    def __init__(
        self,
        store: EventStore,
        bus: EventBus | None = None,
    ) -> None:
        self._store = store
        self._bus = bus
        self._handlers: list[EventHandler] = []

    def add_handler(self, handler: EventHandler) -> None:
        self._handlers.append(handler)

    async def record(self, event: ExecutionEvent) -> ExecutionEvent:
        stored = await self._store.append_event(event)
        if self._bus is not None:
            await self._bus.publish(stored)
        for handler in list(self._handlers):
            try:
                result = handler(stored)
                if result is not None:
                    await result
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "pilot.event_handler_failed",
                    event_id=stored.event_id,
                    event_type=stored.type.value,
                    error=str(exc),
                )
        return stored


__all__ = ["PilotEventPublisher"]
