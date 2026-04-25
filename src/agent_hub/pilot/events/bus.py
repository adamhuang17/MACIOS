"""EventBus：进程内事件 fan-out。

约定：
- 仅做进程内转发，不持久化。
- M2 SSE 流先用 EventStore.list_events 补齐 since_sequence，再订阅 EventBus。
- 队列 bounded，慢订阅者满队列时关闭并抛出明确异常。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import structlog

from agent_hub.pilot.domain.events import ExecutionEvent

logger = structlog.get_logger(__name__)


class SubscriberOverflow(RuntimeError):  # noqa: N818
    """订阅者队列溢出，已被强制关闭。"""


class EventSubscription:
    """单个订阅者。

    使用 ``async for event in subscription`` 消费；
    通过 ``await subscription.aclose()`` 主动取消订阅。
    """

    def __init__(self, workspace_id: str, queue_size: int) -> None:
        self.workspace_id = workspace_id
        self._queue: asyncio.Queue[ExecutionEvent] = asyncio.Queue(maxsize=queue_size)
        self._closed = False
        self._overflow = False

    def _put_nowait(self, event: ExecutionEvent) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._overflow = True
            self._closed = True
            logger.warning(
                "pilot.event_bus.subscriber_overflow",
                workspace_id=self.workspace_id,
            )

    async def get(self) -> ExecutionEvent:
        if not self._queue.empty():
            return self._queue.get_nowait()
        if self._overflow:
            raise SubscriberOverflow(
                f"订阅者 (ws={self.workspace_id}) 队列溢出已关闭",
            )
        if self._closed:
            raise StopAsyncIteration
        return await self._queue.get()

    def __aiter__(self) -> AsyncIterator[ExecutionEvent]:
        return self

    async def __anext__(self) -> ExecutionEvent:
        if not self._queue.empty():
            return self._queue.get_nowait()
        if self._overflow:
            raise SubscriberOverflow(
                f"订阅者 (ws={self.workspace_id}) 队列溢出已关闭",
            )
        if self._closed:
            raise StopAsyncIteration
        try:
            return await self._queue.get()
        except asyncio.CancelledError:
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        self._closed = True


class EventBus:
    """按 workspace_id 维度做 fan-out 的进程内事件总线。

    Args:
        default_queue_size: 每个订阅者的队列容量。
    """

    def __init__(self, default_queue_size: int = 256) -> None:
        self._default_queue_size = default_queue_size
        self._subs: dict[str, list[EventSubscription]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(
        self,
        workspace_id: str,
        *,
        queue_size: int | None = None,
    ) -> EventSubscription:
        sub = EventSubscription(
            workspace_id=workspace_id,
            queue_size=queue_size or self._default_queue_size,
        )
        async with self._lock:
            self._subs.setdefault(workspace_id, []).append(sub)
        return sub

    async def unsubscribe(self, subscription: EventSubscription) -> None:
        await subscription.aclose()
        async with self._lock:
            subs = self._subs.get(subscription.workspace_id, [])
            self._subs[subscription.workspace_id] = [
                s for s in subs if s is not subscription
            ]

    async def publish(self, event: ExecutionEvent) -> None:
        async with self._lock:
            subs = list(self._subs.get(event.workspace_id, []))

        dead: list[EventSubscription] = []
        for sub in subs:
            sub._put_nowait(event)
            if sub._closed and sub._overflow:
                dead.append(sub)

        if dead:
            async with self._lock:
                remaining = self._subs.get(event.workspace_id, [])
                self._subs[event.workspace_id] = [
                    s for s in remaining if s not in dead
                ]

    async def close(self) -> None:
        async with self._lock:
            for subs in self._subs.values():
                for s in subs:
                    await s.aclose()
            self._subs.clear()


__all__ = ["EventBus", "EventSubscription", "SubscriberOverflow"]
