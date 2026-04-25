"""EventBus 订阅与 fan-out 测试。"""

from __future__ import annotations

import asyncio

import pytest

from agent_hub.pilot import EventBus, EventType, ExecutionEvent
from agent_hub.pilot.events.bus import SubscriberOverflow


def _evt(ws: str, msg: str = "") -> ExecutionEvent:
    return ExecutionEvent(
        workspace_id=ws,
        type=EventType.TASK_CREATED,
        message=msg,
        sequence=1,
    )


@pytest.mark.asyncio
async def test_publish_to_single_subscriber() -> None:
    bus = EventBus()
    sub = await bus.subscribe("ws_a")
    await bus.publish(_evt("ws_a", "hi"))
    e = await asyncio.wait_for(sub.get(), timeout=1.0)
    assert e.message == "hi"


@pytest.mark.asyncio
async def test_workspace_isolation() -> None:
    bus = EventBus()
    sub_a = await bus.subscribe("ws_a")
    sub_b = await bus.subscribe("ws_b")
    await bus.publish(_evt("ws_a", "for-a"))
    await bus.publish(_evt("ws_b", "for-b"))
    a = await asyncio.wait_for(sub_a.get(), timeout=1.0)
    b = await asyncio.wait_for(sub_b.get(), timeout=1.0)
    assert a.message == "for-a"
    assert b.message == "for-b"


@pytest.mark.asyncio
async def test_multiple_subscribers_receive_each_event() -> None:
    bus = EventBus()
    s1 = await bus.subscribe("ws")
    s2 = await bus.subscribe("ws")
    await bus.publish(_evt("ws", "x"))
    e1 = await asyncio.wait_for(s1.get(), timeout=1.0)
    e2 = await asyncio.wait_for(s2.get(), timeout=1.0)
    assert e1.message == e2.message == "x"


@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery() -> None:
    bus = EventBus()
    sub = await bus.subscribe("ws")
    await bus.unsubscribe(sub)
    await bus.publish(_evt("ws", "ignored"))
    # unsubscribe 后队列被关闭，async iter 应停止
    received: list[ExecutionEvent] = []

    async def collect() -> None:
        async for e in sub:
            received.append(e)

    await asyncio.wait_for(collect(), timeout=0.5)
    assert received == []


@pytest.mark.asyncio
async def test_overflow_closes_subscriber() -> None:
    bus = EventBus(default_queue_size=2)
    sub = await bus.subscribe("ws")
    for i in range(5):
        await bus.publish(_evt("ws", str(i)))
    # 排干前两条
    await sub.get()
    await sub.get()
    with pytest.raises(SubscriberOverflow):
        await sub.get()


@pytest.mark.asyncio
async def test_close_clears_all_subscribers() -> None:
    bus = EventBus()
    await bus.subscribe("ws_a")
    await bus.subscribe("ws_b")
    await bus.close()
    # 关闭后再 publish 不报错
    await bus.publish(_evt("ws_a"))
