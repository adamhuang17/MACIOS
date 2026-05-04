"""Pilot SSE：先按 ``EventStore`` 回放 backfill，再订阅 ``EventBus`` 实时事件。

设计要点：
- 先 ``subscribe`` 再 ``list_events``，避免回放与 live 之间的窗口丢失。
- 用 sequence 集合 + last_seq 双重去重，防止 backfill 与 live 重叠。
- 可按 ``task_id`` 过滤；workspace 维度的事件流仍由调用方决定。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from agent_hub.pilot.events.bus import EventBus
    from agent_hub.pilot.events.store import EventStore


logger = structlog.get_logger(__name__)


def _serialize(event_type: str, payload: dict[str, object]) -> str:
    return json.dumps({"type": event_type, **payload}, ensure_ascii=False)


async def pilot_event_stream(
    *,
    workspace_id: str,
    task_id: str | None,
    since_sequence: int,
    store: EventStore,
    bus: EventBus,
    backfill_limit: int = 500,
    heartbeat_interval: float = 15.0,
) -> AsyncGenerator[str, None]:
    """生成 SSE data 行（不含 ``data: `` 前缀）。"""
    sub = await bus.subscribe(workspace_id)
    last_seq = since_sequence
    try:
        events = await store.list_events(
            workspace_id, since_sequence=since_sequence, limit=backfill_limit,
        )
        emitted: set[int] = set()
        for ev in events:
            if task_id is not None and ev.task_id != task_id:
                continue
            yield _serialize("event", {
                "sequence": ev.sequence,
                "event": ev.model_dump(mode="json"),
            })
            emitted.add(ev.sequence)
            last_seq = max(last_seq, ev.sequence)

        yield _serialize("backfill_done", {
            "latest_sequence": last_seq,
            "count": len(emitted),
        })

        while True:
            try:
                ev = await asyncio.wait_for(sub.get(), timeout=heartbeat_interval)
            except TimeoutError:
                yield _serialize("heartbeat", {"latest_sequence": last_seq})
                continue
            except StopAsyncIteration:
                break

            if ev.sequence <= last_seq or ev.sequence in emitted:
                continue
            if task_id is not None and ev.task_id != task_id:
                continue
            yield _serialize("event", {
                "sequence": ev.sequence,
                "event": ev.model_dump(mode="json"),
            })
            emitted.add(ev.sequence)
            last_seq = ev.sequence
        # 正常退出（订阅被关闭）才发送 done；GeneratorExit/异常路径下
        # 不能再 yield，否则会触发 "async generator ignored GeneratorExit"。
        yield _serialize("done", {"latest_sequence": last_seq})
    except GeneratorExit:
        # 客户端主动断开 / 调用方 aclose()：直接静默退出。
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("pilot_sse.error", error=str(exc))
        try:
            yield _serialize("error", {"message": str(exc)})
        except GeneratorExit:
            raise
    finally:
        await bus.unsubscribe(sub)


__all__ = ["pilot_event_stream"]
