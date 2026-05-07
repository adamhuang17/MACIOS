"""把 Pilot 进度事件推送回飞书聊天。"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import suppress

import structlog

from agent_hub.connectors.feishu.client import FeishuClientProtocol
from agent_hub.pilot.domain.enums import EventType
from agent_hub.pilot.domain.events import ExecutionEvent
from agent_hub.pilot.services.repository import PilotRepository

logger = structlog.get_logger(__name__)


class FeishuProgressNotifier:
    """监听 ``task.progress`` 事件，并向来源飞书会话发送进度文本。"""

    def __init__(
        self,
        repository: PilotRepository,
        client: FeishuClientProtocol,
        *,
        min_interval_seconds: float = 5.0,
    ) -> None:
        self._repo = repository
        self._client = client
        self._min_interval_seconds = max(1.0, float(min_interval_seconds))
        self._last_sent_at: dict[str, float] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def handle_event(self, event: ExecutionEvent) -> None:
        if event.type is not EventType.TASK_PROGRESS:
            return
        if event.payload.get("notify_feishu") is False:
            return
        task_id = event.task_id or event.event_id
        kind = str(event.payload.get("kind") or "milestone")
        now = time.monotonic()
        last_sent = self._last_sent_at.get(task_id, 0.0)
        if kind == "heartbeat" and now - last_sent < self._min_interval_seconds:
            return
        self._last_sent_at[task_id] = now

        task = asyncio.create_task(self._send(event))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def aclose(self) -> None:
        if not self._tasks:
            return
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            with suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

    async def _send(self, event: ExecutionEvent) -> None:
        workspace = await self._repo.get_workspace(event.workspace_id)
        if (
            workspace is None
            or workspace.source_channel != "feishu"
            or not workspace.feishu_chat_id
        ):
            return

        task = await self._repo.get_task(event.task_id or "")
        title = task.origin_text[:48] if task is not None else event.task_id or "任务"
        text = _format_progress_text(event, title)
        try:
            await self._client.send_message(
                receive_id=workspace.feishu_chat_id,
                receive_id_type="chat_id",
                msg_type="text",
                content=json.dumps({"text": text}, ensure_ascii=False),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "feishu.progress_notify_failed",
                workspace_id=event.workspace_id,
                task_id=event.task_id,
                error=str(exc),
            )


def _format_progress_text(event: ExecutionEvent, title: str) -> str:
    kind = str(event.payload.get("kind") or "milestone")
    phase = _phase_label(str(event.payload.get("phase") or ""))
    if kind == "heartbeat":
        interval = int(event.payload.get("interval_seconds") or 5)
        return f"Agent 仍在工作，最近活动 {interval} 秒前\n任务：{title}\n阶段：{phase}"
    return f"{event.message}\n任务：{title}\n阶段：{phase}"


def _phase_label(phase: str) -> str:
    return {
        "accepted": "已接收",
        "planning": "任务编排",
        "awaiting_approval": "等待审批",
        "execution": "执行中",
        "step_prepare": "准备步骤",
        "step_running": "步骤执行",
        "step_completed": "步骤完成",
        "step_failed": "步骤失败",
        "step_awaiting_approval": "步骤审批",
        "blocked": "等待处理",
        "failed": "失败",
        "completed": "完成",
    }.get(phase, phase or "进度更新")


__all__ = ["FeishuProgressNotifier"]