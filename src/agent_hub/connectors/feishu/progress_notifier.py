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
        min_interval_seconds: float = 30.0,
    ) -> None:
        self._repo = repository
        self._client = client
        self._min_interval_seconds = max(1.0, float(min_interval_seconds))
        self._last_sent_at: dict[str, float] = {}
        self._pending_approvals_by_task: dict[str, set[str]] = {}
        self._send_chains: dict[str, asyncio.Task[None]] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def handle_event(self, event: ExecutionEvent) -> None:
        if event.type is EventType.APPROVAL_REQUESTED:
            self._mark_approval_pending(event)
            return
        if event.type in {
            EventType.APPROVAL_DECIDED,
            EventType.APPROVAL_EXPIRED,
            EventType.APPROVAL_SUPERSEDED,
        }:
            self._clear_approval_pending(event)
            return
        if event.type is not EventType.TASK_PROGRESS:
            return
        if event.payload.get("notify_feishu") is False:
            return
        task_id = event.task_id or event.event_id
        if self._pending_approvals_by_task.get(task_id):
            return
        kind = str(event.payload.get("kind") or "milestone")
        now = time.monotonic()
        last_sent = self._last_sent_at.get(task_id, 0.0)
        if kind == "heartbeat" and now - last_sent < self._min_interval_seconds:
            return
        if not _should_notify_progress(event):
            return
        self._last_sent_at[task_id] = now

        previous = self._send_chains.get(task_id)
        task = asyncio.create_task(self._send_after(previous, event, task_id))
        self._send_chains[task_id] = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(
            lambda done, chained_task_id=task_id: self._clear_send_chain(
                chained_task_id, done,
            ),
        )

    def _mark_approval_pending(self, event: ExecutionEvent) -> None:
        if not event.task_id:
            return
        approval_id = event.approval_id or event.event_id
        self._pending_approvals_by_task.setdefault(event.task_id, set()).add(
            approval_id,
        )

    def _clear_approval_pending(self, event: ExecutionEvent) -> None:
        if not event.task_id:
            return
        if not event.approval_id:
            self._pending_approvals_by_task.pop(event.task_id, None)
            return
        pending = self._pending_approvals_by_task.get(event.task_id)
        if pending is None:
            return
        pending.discard(event.approval_id)
        if not pending:
            self._pending_approvals_by_task.pop(event.task_id, None)

    def _clear_send_chain(
        self,
        task_id: str,
        task: asyncio.Task[None],
    ) -> None:
        if self._send_chains.get(task_id) is task:
            self._send_chains.pop(task_id, None)

    async def _send_after(
        self,
        previous: asyncio.Task[None] | None,
        event: ExecutionEvent,
        task_id: str,
    ) -> None:
        if previous is not None:
            with suppress(Exception):
                await previous
        if self._pending_approvals_by_task.get(task_id):
            return
        await self._send(event)

    async def aclose(self) -> None:
        if not self._tasks:
            return
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            with suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
        self._send_chains.clear()

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
        elapsed = int(event.payload.get("elapsed_seconds") or 0)
        duration = _format_duration(elapsed)
        return f"Agent 仍在工作，已持续 {duration}\n任务：{title}\n阶段：{phase}"
    return f"{event.message}\n任务：{title}\n阶段：{phase}"


def _should_notify_progress(event: ExecutionEvent) -> bool:
    phase = str(event.payload.get("phase") or "")
    message = event.message
    if phase == "step_prepare":
        return False
    if message == "Agent 开始执行任务步骤":
        return False
    return not message.startswith("Agent 正在处理：")


def _format_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{max(seconds, 1)} 秒"
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        if remaining_seconds:
            return f"{minutes} 分 {remaining_seconds} 秒"
        return f"{minutes} 分钟"
    hours, remaining_minutes = divmod(minutes, 60)
    if remaining_minutes:
        return f"{hours} 小时 {remaining_minutes} 分"
    return f"{hours} 小时"


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