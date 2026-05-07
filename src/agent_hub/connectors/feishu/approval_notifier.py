"""飞书审批卡片通知器。"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from agent_hub.connectors.feishu.client import FeishuClientProtocol
from agent_hub.pilot.domain.enums import ApprovalTargetType
from agent_hub.pilot.domain.errors import ConcurrencyConflict
from agent_hub.pilot.domain.models import Approval, Task
from agent_hub.pilot.services.repository import PilotRepository
from agent_hub.pilot.skills.feishu_card import build_approval_card

logger = structlog.get_logger(__name__)


class FeishuApprovalNotifier:
    """把 ``Approval`` 渲染为飞书 interactive 卡片并发送。"""

    def __init__(
        self,
        repository: PilotRepository,
        client: FeishuClientProtocol,
    ) -> None:
        self._repo = repository
        self._client = client

    async def notify_requested(self, approval: Approval) -> None:
        if approval.channel_message_id:
            return

        workspace = await self._repo.get_workspace(approval.workspace_id)
        if (
            workspace is None
            or workspace.source_channel != "feishu"
            or not workspace.feishu_chat_id
        ):
            return

        task = await self._repo.get_task(approval.task_id)
        if task is None:
            logger.warning(
                "feishu.approval_notify_missing_task",
                approval_id=approval.approval_id,
                task_id=approval.task_id,
            )
            return

        title, summary, detail = await self._build_card_content(approval, task)
        card = build_approval_card(
            approval_id=approval.approval_id,
            title=title,
            summary=summary,
            detail=detail,
        )
        sent = await self._client.send_interactive_card(
            receive_id=workspace.feishu_chat_id,
            receive_id_type="chat_id",
            card=card,
        )

        updated = approval.model_copy(update={
            "version": approval.version + 1,
            "updated_at": datetime.now(UTC),
            "channel_message_id": sent.message_id,
        })
        try:
            await self._repo.save(updated, expected_version=approval.version)
        except ConcurrencyConflict:
            logger.warning(
                "feishu.approval_notify_save_conflict",
                approval_id=approval.approval_id,
                message_id=sent.message_id,
            )
        else:
            logger.info(
                "feishu.approval_card_sent",
                approval_id=approval.approval_id,
                chat_id=workspace.feishu_chat_id,
                message_id=sent.message_id,
            )

    async def _build_card_content(
        self,
        approval: Approval,
        task: Task,
    ) -> tuple[str, str, str]:
        task_title = task.origin_text or task.task_id

        if approval.target_type is ApprovalTargetType.STEP:
            step = await self._repo.get_step(approval.target_id)
            step_title = step.title if step is not None else approval.target_id
            summary = (
                f"**任务**: {task_title}\n\n"
                f"**步骤**: {step_title}\n\n"
                "执行进入人工审批，请确认是否继续。"
            )
            detail = (
                f"approval_id: {approval.approval_id}\n"
                f"task_id: {task.task_id}\n"
                f"step_id: {approval.target_id}"
            )
            return f"步骤审批：{step_title}"[:60], summary, detail

        plan = await self._repo.get_plan(approval.target_id)
        plan_goal = plan.goal if plan is not None else task_title
        summary = (
            f"**任务**: {task_title}\n\n"
            f"**计划目标**: {plan_goal}\n\n"
            "计划已生成，等待人工确认后继续执行。"
        )
        detail = (
            f"approval_id: {approval.approval_id}\n"
            f"task_id: {task.task_id}\n"
            f"plan_id: {approval.target_id}"
        )
        return "计划审批"[:60], summary, detail


__all__ = ["FeishuApprovalNotifier"]