"""PilotCommandService：M2 集中处理审批决议、拒绝终止与恢复任务。

route 层只调用本服务；审批后是否触发执行、拒绝后 task 如何收尾、
retry/recover 如何创建子任务，全部在这里集中。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import structlog

from agent_hub.pilot.domain.enums import (
    ApprovalStatus,
    ApprovalTargetType,
    EventLevel,
    EventType,
    TaskAction,
    TaskStatus,
)
from agent_hub.pilot.domain.errors import IllegalTransition
from agent_hub.pilot.domain.events import ExecutionEvent
from agent_hub.pilot.domain.models import Approval
from agent_hub.pilot.domain.state import transition
from agent_hub.pilot.services.dto import RunResult, TaskHandle, TaskRequest

if TYPE_CHECKING:
    from agent_hub.pilot.services.approval import ApprovalService
    from agent_hub.pilot.services.event_publisher import PilotEventPublisher
    from agent_hub.pilot.services.orchestrator import TaskOrchestrator
    from agent_hub.pilot.services.repository import PilotRepository

logger = structlog.get_logger(__name__)


Decision = Literal["approve", "reject"]


class CommandError(RuntimeError):
    """命令层校验错误（用于映射到 4xx）。"""


class PilotCommandService:
    def __init__(
        self,
        repository: PilotRepository,
        approvals: ApprovalService,
        orchestrator: TaskOrchestrator,
        publisher: PilotEventPublisher,
    ) -> None:
        self._repo = repository
        self._approvals = approvals
        self._orchestrator = orchestrator
        self._publisher = publisher

    # ── 审批决议 ───────────────────────────────────

    async def decide_approval(
        self,
        approval_id: str,
        *,
        decision: Decision,
        actor_id: str,
        comment: str | None = None,
        run_after_approval: bool = True,
    ) -> dict[str, object]:
        approval = await self._repo.get_approval(approval_id)
        if approval is None:
            msg = f"unknown approval: {approval_id}"
            raise LookupError(msg)
        if approval.status != ApprovalStatus.REQUESTED:
            msg = (
                f"approval {approval_id} 已处于终态 {approval.status.value}，"
                "不能再次决议"
            )
            raise CommandError(msg)

        if approval.target_type == ApprovalTargetType.PLAN:
            return await self._decide_plan(
                approval,
                decision=decision,
                actor_id=actor_id,
                comment=comment,
                run_after_approval=run_after_approval,
            )
        if approval.target_type == ApprovalTargetType.STEP:
            return await self._decide_step(
                approval,
                decision=decision,
                actor_id=actor_id,
                comment=comment,
            )
        msg = f"M2 暂不支持 target_type={approval.target_type.value}"
        raise CommandError(msg)

    # ── retry / recover ───────────────────────────

    async def create_recovery_task(
        self,
        original_task_id: str,
        *,
        requester_id: str,
        from_step_id: str | None = None,
        auto_approve: bool = False,
    ) -> TaskHandle:
        original = await self._repo.get_task(original_task_id)
        if original is None:
            msg = f"unknown task: {original_task_id}"
            raise LookupError(msg)
        if original.status not in (
            TaskStatus.FAILED,
            TaskStatus.RETRYABLE_FAILED,
            TaskStatus.BLOCKED,
            TaskStatus.CANCELLED,
        ):
            msg = (
                f"task {original_task_id} 处于 {original.status.value}，"
                "不需要恢复"
            )
            raise CommandError(msg)

        ws = await self._repo.get_workspace(original.workspace_id)
        ws_title = ws.title if ws is not None else original.origin_text[:32]

        request = TaskRequest(
            source_channel="recovery",
            raw_text=original.origin_text,
            requester_id=requester_id,
            title=f"Recovery of {ws_title}",
            workspace_id=original.workspace_id,
            tenant_key=ws.tenant_key if ws is not None else None,
            metadata={
                "parent_task_id": original.task_id,
                "parent_step_id": from_step_id,
                "recovery_reason": original.error or "manual",
            },
            auto_approve=auto_approve,
        )
        return await self._orchestrator.submit(request)

    # ── 内部 ───────────────────────────────────────

    async def _decide_plan(
        self,
        approval: Approval,
        *,
        decision: Decision,
        actor_id: str,
        comment: str | None,
        run_after_approval: bool,
    ) -> dict[str, object]:
        plan = await self._repo.get_plan(approval.target_id)
        if plan is None:
            msg = f"plan 已被清理: {approval.target_id}"
            raise CommandError(msg)
        task = await self._repo.get_task(plan.task_id)
        if task is None:
            msg = f"task 已被清理: {plan.task_id}"
            raise CommandError(msg)

        new_approval, new_plan = await self._approvals.decide_plan(
            approval, plan, task,
            decision=decision,
            approver_id=actor_id,
            comment=comment,
        )

        result_payload: dict[str, object] = {
            "approval_id": new_approval.approval_id,
            "approval_status": new_approval.status.value,
            "plan_id": new_plan.plan_id,
            "plan_status": new_plan.status.value,
            "task_id": task.task_id,
        }

        if decision == "approve":
            if run_after_approval:
                run_result = await self._orchestrator.run_after_approval(
                    task.task_id,
                )
                result_payload.update(
                    task_status=run_result.final_status.value,
                    run_result=_summarize_run(run_result),
                )
            else:
                result_payload["task_status"] = task.status.value
            return result_payload

        # decision == "reject"
        latest_task = await self._repo.get_task(task.task_id) or task
        try:
            failed = transition(latest_task, TaskAction.FAIL, patch={
                "error": "plan rejected by approver",
            })
        except IllegalTransition:
            # task 已经被其它路径终结（cancelled/failed），只记录事件即可
            failed = latest_task
        else:
            await self._repo.save(failed)
            await self._publisher.record(ExecutionEvent(
                workspace_id=failed.workspace_id,
                trace_id=failed.trace_id,
                task_id=failed.task_id,
                plan_id=plan.plan_id,
                approval_id=new_approval.approval_id,
                type=EventType.TASK_FAILED,
                level=EventLevel.WARN,
                message="task failed: plan rejected",
                payload={"status": failed.status.value},
                actor_id=actor_id,
            ))

        result_payload["task_status"] = failed.status.value
        return result_payload

    async def _decide_step(
        self,
        approval: Approval,
        *,
        decision: Decision,
        actor_id: str,
        comment: str | None,
    ) -> dict[str, object]:
        step = await self._repo.get_step(approval.target_id)
        if step is None:
            msg = f"step 已被清理: {approval.target_id}"
            raise CommandError(msg)
        task = await self._repo.get_task(step.task_id)
        if task is None:
            msg = f"task 已被清理: {step.task_id}"
            raise CommandError(msg)

        new_approval, new_step = await self._approvals.decide_step(
            approval, step, task,
            decision=decision,
            approver_id=actor_id,
            comment=comment,
        )
        return {
            "approval_id": new_approval.approval_id,
            "approval_status": new_approval.status.value,
            "step_id": new_step.step_id,
            "step_status": new_step.status.value,
            "task_id": task.task_id,
            "note": (
                "step approved；M2 不在原 task 上自动续跑，"
                "请通过 recovery task 重启执行链路"
                if decision == "approve" else
                "step rejected"
            ),
        }


def _summarize_run(result: RunResult) -> dict[str, object]:
    return {
        "task_id": result.task.task_id,
        "final_status": result.final_status.value,
        "step_count": len(result.executions),
        "artifact_count": len(result.artifacts),
    }


__all__ = ["CommandError", "Decision", "PilotCommandService"]
