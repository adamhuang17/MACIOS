"""PilotCommandService：M2 集中处理审批决议、拒绝终止与恢复任务。

route 层只调用本服务；审批后是否触发执行、拒绝后 task 如何收尾、
retry/recover 如何创建子任务，全部在这里集中。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

import structlog

from agent_hub.pilot.domain.enums import (
    ApprovalAction,
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
_DELETE_BLOCKED_STATUSES = {
    TaskStatus.PLANNING,
    TaskStatus.APPROVED,
    TaskStatus.RUNNING,
}
_CANCEL_ON_DELETE_STATUSES = {
    TaskStatus.CREATED,
    TaskStatus.AWAITING_APPROVAL,
    TaskStatus.BLOCKED,
    TaskStatus.RETRYABLE_FAILED,
}
_PAUSABLE_STATUSES = {
    TaskStatus.RUNNING,
}


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
        if approval.approvers and actor_id not in approval.approvers:
            msg = (
                f"actor_id={actor_id} 不在审批人名单内，"
                f"不能决议 approval {approval_id}"
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
                run_after_approval=run_after_approval,
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

    async def recover_failed_step(
        self,
        task_id: str,
        step_id: str,
        *,
        requester_id: str,
    ) -> dict[str, object]:
        """Resume an existing failed task from one failed step."""
        run_result = await self._orchestrator.recover_failed_step(
            task_id,
            step_id,
        )
        return {
            "task_id": run_result.task.task_id,
            "task_status": run_result.final_status.value,
            "run_result": _summarize_run(run_result),
            "recovered_step_id": step_id,
            "requester_id": requester_id,
        }

    # ── pause / delete tasks ────────────────────────

    async def pause_task(self, task_id: str, *, actor_id: str) -> dict[str, object]:
        """Request a safe pause after the currently running step completes."""
        task = await self._repo.get_task(task_id)
        if task is None:
            msg = f"unknown task: {task_id}"
            raise LookupError(msg)
        if task.metadata.get("deleted_at"):
            msg = f"task {task_id} has been deleted"
            raise CommandError(msg)
        if task.metadata.get("pause_requested_at"):
            return {
                "task_id": task_id,
                "task_status": task.status.value,
                "pause_requested": True,
            }
        if task.status not in _PAUSABLE_STATUSES:
            msg = (
                f"task {task_id} current status={task.status.value}, "
                "only running tasks can be paused"
            )
            raise CommandError(msg)

        now = datetime.now(UTC)
        metadata = dict(task.metadata)
        metadata.update({
            "pause_requested": True,
            "pause_requested_at": now.isoformat(),
            "pause_requested_by": actor_id,
        })
        updated = task.model_copy(update={
            "metadata": metadata,
            "updated_at": now,
            "version": task.version + 1,
        })
        await self._repo.save(updated, expected_version=task.version)
        await self._publisher.record(ExecutionEvent(
            workspace_id=updated.workspace_id,
            trace_id=updated.trace_id,
            task_id=updated.task_id,
            type=EventType.TASK_PROGRESS,
            level=EventLevel.INFO,
            message="task pause requested",
            payload={
                "pause_requested": True,
                "pause_requested_by": actor_id,
                "phase": "pause_requested",
            },
            actor_id=actor_id,
        ))
        return {
            "task_id": task_id,
            "task_status": updated.status.value,
            "pause_requested": True,
        }

    async def delete_task(self, task_id: str, *, actor_id: str) -> dict[str, object]:
        """Soft-delete a task from dashboard views while keeping audit snapshots."""
        task = await self._repo.get_task(task_id)
        if task is None:
            msg = f"unknown task: {task_id}"
            raise LookupError(msg)
        if task.metadata.get("deleted_at"):
            return {
                "task_id": task_id,
                "task_status": task.status.value,
                "deleted": True,
            }
        if task.status in _DELETE_BLOCKED_STATUSES:
            msg = (
                f"task {task_id} 当前 status={task.status.value}，"
                "正在执行或即将执行，不能删除"
            )
            raise CommandError(msg)

        now = datetime.now(UTC)
        base_task = task
        if task.status in _CANCEL_ON_DELETE_STATUSES:
            base_task = transition(task, TaskAction.CANCEL, patch={
                "error": task.error or "deleted by user",
            })

        metadata = dict(base_task.metadata)
        metadata.update({
            "deleted_at": now.isoformat(),
            "deleted_by": actor_id,
        })
        deleted_task = base_task.model_copy(update={
            "metadata": metadata,
            "updated_at": now,
            "version": base_task.version if base_task is not task else task.version + 1,
        })
        await self._repo.save(deleted_task, expected_version=task.version)

        await self._cancel_requested_approvals(deleted_task)
        await self._clear_active_task(deleted_task)
        await self._publisher.record(ExecutionEvent(
            workspace_id=deleted_task.workspace_id,
            trace_id=deleted_task.trace_id,
            task_id=deleted_task.task_id,
            type=EventType.TASK_STATUS_CHANGED,
            level=EventLevel.INFO,
            message="task deleted",
            payload={
                "deleted": True,
                "deleted_by": actor_id,
                "status": deleted_task.status.value,
            },
            actor_id=actor_id,
        ))
        return {
            "task_id": task_id,
            "task_status": deleted_task.status.value,
            "deleted": True,
        }

    async def _cancel_requested_approvals(self, task: Task) -> None:
        approvals = await self._repo.list_approvals_for_task(
            task.task_id,
            workspace_id=task.workspace_id,
        )
        for approval in approvals:
            if approval.status != ApprovalStatus.REQUESTED:
                continue
            cancelled = transition(approval, ApprovalAction.CANCEL)
            await self._repo.save(cancelled, expected_version=approval.version)

    async def _clear_active_task(self, task: Task) -> None:
        workspace = await self._repo.get_workspace(task.workspace_id)
        if workspace is None or workspace.active_task_id != task.task_id:
            return
        updated = workspace.model_copy(update={
            "active_task_id": None,
            "version": workspace.version + 1,
            "updated_at": datetime.now(UTC),
        })
        await self._repo.save(updated, expected_version=workspace.version)

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
        run_after_approval: bool,
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

        result_payload: dict[str, object] = {
            "approval_id": new_approval.approval_id,
            "approval_status": new_approval.status.value,
            "step_id": new_step.step_id,
            "step_status": new_step.status.value,
            "task_id": task.task_id,
        }

        if decision == "approve":
            if run_after_approval:
                run_result = await self._orchestrator.resume_after_step_approval(
                    task.task_id,
                )
                result_payload.update(
                    task_status=run_result.final_status.value,
                    run_result=_summarize_run(run_result),
                    note="step approved，已在原 task 上续跑",
                )
            else:
                latest_task = await self._repo.get_task(task.task_id) or task
                result_payload.update(
                    task_status=latest_task.status.value,
                    note="step approved；run_after_approval=false，等待手动 resume",
                )
            return result_payload

        # decision == "reject"：把当前 task 明确转为 failed
        latest_task = await self._repo.get_task(task.task_id) or task
        try:
            failed = transition(latest_task, TaskAction.FAIL, patch={
                "error": "step rejected by approver",
            })
        except IllegalTransition:
            failed = latest_task
        else:
            await self._repo.save(failed)
            await self._publisher.record(ExecutionEvent(
                workspace_id=failed.workspace_id,
                trace_id=failed.trace_id,
                task_id=failed.task_id,
                plan_id=new_step.plan_id,
                step_id=new_step.step_id,
                approval_id=new_approval.approval_id,
                type=EventType.TASK_FAILED,
                level=EventLevel.WARN,
                message="task failed: step rejected",
                payload={"status": failed.status.value},
                actor_id=actor_id,
            ))
        result_payload.update(
            task_status=failed.status.value,
            note="step rejected，task 已终止",
        )
        return result_payload

    # ── 手动 resume ───────────────────────────────

    async def resume_task(self, task_id: str) -> dict[str, object]:
        """在已 BLOCKED 的 task 上手动续跑（兼容历史 approved-but-not-resumed 数据）。"""
        task = await self._repo.get_task(task_id)
        if task is None:
            msg = f"unknown task: {task_id}"
            raise LookupError(msg)
        if task.status != TaskStatus.BLOCKED:
            msg = (
                f"task {task_id} 当前 status={task.status.value}，无需 resume"
            )
            raise CommandError(msg)
        run_result = await self._orchestrator.resume_after_step_approval(task_id)
        return {
            "task_id": task_id,
            "task_status": run_result.final_status.value,
            "run_result": _summarize_run(run_result),
        }


def _summarize_run(result: RunResult) -> dict[str, object]:
    return {
        "task_id": result.task.task_id,
        "final_status": result.final_status.value,
        "step_count": len(result.executions),
        "artifact_count": len(result.artifacts),
    }


__all__ = ["CommandError", "Decision", "PilotCommandService"]
