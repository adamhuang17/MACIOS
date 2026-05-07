"""ApprovalService：集中审批状态推进。

所有审批状态变更必须通过本类，从而保证 ``Approval``、``Plan``、
``PlanStep`` 之间的相互联动（例如 Plan v2 supersede 旧 approval）
全部走 M0 ``transition`` 状态机。
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
    PlanAction,
    PlanStatus,
    PlanStepAction,
    PlanStepStatus,
)
from agent_hub.pilot.domain.events import ExecutionEvent
from agent_hub.pilot.domain.models import (
    Approval,
    ApprovalDecision,
    Plan,
    PlanStep,
    Task,
)
from agent_hub.pilot.domain.state import transition

if TYPE_CHECKING:
    from agent_hub.pilot.services.event_publisher import PilotEventPublisher
    from agent_hub.pilot.services.repository import PilotRepository

logger = structlog.get_logger(__name__)


Decision = Literal["approve", "reject"]


class ApprovalService:
    def __init__(
        self,
        repository: PilotRepository,
        publisher: PilotEventPublisher,
    ) -> None:
        self._repo = repository
        self._publisher = publisher

    # ── Plan 审批 ──────────────────────────────────

    async def request_plan_approval(
        self,
        plan: Plan,
        task: Task,
        *,
        requester_id: str,
        approvers: list[str] | None = None,
        reason: str | None = None,
    ) -> tuple[Approval, Plan]:
        approval = Approval(
            workspace_id=plan.workspace_id,
            task_id=plan.task_id,
            target_type=ApprovalTargetType.PLAN,
            target_id=plan.plan_id,
            requester_id=requester_id,
            approvers=list(approvers or []),
            reason=reason,
            status=ApprovalStatus.REQUESTED,
        )
        new_plan = transition(plan, PlanAction.REQUEST_APPROVAL)
        await self._repo.save(approval, expected_version=None)
        await self._repo.save(new_plan)
        await self._publisher.record(ExecutionEvent(
            workspace_id=plan.workspace_id,
            trace_id=task.trace_id,
            task_id=task.task_id,
            plan_id=plan.plan_id,
            approval_id=approval.approval_id,
            type=EventType.APPROVAL_REQUESTED,
            message="plan approval requested",
        ))
        return approval, new_plan

    async def auto_approve_plan(
        self,
        plan: Plan,
        task: Task,
        *,
        requester_id: str,
    ) -> tuple[Approval, Plan]:
        """演示 / 测试用：一次性走完 request → auto_approve → APPROVED。"""
        approval, plan = await self.request_plan_approval(
            plan, task, requester_id=requester_id,
        )
        return await self.decide_plan(
            approval, plan, task,
            decision="approve",
            approver_id="system",
            comment="auto-approved",
            auto=True,
        )

    async def decide_plan(
        self,
        approval: Approval,
        plan: Plan,
        task: Task,
        *,
        decision: Decision,
        approver_id: str,
        comment: str | None = None,
        auto: bool = False,
    ) -> tuple[Approval, Plan]:
        action = (
            ApprovalAction.AUTO_APPROVE
            if auto and decision == "approve"
            else ApprovalAction.APPROVE if decision == "approve"
            else ApprovalAction.REJECT
        )
        decided_at = datetime.now(UTC)
        new_approval = transition(approval, action, patch={
            "decided_at": decided_at,
            "decision_records": [
                *approval.decision_records,
                ApprovalDecision(
                    approver_id=approver_id,
                    decision=decision,
                    comment=comment,
                    decided_at=decided_at,
                ),
            ],
        })
        new_plan = transition(
            plan,
            PlanAction.APPROVE if decision == "approve" else PlanAction.REJECT,
        )
        await self._repo.save(new_approval)
        await self._repo.save(new_plan)
        await self._publisher.record(ExecutionEvent(
            workspace_id=plan.workspace_id,
            trace_id=task.trace_id,
            task_id=task.task_id,
            plan_id=plan.plan_id,
            approval_id=approval.approval_id,
            type=EventType.APPROVAL_DECIDED,
            level=EventLevel.INFO,
            message=f"plan approval decided: {decision}",
            payload={"decision": decision, "auto": auto},
        ))
        return new_approval, new_plan

    # ── Step 审批 ──────────────────────────────────

    async def request_step_approval(
        self,
        step: PlanStep,
        task: Task,
        *,
        requester_id: str,
        approvers: list[str] | None = None,
    ) -> tuple[Approval, PlanStep]:
        approval = Approval(
            workspace_id=step.workspace_id,
            task_id=step.task_id,
            target_type=ApprovalTargetType.STEP,
            target_id=step.step_id,
            requester_id=requester_id,
            approvers=list(approvers or []),
            status=ApprovalStatus.REQUESTED,
        )
        if step.status not in (
            PlanStepStatus.PENDING,
            PlanStepStatus.DRY_RUNNING,
        ):
            msg = (
                f"step {step.step_id} 不在可申请审批的状态: {step.status.value}"
            )
            raise ValueError(msg)
        new_step = transition(step, PlanStepAction.REQUEST_APPROVAL)
        await self._repo.save(approval, expected_version=None)
        await self._repo.save(new_step)
        await self._publisher.record(ExecutionEvent(
            workspace_id=step.workspace_id,
            trace_id=task.trace_id,
            task_id=step.task_id,
            plan_id=step.plan_id,
            step_id=step.step_id,
            approval_id=approval.approval_id,
            type=EventType.APPROVAL_REQUESTED,
            message="step approval requested",
        ))
        return approval, new_step

    async def decide_step(
        self,
        approval: Approval,
        step: PlanStep,
        task: Task,
        *,
        decision: Decision,
        approver_id: str,
        comment: str | None = None,
        auto: bool = False,
    ) -> tuple[Approval, PlanStep]:
        action = (
            ApprovalAction.AUTO_APPROVE
            if auto and decision == "approve"
            else ApprovalAction.APPROVE if decision == "approve"
            else ApprovalAction.REJECT
        )
        decided_at = datetime.now(UTC)
        new_approval = transition(approval, action, patch={
            "decided_at": decided_at,
            "decision_records": [
                *approval.decision_records,
                ApprovalDecision(
                    approver_id=approver_id,
                    decision=decision,
                    comment=comment,
                    decided_at=decided_at,
                ),
            ],
        })
        new_step = transition(
            step,
            PlanStepAction.APPROVE if decision == "approve" else PlanStepAction.FAIL,
            patch={"error": None if decision == "approve" else "approval rejected"},
        )
        await self._repo.save(new_approval)
        await self._repo.save(new_step)
        await self._publisher.record(ExecutionEvent(
            workspace_id=step.workspace_id,
            trace_id=task.trace_id,
            task_id=task.task_id,
            plan_id=step.plan_id,
            step_id=step.step_id,
            approval_id=approval.approval_id,
            type=EventType.APPROVAL_DECIDED,
            message=f"step approval decided: {decision}",
            payload={"decision": decision, "auto": auto},
        ))
        return new_approval, new_step

    # ── Plan v2 supersede ──────────────────────────

    async def supersede_for_new_plan(
        self,
        old_plan: Plan,
        task: Task,
    ) -> Plan:
        """Plan v2 出现时，把旧 plan 与其待决 approvals 全部 supersede。"""
        approvals = await self._repo.list_approvals_for_plan(
            old_plan.plan_id, workspace_id=old_plan.workspace_id,
        )
        for appr in approvals:
            if appr.status == ApprovalStatus.REQUESTED:
                superseded = transition(appr, ApprovalAction.SUPERSEDE)
                await self._repo.save(superseded)
                await self._publisher.record(ExecutionEvent(
                    workspace_id=appr.workspace_id,
                    trace_id=task.trace_id,
                    task_id=task.task_id,
                    plan_id=old_plan.plan_id,
                    approval_id=appr.approval_id,
                    type=EventType.APPROVAL_SUPERSEDED,
                    message="approval superseded by new plan",
                ))

        if old_plan.status != PlanStatus.SUPERSEDED:
            new_old = transition(old_plan, PlanAction.SUPERSEDE)
            await self._repo.save(new_old)
            return new_old
        return old_plan


__all__ = ["ApprovalService", "Decision"]
