"""PilotQueryService：M2 Dashboard 与 API 共用的只读投影。

服务层只通过本类向 API 暴露聚合 read model；route 层禁止直接读取
``PilotRepository``/``EventStore`` 拼装业务真相。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from agent_hub.pilot.domain.enums import (
    ApprovalStatus,
    PlanStatus,
    PlanStepStatus,
    TaskStatus,
)
from agent_hub.pilot.domain.events import ExecutionEvent
from agent_hub.pilot.domain.models import (
    Approval,
    Artifact,
    Plan,
    PlanStep,
    Task,
    Workspace,
)

if TYPE_CHECKING:
    from agent_hub.pilot.events.store import EventStore
    from agent_hub.pilot.services.repository import PilotRepository


class TaskSummary(BaseModel):
    """Dashboard 列表行。"""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    task_id: str
    title: str
    status: TaskStatus
    plan_id: str | None
    plan_status: PlanStatus | None
    pending_approvals: int
    artifact_count: int
    latest_sequence: int
    requester_id: str
    updated_at: str


class WorkspaceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace: Workspace
    task_count: int
    active_task_id: str | None
    latest_sequence: int


class TaskDetail(BaseModel):
    """Dashboard 详情聚合。"""

    model_config = ConfigDict(extra="forbid")

    workspace: Workspace
    task: Task
    active_plan: Plan | None
    steps: list[PlanStep]
    approvals: list[Approval]
    artifacts: list[Artifact]
    recent_events: list[ExecutionEvent]
    latest_sequence: int
    available_actions: list[dict[str, Any]] = Field(default_factory=list)


class PilotQueryService:
    def __init__(
        self,
        repository: PilotRepository,
        event_store: EventStore,
    ) -> None:
        self._repo = repository
        self._events = event_store

    # ── workspaces ─────────────────────────────────

    async def list_workspaces(self) -> list[WorkspaceSummary]:
        out: list[WorkspaceSummary] = []
        for ws in await self._repo.list_workspaces():
            tasks = await self._repo.list_tasks(workspace_id=ws.workspace_id)
            latest_seq = await self._latest_sequence(ws.workspace_id)
            out.append(WorkspaceSummary(
                workspace=ws,
                task_count=len(tasks),
                active_task_id=ws.active_task_id,
                latest_sequence=latest_seq,
            ))
        return out

    # ── tasks ──────────────────────────────────────

    async def list_tasks(
        self,
        *,
        workspace_id: str | None = None,
        status: TaskStatus | None = None,
        limit: int = 50,
    ) -> list[TaskSummary]:
        tasks = await self._repo.list_tasks(workspace_id=workspace_id)
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        tasks.sort(key=lambda t: t.updated_at, reverse=True)
        tasks = tasks[:limit]

        # 按 workspace 维度缓存 latest sequence，避免重复扫表。
        seq_cache: dict[str, int] = {}
        out: list[TaskSummary] = []
        for t in tasks:
            seq = seq_cache.get(t.workspace_id)
            if seq is None:
                seq = await self._latest_sequence(t.workspace_id)
                seq_cache[t.workspace_id] = seq

            plan = await self._repo.get_plan(t.plan_id) if t.plan_id else None
            approvals = await self._repo.list_approvals_for_task(
                t.task_id, workspace_id=t.workspace_id,
            )
            pending = sum(
                1 for a in approvals if a.status == ApprovalStatus.REQUESTED
            )
            artifacts = await self._repo.list_task_artifacts(
                t.task_id, workspace_id=t.workspace_id,
            )
            out.append(TaskSummary(
                workspace_id=t.workspace_id,
                task_id=t.task_id,
                title=t.origin_text[:80] or "Untitled",
                status=t.status,
                plan_id=t.plan_id,
                plan_status=plan.status if plan else None,
                pending_approvals=pending,
                artifact_count=len(artifacts),
                latest_sequence=seq,
                requester_id=t.requester_id,
                updated_at=t.updated_at.isoformat(),
            ))
        return out

    async def get_task_detail(
        self,
        task_id: str,
        *,
        recent_event_limit: int = 50,
    ) -> TaskDetail | None:
        task = await self._repo.get_task(task_id)
        if task is None:
            return None
        ws = await self._repo.get_workspace(task.workspace_id)
        if ws is None:
            return None

        plan = await self._repo.get_plan(task.plan_id) if task.plan_id else None
        steps: list[PlanStep] = []
        if plan is not None:
            steps = await self._repo.list_steps(
                plan.plan_id, workspace_id=ws.workspace_id,
            )
        approvals = await self._repo.list_approvals_for_task(
            task_id, workspace_id=ws.workspace_id,
        )
        artifacts = await self._repo.list_task_artifacts(
            task_id, workspace_id=ws.workspace_id,
        )

        all_events = await self._events.list_events(
            ws.workspace_id, since_sequence=0, limit=10_000,
        )
        task_events = [e for e in all_events if e.task_id == task_id]
        latest_seq = (
            max((e.sequence for e in all_events), default=0)
        )
        recent = task_events[-recent_event_limit:]

        actions = self._derive_actions(task, plan, approvals, steps)
        return TaskDetail(
            workspace=ws,
            task=task,
            active_plan=plan,
            steps=steps,
            approvals=approvals,
            artifacts=artifacts,
            recent_events=recent,
            latest_sequence=latest_seq,
            available_actions=actions,
        )

    # ── artifacts / events ────────────────────────

    async def get_artifact(self, artifact_id: str) -> Artifact | None:
        return await self._repo.get_artifact(artifact_id)

    async def list_events_for_task(
        self,
        task_id: str,
        *,
        since_sequence: int = 0,
        limit: int = 500,
    ) -> tuple[list[ExecutionEvent], int]:
        """返回 (events, latest_sequence)；找不到 task 时抛 LookupError。"""
        task = await self._repo.get_task(task_id)
        if task is None:
            msg = f"unknown task: {task_id}"
            raise LookupError(msg)
        events = await self._events.list_events(
            task.workspace_id,
            since_sequence=since_sequence,
            limit=limit,
        )
        latest_seq = max((e.sequence for e in events), default=since_sequence)
        filtered = [e for e in events if e.task_id == task_id]
        return filtered, latest_seq

    async def get_workspace_id_for_task(self, task_id: str) -> str | None:
        task = await self._repo.get_task(task_id)
        return task.workspace_id if task is not None else None

    # ── helpers ────────────────────────────────────

    async def _latest_sequence(self, workspace_id: str) -> int:
        # InMemory 与 SQLite 都按 sequence 升序返回；取最后一条。
        events = await self._events.list_events(
            workspace_id, since_sequence=0, limit=10_000,
        )
        return events[-1].sequence if events else 0

    @staticmethod
    def _derive_actions(
        task: Task,
        plan: Plan | None,
        approvals: list[Approval],
        steps: list[PlanStep],
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        if task.status == TaskStatus.AWAITING_APPROVAL and plan is not None:
            for a in approvals:
                if (
                    a.target_id == plan.plan_id
                    and a.status == ApprovalStatus.REQUESTED
                ):
                    actions.append({
                        "type": "decide_plan",
                        "approval_id": a.approval_id,
                        "decisions": ["approve", "reject"],
                    })
        if task.status == TaskStatus.BLOCKED:
            for step in steps:
                if step.status == PlanStepStatus.WAITING_APPROVAL:
                    pending = [
                        a for a in approvals
                        if a.target_id == step.step_id
                        and a.status == ApprovalStatus.REQUESTED
                    ]
                    for a in pending:
                        actions.append({
                            "type": "decide_step",
                            "approval_id": a.approval_id,
                            "step_id": step.step_id,
                            "decisions": ["approve", "reject"],
                        })
        if task.status in (TaskStatus.FAILED, TaskStatus.RETRYABLE_FAILED):
            actions.append({"type": "retry_task"})
            failed_steps = [
                s for s in steps if s.status == PlanStepStatus.FAILED
            ]
            for s in failed_steps:
                actions.append({"type": "recover_from", "step_id": s.step_id})
        return actions


__all__ = [
    "PilotQueryService",
    "TaskDetail",
    "TaskSummary",
    "WorkspaceSummary",
]
