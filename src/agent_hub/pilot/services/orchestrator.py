"""TaskOrchestrator：M1 主入口。

职责：
- 创建/复用 Workspace；
- 通过 ExecutionEvent 的 idempotency_key 做去重，避免重复创建 Task；
- 走完 Task 创建 → 规划 → 审批申请 → （可选）自动审批 → 执行；
- 返回轻量 :class:`TaskHandle`，详细状态由调用方按 task_id 再查。

为了让 idempotency 可用，调用方需要要么传入 ``workspace_id``，
要么允许 orchestrator 创建新的 workspace；同一个 workspace 内的
重复 ``idempotency_key`` 会返回首个 task 的 handle。
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import structlog

from agent_hub.pilot.domain.enums import (
    EventType,
    TaskAction,
    WorkspaceStatus,
)
from agent_hub.pilot.domain.events import ExecutionEvent
from agent_hub.pilot.domain.models import Task, Workspace
from agent_hub.pilot.domain.state import transition
from agent_hub.pilot.services.dto import RunResult, TaskHandle, TaskRequest
from agent_hub.pilot.services.model_gateway import PlanContext

if TYPE_CHECKING:
    from agent_hub.pilot.services.approval import ApprovalService
    from agent_hub.pilot.services.event_publisher import PilotEventPublisher
    from agent_hub.pilot.services.execution import ExecutionEngine
    from agent_hub.pilot.services.planning import PlanningService
    from agent_hub.pilot.services.repository import PilotRepository

logger = structlog.get_logger(__name__)


class TaskOrchestrator:
    def __init__(
        self,
        repository: PilotRepository,
        publisher: PilotEventPublisher,
        planning: PlanningService,
        approvals: ApprovalService,
        execution: ExecutionEngine,
    ) -> None:
        self._repo = repository
        self._publisher = publisher
        self._planning = planning
        self._approvals = approvals
        self._execution = execution

    async def submit(self, request: TaskRequest) -> TaskHandle:
        workspace = await self._get_or_create_workspace(request)

        idem_key = request.idempotency_key or self._derive_idem_key(request)
        trace_id = uuid.uuid4().hex
        candidate = Task(
            workspace_id=workspace.workspace_id,
            trace_id=trace_id,
            idempotency_key=idem_key,
            source_message_id=request.source_message_id,
            origin_text=request.raw_text,
            requester_id=request.requester_id,
            metadata=dict(request.metadata),
        )

        # 先尝试通过 ExecutionEvent.idempotency_key 实现幂等：
        # 如果 store 已有同 (workspace_id, idem_key) 事件，则视为重复请求。
        proposed_event = ExecutionEvent(
            workspace_id=workspace.workspace_id,
            trace_id=candidate.trace_id,
            task_id=candidate.task_id,
            type=EventType.TASK_CREATED,
            message="task created",
            idempotency_key=idem_key,
            payload={"task_id": candidate.task_id},
        )
        stored_event = await self._publisher.record(proposed_event)

        if stored_event.task_id and stored_event.task_id != candidate.task_id:
            existing = await self._repo.get_task(stored_event.task_id)
            if existing is not None:
                logger.info(
                    "pilot.orchestrator_dedup",
                    workspace_id=workspace.workspace_id,
                    task_id=existing.task_id,
                )
                return TaskHandle(
                    workspace_id=existing.workspace_id,
                    task_id=existing.task_id,
                    plan_id=existing.plan_id,
                    status=existing.status,
                    trace_id=existing.trace_id,
                    deduplicated=True,
                )

        # 真正创建 Task 快照
        task = candidate
        await self._repo.save(task, expected_version=None)

        # PLANNING
        task = transition(task, TaskAction.START_PLANNING)
        await self._repo.save(task)

        plan_ctx = PlanContext(
            raw_text=request.raw_text,
            requester_id=request.requester_id,
            title=request.title,
            attachments=list(request.attachments),
            metadata=dict(request.metadata),
            profile=request.plan_profile,
            source_channel=request.source_channel,
            source_conversation_id=request.source_conversation_id,
            source_message_id=request.source_message_id,
        )
        plan, _ = await self._planning.generate(task, plan_ctx)
        task = task.model_copy(update={"plan_id": plan.plan_id})
        # plan_id 是非受控字段，用 expected_version=None 跳过版本检查
        await self._repo.save(task, expected_version=None)

        # 审批
        if request.auto_approve:
            _, plan = await self._approvals.auto_approve_plan(
                plan, task, requester_id=request.requester_id,
            )
        else:
            _, plan = await self._approvals.request_plan_approval(
                plan, task, requester_id=request.requester_id,
            )
            task = transition(task, TaskAction.REQUEST_APPROVAL)
            await self._repo.save(task)
            return TaskHandle(
                workspace_id=workspace.workspace_id,
                task_id=task.task_id,
                plan_id=plan.plan_id,
                status=task.status,
                trace_id=task.trace_id,
            )

        # 自动审批分支：APPROVED → 执行
        task = transition(task, TaskAction.REQUEST_APPROVAL)
        await self._repo.save(task)
        task = transition(task, TaskAction.APPROVE)
        await self._repo.save(task)

        result = await self._execution.run_plan(task, plan)
        return TaskHandle(
            workspace_id=workspace.workspace_id,
            task_id=result.task.task_id,
            plan_id=plan.plan_id,
            status=result.final_status,
            trace_id=result.task.trace_id,
        )

    async def run_after_approval(self, task_id: str) -> RunResult:
        task = await self._repo.get_task(task_id)
        if task is None:
            msg = f"未知 task: {task_id}"
            raise LookupError(msg)
        if task.plan_id is None:
            msg = f"task {task_id} 没有关联 plan"
            raise ValueError(msg)
        plan = await self._repo.get_plan(task.plan_id)
        if plan is None:
            msg = f"未知 plan: {task.plan_id}"
            raise LookupError(msg)

        task = transition(task, TaskAction.APPROVE)
        await self._repo.save(task)
        return await self._execution.run_plan(task, plan)

    # ── helpers ────────────────────────────────────

    async def _get_or_create_workspace(self, request: TaskRequest) -> Workspace:
        if request.workspace_id is not None:
            existing = await self._repo.get_workspace(request.workspace_id)
            if existing is not None:
                return existing

        ws_kwargs = {
            "tenant_key": request.tenant_key,
            "title": request.title or request.raw_text[:32] or "Untitled Workspace",
            "source_channel": request.source_channel,
            "source_conversation_id": request.source_conversation_id,
            "created_by": request.requester_id,
            "members": [request.requester_id],
            "status": WorkspaceStatus.ACTIVE,
        }
        if request.workspace_id is not None:
            ws_kwargs["workspace_id"] = request.workspace_id
        workspace = Workspace(**ws_kwargs)
        await self._repo.save(workspace, expected_version=None)
        await self._publisher.record(ExecutionEvent(
            workspace_id=workspace.workspace_id,
            type=EventType.WORKSPACE_CREATED,
            message="workspace created",
        ))
        return workspace

    @staticmethod
    def _derive_idem_key(request: TaskRequest) -> str:
        if request.source_message_id:
            return f"msg:{request.source_channel}:{request.source_message_id}"
        # 无外部唯一标识时，每次请求生成新 key（不去重）
        return f"auto:{uuid.uuid4().hex}"


__all__ = ["TaskOrchestrator"]
