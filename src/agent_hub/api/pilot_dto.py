"""Pilot M2 API DTO：HTTP 边界上的 request/response 模型。

仅依赖 pydantic 与 Pilot 领域枚举/模型；FastAPI 路由通过本模块上送下行。
read 模型默认允许从 ORM-like 对象（``model_validate(..., from_attributes=True)``）
适配，以便复用 ``PilotQueryService`` 的输出。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_hub.pilot.domain.enums import PlanStatus, TaskStatus
from agent_hub.pilot.domain.events import ExecutionEvent
from agent_hub.pilot.domain.models import (
    Approval,
    Artifact,
    Plan,
    PlanStep,
    Task,
    Workspace,
)
from agent_hub.pilot.services.dto import PlanProfile

# ── 提交任务 ─────────────────────────────────────────


class SubmitTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_channel: str = Field(default="api")
    raw_text: str
    requester_id: str
    title: str | None = None
    source_conversation_id: str | None = None
    source_message_id: str | None = None
    workspace_id: str | None = None
    tenant_key: str | None = None
    attachments: list[str] = Field(default_factory=list)
    idempotency_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    auto_approve: bool = False
    plan_profile: PlanProfile | None = None


class SubmitTaskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    task_id: str
    plan_id: str | None
    status: TaskStatus
    trace_id: str
    deduplicated: bool
    detail_url: str
    events_url: str


# ── Dashboard 读模型 ─────────────────────────────────


class TaskSummaryView(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

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


class WorkspaceSummaryView(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    workspace: Workspace
    task_count: int
    active_task_id: str | None
    latest_sequence: int


class TaskDetailView(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    workspace: Workspace
    task: Task
    active_plan: Plan | None
    steps: list[PlanStep]
    approvals: list[Approval]
    artifacts: list[Artifact]
    recent_events: list[ExecutionEvent]
    latest_sequence: int
    available_actions: list[dict[str, Any]] = Field(default_factory=list)


class ArtifactView(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    artifact_id: str
    workspace_id: str
    task_id: str
    step_id: str | None
    type: str
    title: str
    artifact_version: int
    parent_artifact_id: str | None
    status: str
    uri: str | None
    storage_key: str | None
    mime_type: str | None
    checksum: str | None
    feishu_token: str | None
    share_url: str | None
    metadata: dict[str, Any]
    version: int
    created_at: Any
    updated_at: Any


# ── 审批 ────────────────────────────────────────────


class ApprovalDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "reject"]
    actor_id: str
    comment: str | None = None
    run_after_approval: bool = True


class ApprovalDecisionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    approval_id: str
    approval_status: str
    task_id: str
    task_status: str | None = None
    plan_id: str | None = None
    plan_status: str | None = None
    step_id: str | None = None
    step_status: str | None = None
    run_result: dict[str, Any] | None = None
    note: str | None = None


# ── 恢复任务 ────────────────────────────────────────


class RecoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requester_id: str
    auto_approve: bool = False


# ── 续跑任务 ────────────────────────────────────────


class ResumeTaskResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str
    task_status: str
    run_result: dict[str, Any] | None = None


__all__ = [
    "ApprovalDecisionRequest",
    "ApprovalDecisionResponse",
    "ArtifactView",
    "RecoveryRequest",
    "ResumeTaskResponse",
    "SubmitTaskRequest",
    "SubmitTaskResponse",
    "TaskDetailView",
    "TaskSummaryView",
    "WorkspaceSummaryView",
]
