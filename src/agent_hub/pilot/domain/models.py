"""Pilot 领域 Pydantic 模型：实体快照。

所有时间字段使用 UTC aware datetime。
所有实体均带 ``version`` 字段用于乐观并发控制。
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_hub.pilot.domain.enums import (
    ApprovalStatus,
    ApprovalTargetType,
    ArtifactStatus,
    ArtifactType,
    PlanStatus,
    PlanStepKind,
    PlanStepStatus,
    RiskLevel,
    TaskStatus,
    WorkspaceStatus,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class _PilotModel(BaseModel):
    """Pilot 实体共享基类。

    - frozen=True: 强制通过 ``model_copy(update=...)`` 产生新对象，配合状态机使用。
    - 所有实体必有 ``version``、``created_at``、``updated_at``。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = 1
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


# ── Workspace ────────────────────────────────────────


class Workspace(_PilotModel):
    workspace_id: str = Field(default_factory=lambda: _new_id("ws"))
    tenant_key: str | None = None
    title: str
    source_channel: str
    source_conversation_id: str | None = None
    feishu_chat_id: str | None = None
    created_by: str
    members: list[str] = Field(default_factory=list)
    status: WorkspaceStatus = WorkspaceStatus.ACTIVE
    active_task_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Task ─────────────────────────────────────────────


class Task(_PilotModel):
    task_id: str = Field(default_factory=lambda: _new_id("task"))
    workspace_id: str
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    idempotency_key: str | None = None
    source_message_id: str | None = None
    origin_text: str
    requester_id: str
    status: TaskStatus = TaskStatus.CREATED
    plan_id: str | None = None
    current_step_id: str | None = None
    approval_ids: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    parent_task_id: str | None = None
    parent_step_id: str | None = None
    retry_count: int = 0
    error: str | None = None
    final_summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Plan ─────────────────────────────────────────────


class Plan(_PilotModel):
    plan_id: str = Field(default_factory=lambda: _new_id("plan"))
    workspace_id: str
    task_id: str
    plan_version: int = 1
    parent_plan_id: str | None = None
    goal: str
    assumptions: list[str] = Field(default_factory=list)
    step_ids: list[str] = Field(default_factory=list)
    status: PlanStatus = PlanStatus.DRAFT
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── PlanStep ─────────────────────────────────────────


class PlanStep(_PilotModel):
    step_id: str = Field(default_factory=lambda: _new_id("step"))
    workspace_id: str
    task_id: str
    plan_id: str
    index: int = Field(ge=0)
    title: str
    kind: PlanStepKind
    skill_name: str
    input_params: dict[str, Any] = Field(default_factory=dict)
    inputs_from: list[str] = Field(default_factory=list)
    output_artifact_ref: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.READ
    requires_approval: bool = False
    status: PlanStepStatus = PlanStepStatus.PENDING
    retry_count: int = 0
    max_retries: int = 0
    timeout_ms: int = 30_000
    idempotency_key: str | None = None
    dry_run_result: dict[str, Any] | None = None
    output_artifact_id: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


# ── Approval ─────────────────────────────────────────


class ApprovalDecision(BaseModel):
    """单条审批决议记录。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    approver_id: str
    decision: str  # approved / rejected / requested_changes
    comment: str | None = None
    decided_at: datetime = Field(default_factory=_utcnow)


class Approval(_PilotModel):
    approval_id: str = Field(default_factory=lambda: _new_id("appr"))
    workspace_id: str
    task_id: str
    target_type: ApprovalTargetType
    target_id: str
    requester_id: str
    approvers: list[str] = Field(default_factory=list)
    reason: str | None = None
    preview: str | None = None
    status: ApprovalStatus = ApprovalStatus.REQUESTED
    decision_records: list[ApprovalDecision] = Field(default_factory=list)
    channel_message_id: str | None = None
    expires_at: datetime | None = None
    decided_at: datetime | None = None
    superseded_by: str | None = None


# ── Artifact ─────────────────────────────────────────


class Artifact(_PilotModel):
    artifact_id: str = Field(default_factory=lambda: _new_id("art"))
    workspace_id: str
    task_id: str
    step_id: str | None = None
    type: ArtifactType
    title: str
    artifact_version: int = 1
    parent_artifact_id: str | None = None
    status: ArtifactStatus = ArtifactStatus.DRAFT
    uri: str | None = None
    storage_key: str | None = None
    mime_type: str | None = None
    checksum: str | None = None
    feishu_token: str | None = None
    share_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
