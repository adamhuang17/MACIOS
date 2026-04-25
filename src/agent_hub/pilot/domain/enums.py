"""Pilot 领域枚举。

包含实体状态、动作、事件类型等。所有值使用稳定小写字符串，
便于 SQLite JSON、SSE 与前端消费。
"""

from __future__ import annotations

from enum import StrEnum

# ── 实体状态 ─────────────────────────────────────────


class WorkspaceStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    FAILED = "failed"


class TaskStatus(StrEnum):
    CREATED = "created"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    RUNNING = "running"
    BLOCKED = "blocked"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RETRYABLE_FAILED = "retryable_failed"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"


class PlanStatus(StrEnum):
    DRAFT = "draft"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class PlanStepStatus(StrEnum):
    PENDING = "pending"
    DRY_RUNNING = "dry_running"
    WAITING_APPROVAL = "waiting_approval"
    APPROVED = "approved"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RETRYABLE_FAILED = "retryable_failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class ApprovalStatus(StrEnum):
    REQUESTED = "requested"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"
    AUTO_APPROVED = "auto_approved"
    CANCELLED = "cancelled"


class ArtifactStatus(StrEnum):
    DRAFT = "draft"
    GENERATED = "generated"
    UPLOADED = "uploaded"
    SHARED = "shared"
    SUPERSEDED = "superseded"
    FAILED = "failed"
    ARCHIVED = "archived"


# ── 动作 ─────────────────────────────────────────────


class WorkspaceAction(StrEnum):
    ARCHIVE = "archive"
    FAIL = "fail"
    REACTIVATE = "reactivate"


class TaskAction(StrEnum):
    START_PLANNING = "start_planning"
    REQUEST_APPROVAL = "request_approval"
    APPROVE = "approve"
    START_RUNNING = "start_running"
    BLOCK = "block"
    UNBLOCK = "unblock"
    SUCCEED = "succeed"
    FAIL = "fail"
    MARK_RETRYABLE = "mark_retryable"
    RETRY = "retry"
    CANCEL = "cancel"
    ARCHIVE = "archive"


class PlanAction(StrEnum):
    REQUEST_APPROVAL = "request_approval"
    APPROVE = "approve"
    REJECT = "reject"
    SUPERSEDE = "supersede"


class PlanStepAction(StrEnum):
    START_DRY_RUN = "start_dry_run"
    REQUEST_APPROVAL = "request_approval"
    APPROVE = "approve"
    START_RUNNING = "start_running"
    SUCCEED = "succeed"
    FAIL = "fail"
    MARK_RETRYABLE = "mark_retryable"
    RETRY = "retry"
    SKIP = "skip"
    CANCEL = "cancel"


class ApprovalAction(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    EXPIRE = "expire"
    SUPERSEDE = "supersede"
    AUTO_APPROVE = "auto_approve"
    CANCEL = "cancel"


class ArtifactAction(StrEnum):
    GENERATE = "generate"
    UPLOAD = "upload"
    SHARE = "share"
    SUPERSEDE = "supersede"
    FAIL = "fail"
    ARCHIVE = "archive"


# ── 事件 / 审计 ──────────────────────────────────────


class EventLevel(StrEnum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class ActorType(StrEnum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"
    SKILL = "skill"


class EventType(StrEnum):
    WORKSPACE_CREATED = "workspace.created"
    WORKSPACE_STATUS_CHANGED = "workspace.status_changed"
    TASK_CREATED = "task.created"
    TASK_STATUS_CHANGED = "task.status_changed"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    PLAN_GENERATED = "plan.generated"
    PLAN_STATUS_CHANGED = "plan.status_changed"
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_DECIDED = "approval.decided"
    APPROVAL_EXPIRED = "approval.expired"
    APPROVAL_SUPERSEDED = "approval.superseded"
    STEP_STATUS_CHANGED = "step.status_changed"
    SKILL_DRY_RUN = "skill.dry_run"
    SKILL_INVOKED = "skill.invoked"
    ARTIFACT_CREATED = "artifact.created"
    ARTIFACT_STATUS_CHANGED = "artifact.status_changed"


# ── 风险 / 类型 ──────────────────────────────────────


class RiskLevel(StrEnum):
    READ = "read"
    WRITE = "write"
    SHARE = "share"
    ADMIN = "admin"


class ApprovalTargetType(StrEnum):
    PLAN = "plan"
    STEP = "step"
    SHARE = "share"
    WRITE = "write"


class PlanStepKind(StrEnum):
    READ_CONTEXT = "read_context"
    GENERATE_DOC = "generate_doc"
    GENERATE_SLIDE_SPEC = "generate_slide_spec"
    RENDER_SLIDES = "render_slides"
    UPLOAD = "upload"
    SHARE = "share"
    SUMMARIZE = "summarize"
    MANUAL = "manual"


class ArtifactType(StrEnum):
    CONTEXT_BUNDLE = "context_bundle"
    PROJECT_BRIEF_MD = "project_brief_md"
    SLIDE_SPEC_JSON = "slide_spec_json"
    PPTX = "pptx"
    FEISHU_DOC = "feishu_doc"
    FEISHU_SLIDE = "feishu_slide"
    DRIVE_FILE = "drive_file"
    SHARE_LINK = "share_link"
    SUMMARY = "summary"
