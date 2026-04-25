"""Pilot 状态机：纯函数 ``transition`` 与 Plan DAG 校验。

不写库、不发事件、不访问外部资源。所有状态变更必须通过 ``transition``。
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from agent_hub.pilot.domain.enums import (
    ApprovalAction,
    ApprovalStatus,
    ArtifactAction,
    ArtifactStatus,
    PlanAction,
    PlanStatus,
    PlanStepAction,
    PlanStepStatus,
    TaskAction,
    TaskStatus,
    WorkspaceAction,
    WorkspaceStatus,
)
from agent_hub.pilot.domain.errors import IllegalTransition, PlanGraphError
from agent_hub.pilot.domain.models import (
    Approval,
    Artifact,
    Plan,
    PlanStep,
    Task,
    Workspace,
)

Entity = Workspace | Task | Plan | PlanStep | Approval | Artifact
Action = (
    WorkspaceAction
    | TaskAction
    | PlanAction
    | PlanStepAction
    | ApprovalAction
    | ArtifactAction
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


# ── 状态迁移表 ───────────────────────────────────────


_WORKSPACE_TABLE: dict[tuple[WorkspaceStatus, WorkspaceAction], WorkspaceStatus] = {
    (WorkspaceStatus.ACTIVE, WorkspaceAction.ARCHIVE): WorkspaceStatus.ARCHIVED,
    (WorkspaceStatus.ACTIVE, WorkspaceAction.FAIL): WorkspaceStatus.FAILED,
    (WorkspaceStatus.ARCHIVED, WorkspaceAction.REACTIVATE): WorkspaceStatus.ACTIVE,
    (WorkspaceStatus.FAILED, WorkspaceAction.REACTIVATE): WorkspaceStatus.ACTIVE,
}


_TASK_TABLE: dict[tuple[TaskStatus, TaskAction], TaskStatus] = {
    (TaskStatus.CREATED, TaskAction.START_PLANNING): TaskStatus.PLANNING,
    (TaskStatus.CREATED, TaskAction.CANCEL): TaskStatus.CANCELLED,
    (TaskStatus.PLANNING, TaskAction.REQUEST_APPROVAL): TaskStatus.AWAITING_APPROVAL,
    (TaskStatus.PLANNING, TaskAction.FAIL): TaskStatus.FAILED,
    (TaskStatus.PLANNING, TaskAction.CANCEL): TaskStatus.CANCELLED,
    (TaskStatus.AWAITING_APPROVAL, TaskAction.APPROVE): TaskStatus.APPROVED,
    (TaskStatus.AWAITING_APPROVAL, TaskAction.FAIL): TaskStatus.FAILED,
    (TaskStatus.AWAITING_APPROVAL, TaskAction.CANCEL): TaskStatus.CANCELLED,
    (TaskStatus.APPROVED, TaskAction.START_RUNNING): TaskStatus.RUNNING,
    (TaskStatus.APPROVED, TaskAction.CANCEL): TaskStatus.CANCELLED,
    (TaskStatus.RUNNING, TaskAction.BLOCK): TaskStatus.BLOCKED,
    (TaskStatus.RUNNING, TaskAction.SUCCEED): TaskStatus.SUCCEEDED,
    (TaskStatus.RUNNING, TaskAction.FAIL): TaskStatus.FAILED,
    (TaskStatus.RUNNING, TaskAction.MARK_RETRYABLE): TaskStatus.RETRYABLE_FAILED,
    (TaskStatus.RUNNING, TaskAction.CANCEL): TaskStatus.CANCELLED,
    (TaskStatus.BLOCKED, TaskAction.UNBLOCK): TaskStatus.RUNNING,
    (TaskStatus.BLOCKED, TaskAction.FAIL): TaskStatus.FAILED,
    (TaskStatus.BLOCKED, TaskAction.CANCEL): TaskStatus.CANCELLED,
    (TaskStatus.RETRYABLE_FAILED, TaskAction.RETRY): TaskStatus.RUNNING,
    (TaskStatus.RETRYABLE_FAILED, TaskAction.FAIL): TaskStatus.FAILED,
    (TaskStatus.RETRYABLE_FAILED, TaskAction.CANCEL): TaskStatus.CANCELLED,
    (TaskStatus.SUCCEEDED, TaskAction.ARCHIVE): TaskStatus.ARCHIVED,
    (TaskStatus.FAILED, TaskAction.ARCHIVE): TaskStatus.ARCHIVED,
    (TaskStatus.CANCELLED, TaskAction.ARCHIVE): TaskStatus.ARCHIVED,
}


_PLAN_TABLE: dict[tuple[PlanStatus, PlanAction], PlanStatus] = {
    (PlanStatus.DRAFT, PlanAction.REQUEST_APPROVAL): PlanStatus.APPROVAL_REQUESTED,
    (PlanStatus.DRAFT, PlanAction.SUPERSEDE): PlanStatus.SUPERSEDED,
    (PlanStatus.APPROVAL_REQUESTED, PlanAction.APPROVE): PlanStatus.APPROVED,
    (PlanStatus.APPROVAL_REQUESTED, PlanAction.REJECT): PlanStatus.REJECTED,
    (PlanStatus.APPROVAL_REQUESTED, PlanAction.SUPERSEDE): PlanStatus.SUPERSEDED,
    (PlanStatus.APPROVED, PlanAction.SUPERSEDE): PlanStatus.SUPERSEDED,
    (PlanStatus.REJECTED, PlanAction.SUPERSEDE): PlanStatus.SUPERSEDED,
}


_PLAN_STEP_TABLE: dict[tuple[PlanStepStatus, PlanStepAction], PlanStepStatus] = {
    (PlanStepStatus.PENDING, PlanStepAction.START_DRY_RUN): PlanStepStatus.DRY_RUNNING,
    (PlanStepStatus.PENDING, PlanStepAction.REQUEST_APPROVAL): PlanStepStatus.WAITING_APPROVAL,
    (PlanStepStatus.PENDING, PlanStepAction.START_RUNNING): PlanStepStatus.RUNNING,
    (PlanStepStatus.PENDING, PlanStepAction.SKIP): PlanStepStatus.SKIPPED,
    (PlanStepStatus.PENDING, PlanStepAction.CANCEL): PlanStepStatus.CANCELLED,
    (PlanStepStatus.DRY_RUNNING, PlanStepAction.REQUEST_APPROVAL): PlanStepStatus.WAITING_APPROVAL,
    (PlanStepStatus.DRY_RUNNING, PlanStepAction.START_RUNNING): PlanStepStatus.RUNNING,
    (PlanStepStatus.DRY_RUNNING, PlanStepAction.FAIL): PlanStepStatus.FAILED,
    (PlanStepStatus.DRY_RUNNING, PlanStepAction.CANCEL): PlanStepStatus.CANCELLED,
    (PlanStepStatus.WAITING_APPROVAL, PlanStepAction.APPROVE): PlanStepStatus.APPROVED,
    (PlanStepStatus.WAITING_APPROVAL, PlanStepAction.CANCEL): PlanStepStatus.CANCELLED,
    (PlanStepStatus.WAITING_APPROVAL, PlanStepAction.FAIL): PlanStepStatus.FAILED,
    (PlanStepStatus.APPROVED, PlanStepAction.START_RUNNING): PlanStepStatus.RUNNING,
    (PlanStepStatus.APPROVED, PlanStepAction.CANCEL): PlanStepStatus.CANCELLED,
    (PlanStepStatus.RUNNING, PlanStepAction.SUCCEED): PlanStepStatus.SUCCEEDED,
    (PlanStepStatus.RUNNING, PlanStepAction.FAIL): PlanStepStatus.FAILED,
    (PlanStepStatus.RUNNING, PlanStepAction.MARK_RETRYABLE): PlanStepStatus.RETRYABLE_FAILED,
    (PlanStepStatus.RUNNING, PlanStepAction.CANCEL): PlanStepStatus.CANCELLED,
    (PlanStepStatus.RETRYABLE_FAILED, PlanStepAction.RETRY): PlanStepStatus.RUNNING,
    (PlanStepStatus.RETRYABLE_FAILED, PlanStepAction.FAIL): PlanStepStatus.FAILED,
    (PlanStepStatus.RETRYABLE_FAILED, PlanStepAction.CANCEL): PlanStepStatus.CANCELLED,
}


_APPROVAL_TABLE: dict[tuple[ApprovalStatus, ApprovalAction], ApprovalStatus] = {
    (ApprovalStatus.REQUESTED, ApprovalAction.APPROVE): ApprovalStatus.APPROVED,
    (ApprovalStatus.REQUESTED, ApprovalAction.REJECT): ApprovalStatus.REJECTED,
    (ApprovalStatus.REQUESTED, ApprovalAction.EXPIRE): ApprovalStatus.EXPIRED,
    (ApprovalStatus.REQUESTED, ApprovalAction.SUPERSEDE): ApprovalStatus.SUPERSEDED,
    (ApprovalStatus.REQUESTED, ApprovalAction.AUTO_APPROVE): ApprovalStatus.AUTO_APPROVED,
    (ApprovalStatus.REQUESTED, ApprovalAction.CANCEL): ApprovalStatus.CANCELLED,
}


_ARTIFACT_TABLE: dict[tuple[ArtifactStatus, ArtifactAction], ArtifactStatus] = {
    (ArtifactStatus.DRAFT, ArtifactAction.GENERATE): ArtifactStatus.GENERATED,
    (ArtifactStatus.DRAFT, ArtifactAction.FAIL): ArtifactStatus.FAILED,
    (ArtifactStatus.GENERATED, ArtifactAction.UPLOAD): ArtifactStatus.UPLOADED,
    (ArtifactStatus.GENERATED, ArtifactAction.SUPERSEDE): ArtifactStatus.SUPERSEDED,
    (ArtifactStatus.GENERATED, ArtifactAction.FAIL): ArtifactStatus.FAILED,
    (ArtifactStatus.UPLOADED, ArtifactAction.SHARE): ArtifactStatus.SHARED,
    (ArtifactStatus.UPLOADED, ArtifactAction.SUPERSEDE): ArtifactStatus.SUPERSEDED,
    (ArtifactStatus.UPLOADED, ArtifactAction.FAIL): ArtifactStatus.FAILED,
    (ArtifactStatus.SHARED, ArtifactAction.SUPERSEDE): ArtifactStatus.SUPERSEDED,
    (ArtifactStatus.SHARED, ArtifactAction.ARCHIVE): ArtifactStatus.ARCHIVED,
    (ArtifactStatus.SUPERSEDED, ArtifactAction.ARCHIVE): ArtifactStatus.ARCHIVED,
    (ArtifactStatus.FAILED, ArtifactAction.ARCHIVE): ArtifactStatus.ARCHIVED,
}


# ── 分派表 ───────────────────────────────────────────


_DISPATCH: dict[type, tuple[type, dict[tuple[Any, Any], Any], str, str]] = {
    Workspace: (WorkspaceAction, _WORKSPACE_TABLE, "Workspace", "workspace_id"),
    Task: (TaskAction, _TASK_TABLE, "Task", "task_id"),
    Plan: (PlanAction, _PLAN_TABLE, "Plan", "plan_id"),
    PlanStep: (PlanStepAction, _PLAN_STEP_TABLE, "PlanStep", "step_id"),
    Approval: (ApprovalAction, _APPROVAL_TABLE, "Approval", "approval_id"),
    Artifact: (ArtifactAction, _ARTIFACT_TABLE, "Artifact", "artifact_id"),
}


# ── 公共入口 ─────────────────────────────────────────


def transition(
    obj: Entity,
    action: Action,
    *,
    now: datetime | None = None,
    patch: dict[str, Any] | None = None,
) -> Entity:
    """对实体执行一次状态迁移，返回新对象。

    Args:
        obj: 当前实体快照。
        action: 动作枚举（必须与实体类型匹配）。
        now: 用于 ``updated_at`` 的时间戳；默认为 ``datetime.now(UTC)``。
        patch: 额外字段更新（在状态更新之外应用，如 error/decided_at/started_at 等）。

    Returns:
        新的实体对象，``status``/``version``/``updated_at`` 已更新。

    Raises:
        IllegalTransition: 当前状态下不允许该动作。
        TypeError: action 类型与实体类型不匹配。
    """
    entry = _DISPATCH.get(type(obj))
    if entry is None:
        msg = f"未知实体类型: {type(obj).__name__}"
        raise TypeError(msg)

    action_cls, table, entity_name, id_field = entry
    if not isinstance(action, action_cls):
        msg = (
            f"动作类型不匹配: {entity_name} 需要 {action_cls.__name__}，"
            f"得到 {type(action).__name__}"
        )
        raise TypeError(msg)

    current_status = obj.status  # type: ignore[attr-defined]
    new_status = table.get((current_status, action))
    if new_status is None:
        raise IllegalTransition(
            entity_type=entity_name,
            entity_id=getattr(obj, id_field),
            current_status=current_status.value,
            action=action.value,
        )

    update: dict[str, Any] = {
        "status": new_status,
        "version": obj.version + 1,
        "updated_at": now or _utcnow(),
    }
    if patch:
        # patch 不允许覆盖 status / version / updated_at
        for forbidden in ("status", "version", "updated_at"):
            if forbidden in patch:
                msg = f"patch 不能修改受控字段: {forbidden}"
                raise ValueError(msg)
        update.update(patch)

    return obj.model_copy(update=update)


# ── Plan DAG 校验 ────────────────────────────────────


def validate_plan_graph(steps: Iterable[PlanStep]) -> None:
    """校验同一 Plan 内的 PlanStep 构成合法 DAG。

    规则：
    - step_id 唯一；
    - index 唯一；
    - 同属一个 plan_id；
    - depends_on 仅引用同一 Plan 内的 step_id；
    - 无环。

    Raises:
        PlanGraphError: 任一规则失败。
    """
    step_list = list(steps)
    if not step_list:
        return

    plan_ids = {s.plan_id for s in step_list}
    if len(plan_ids) > 1:
        msg = f"步骤跨越多个 plan_id: {plan_ids}"
        raise PlanGraphError(msg)

    ids = [s.step_id for s in step_list]
    if len(set(ids)) != len(ids):
        msg = "存在重复的 step_id"
        raise PlanGraphError(msg)

    indices = [s.index for s in step_list]
    if len(set(indices)) != len(indices):
        msg = "存在重复的 step.index"
        raise PlanGraphError(msg)

    id_set = set(ids)
    for s in step_list:
        for dep in s.depends_on:
            if dep not in id_set:
                msg = f"step {s.step_id} 依赖未知 step {dep}"
                raise PlanGraphError(msg)
            if dep == s.step_id:
                msg = f"step {s.step_id} 不能依赖自身"
                raise PlanGraphError(msg)

    # Kahn 拓扑排序检测环
    in_degree = {s.step_id: len(s.depends_on) for s in step_list}
    children: dict[str, list[str]] = {s.step_id: [] for s in step_list}
    for s in step_list:
        for dep in s.depends_on:
            children[dep].append(s.step_id)

    queue = [sid for sid, d in in_degree.items() if d == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for child in children[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if visited != len(step_list):
        msg = "Plan DAG 中存在环依赖"
        raise PlanGraphError(msg)
