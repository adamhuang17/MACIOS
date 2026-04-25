"""M0 领域模型与状态机测试。"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agent_hub.pilot import (
    Approval,
    ApprovalAction,
    ApprovalStatus,
    ApprovalTargetType,
    Artifact,
    ArtifactAction,
    ArtifactStatus,
    ArtifactType,
    IllegalTransition,
    Plan,
    PlanAction,
    PlanGraphError,
    PlanStatus,
    PlanStep,
    PlanStepAction,
    PlanStepKind,
    PlanStepStatus,
    Task,
    TaskAction,
    TaskStatus,
    Workspace,
    WorkspaceAction,
    WorkspaceStatus,
    transition,
    validate_plan_graph,
)

# ── 模型默认值 / UTC 时间 ─────────────────────────────


def _make_workspace() -> Workspace:
    return Workspace(title="t", source_channel="api", created_by="u1")


def _make_task(ws: Workspace) -> Task:
    return Task(
        workspace_id=ws.workspace_id,
        origin_text="hi",
        requester_id="u1",
    )


def test_workspace_defaults_utc() -> None:
    ws = _make_workspace()
    assert ws.status is WorkspaceStatus.ACTIVE
    assert ws.version == 1
    assert ws.created_at.tzinfo is not None
    assert ws.created_at.utcoffset() == UTC.utcoffset(datetime.now())


def test_pilot_model_is_frozen() -> None:
    ws = _make_workspace()
    with pytest.raises(Exception):
        ws.title = "x"  # type: ignore[misc]


def test_extra_fields_forbidden() -> None:
    with pytest.raises(Exception):
        Workspace(  # type: ignore[call-arg]
            title="t",
            source_channel="api",
            created_by="u1",
            unknown_field="x",
        )


# ── 状态机：合法迁移 ─────────────────────────────────


def test_task_happy_path() -> None:
    ws = _make_workspace()
    task = _make_task(ws)

    task = transition(task, TaskAction.START_PLANNING)
    assert task.status is TaskStatus.PLANNING
    assert task.version == 2

    task = transition(task, TaskAction.REQUEST_APPROVAL)
    assert task.status is TaskStatus.AWAITING_APPROVAL

    task = transition(task, TaskAction.APPROVE)
    assert task.status is TaskStatus.APPROVED

    task = transition(task, TaskAction.START_RUNNING)
    assert task.status is TaskStatus.RUNNING

    task = transition(task, TaskAction.SUCCEED)
    assert task.status is TaskStatus.SUCCEEDED
    assert task.version == 6


def test_workspace_archive_then_reactivate() -> None:
    ws = _make_workspace()
    ws = transition(ws, WorkspaceAction.ARCHIVE)
    assert ws.status is WorkspaceStatus.ARCHIVED
    ws = transition(ws, WorkspaceAction.REACTIVATE)
    assert ws.status is WorkspaceStatus.ACTIVE


def test_plan_supersede_from_any_active_state() -> None:
    plan = Plan(workspace_id="ws_x", task_id="t_x", goal="g")
    plan = transition(plan, PlanAction.REQUEST_APPROVAL)
    plan = transition(plan, PlanAction.APPROVE)
    plan = transition(plan, PlanAction.SUPERSEDE)
    assert plan.status is PlanStatus.SUPERSEDED


def test_plan_step_full_cycle() -> None:
    s = PlanStep(
        workspace_id="ws", task_id="t", plan_id="p",
        index=0, title="x", kind=PlanStepKind.READ_CONTEXT, skill_name="noop",
    )
    s = transition(s, PlanStepAction.START_DRY_RUN)
    s = transition(s, PlanStepAction.REQUEST_APPROVAL)
    s = transition(s, PlanStepAction.APPROVE)
    s = transition(s, PlanStepAction.START_RUNNING)
    s = transition(s, PlanStepAction.MARK_RETRYABLE)
    assert s.status is PlanStepStatus.RETRYABLE_FAILED
    s = transition(s, PlanStepAction.RETRY)
    s = transition(s, PlanStepAction.SUCCEED)
    assert s.status is PlanStepStatus.SUCCEEDED


def test_approval_decide() -> None:
    a = Approval(
        workspace_id="ws", task_id="t",
        target_type=ApprovalTargetType.PLAN, target_id="p", requester_id="u1",
    )
    a = transition(a, ApprovalAction.APPROVE, patch={"decided_at": datetime.now(UTC)})
    assert a.status is ApprovalStatus.APPROVED
    assert a.decided_at is not None


def test_artifact_lifecycle() -> None:
    art = Artifact(
        workspace_id="ws", task_id="t",
        type=ArtifactType.PPTX, title="deck",
    )
    art = transition(art, ArtifactAction.GENERATE)
    art = transition(art, ArtifactAction.UPLOAD)
    art = transition(art, ArtifactAction.SHARE)
    assert art.status is ArtifactStatus.SHARED


# ── 状态机：非法迁移 / 类型错误 ──────────────────────


def test_illegal_transition_raises() -> None:
    ws = _make_workspace()
    task = _make_task(ws)
    with pytest.raises(IllegalTransition) as exc:
        transition(task, TaskAction.SUCCEED)
    assert exc.value.entity_type == "Task"
    assert exc.value.current_status == TaskStatus.CREATED.value
    assert exc.value.action == TaskAction.SUCCEED.value


def test_action_type_mismatch_raises() -> None:
    task = _make_task(_make_workspace())
    with pytest.raises(TypeError):
        transition(task, ApprovalAction.APPROVE)  # type: ignore[arg-type]


def test_patch_cannot_override_controlled_fields() -> None:
    task = _make_task(_make_workspace())
    with pytest.raises(ValueError):
        transition(task, TaskAction.START_PLANNING, patch={"version": 99})


# ── Plan DAG 校验 ────────────────────────────────────


def _step(plan_id: str, sid: str, idx: int, deps: list[str]) -> PlanStep:
    return PlanStep(
        step_id=sid,
        workspace_id="ws", task_id="t", plan_id=plan_id,
        index=idx, title=sid, kind=PlanStepKind.READ_CONTEXT, skill_name="noop",
        depends_on=deps,
    )


def test_validate_plan_graph_ok() -> None:
    steps = [
        _step("p", "a", 0, []),
        _step("p", "b", 1, ["a"]),
        _step("p", "c", 2, ["a", "b"]),
    ]
    validate_plan_graph(steps)


def test_validate_plan_graph_empty_ok() -> None:
    validate_plan_graph([])


def test_validate_plan_graph_unknown_dep() -> None:
    steps = [_step("p", "a", 0, ["ghost"])]
    with pytest.raises(PlanGraphError):
        validate_plan_graph(steps)


def test_validate_plan_graph_self_dep() -> None:
    steps = [_step("p", "a", 0, ["a"])]
    with pytest.raises(PlanGraphError):
        validate_plan_graph(steps)


def test_validate_plan_graph_cycle() -> None:
    steps = [
        _step("p", "a", 0, ["b"]),
        _step("p", "b", 1, ["a"]),
    ]
    with pytest.raises(PlanGraphError):
        validate_plan_graph(steps)


def test_validate_plan_graph_duplicate_index() -> None:
    steps = [
        _step("p", "a", 0, []),
        _step("p", "b", 0, []),
    ]
    with pytest.raises(PlanGraphError):
        validate_plan_graph(steps)


def test_validate_plan_graph_cross_plan() -> None:
    steps = [
        _step("p1", "a", 0, []),
        _step("p2", "b", 1, []),
    ]
    with pytest.raises(PlanGraphError):
        validate_plan_graph(steps)
