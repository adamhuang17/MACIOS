"""Pilot M1 服务编排层测试。

覆盖：PlanningService 模板约束、ApprovalService 状态推进、
ExecutionEngine demo 路径与审批阻塞、TaskOrchestrator 幂等。
"""

from __future__ import annotations

import pytest

from agent_hub.pilot import (
    ApprovalService,
    ApprovalStatus,
    ArtifactStore,
    EventBus,
    ExecutionEngine,
    FakeModelGateway,
    InMemoryEventStore,
    PilotEventPublisher,
    PilotRepository,
    PlanContext,
    PlanningService,
    PlanStatus,
    PlanStepDraft,
    PlanStepKind,
    PlanStepStatus,
    PlanTemplateError,
    RiskLevel,
    SkillRegistry,
    Task,
    TaskOrchestrator,
    TaskRequest,
    TaskStatus,
    Workspace,
    WorkspaceStatus,
    register_fake_skills,
)
from agent_hub.pilot.services.dto import PlanBlueprint

# ── 通用 fixtures ────────────────────────────────────


@pytest.fixture()
def store() -> InMemoryEventStore:
    return InMemoryEventStore()


@pytest.fixture()
def repo(store: InMemoryEventStore) -> PilotRepository:
    return PilotRepository(store)


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def publisher(store: InMemoryEventStore, bus: EventBus) -> PilotEventPublisher:
    return PilotEventPublisher(store, bus)


@pytest.fixture()
def registry() -> SkillRegistry:
    r = SkillRegistry()
    register_fake_skills(r)
    return r


@pytest.fixture()
def gateway() -> FakeModelGateway:
    return FakeModelGateway()


@pytest.fixture()
def planning(
    gateway: FakeModelGateway,
    repo: PilotRepository,
    publisher: PilotEventPublisher,
) -> PlanningService:
    return PlanningService(gateway, repo, publisher)


@pytest.fixture()
def approvals(
    repo: PilotRepository, publisher: PilotEventPublisher,
) -> ApprovalService:
    return ApprovalService(repo, publisher)


@pytest.fixture()
def artifacts(
    repo: PilotRepository, publisher: PilotEventPublisher,
) -> ArtifactStore:
    return ArtifactStore(repo, publisher)


@pytest.fixture()
def execution(
    repo: PilotRepository,
    publisher: PilotEventPublisher,
    registry: SkillRegistry,
    artifacts: ArtifactStore,
    approvals: ApprovalService,
) -> ExecutionEngine:
    return ExecutionEngine(
        repository=repo,
        publisher=publisher,
        registry=registry,
        artifacts=artifacts,
        approvals=approvals,
        auto_approve_writes=True,
    )


@pytest.fixture()
def orchestrator(
    repo: PilotRepository,
    publisher: PilotEventPublisher,
    planning: PlanningService,
    approvals: ApprovalService,
    execution: ExecutionEngine,
) -> TaskOrchestrator:
    return TaskOrchestrator(
        repository=repo,
        publisher=publisher,
        planning=planning,
        approvals=approvals,
        execution=execution,
    )


def _make_task() -> tuple[Workspace, Task]:
    ws = Workspace(
        title="t", source_channel="api", created_by="u1",
        status=WorkspaceStatus.ACTIVE,
    )
    task = Task(
        workspace_id=ws.workspace_id,
        origin_text="please make a deck",
        requester_id="u1",
    )
    return ws, task


# ── PlanningService ─────────────────────────────────


@pytest.mark.asyncio
async def test_planning_full_template(
    planning: PlanningService, repo: PilotRepository,
) -> None:
    ws, task = _make_task()
    await repo.save(ws, expected_version=None)
    await repo.save(task, expected_version=None)

    plan, steps = await planning.generate(task, PlanContext(
        raw_text="hi", requester_id="u1", title="Demo",
    ))
    assert plan.status == PlanStatus.DRAFT
    assert {s.kind for s in steps} >= {
        PlanStepKind.READ_CONTEXT,
        PlanStepKind.GENERATE_DOC,
        PlanStepKind.GENERATE_SLIDE_SPEC,
        PlanStepKind.RENDER_SLIDES,
        PlanStepKind.UPLOAD,
        PlanStepKind.SUMMARIZE,
    }
    # write/share 自动 requires_approval
    write_or_share = [
        s for s in steps if s.risk_level in (RiskLevel.WRITE, RiskLevel.SHARE)
    ]
    assert write_or_share and all(s.requires_approval for s in write_or_share)
    # depends_on 已映射成真实 step_id
    real_ids = {s.step_id for s in steps}
    for s in steps:
        assert all(d in real_ids for d in s.depends_on)


@pytest.mark.asyncio
async def test_planning_template_missing_kind(
    repo: PilotRepository, publisher: PilotEventPublisher,
) -> None:
    class MissingGateway:
        async def complex_plan(self, ctx: PlanContext) -> PlanBlueprint:
            return PlanBlueprint(
                goal="g",
                steps=[
                    PlanStepDraft(
                        ref="s1", title="only-read", kind=PlanStepKind.READ_CONTEXT,
                        skill_name="fake_im_fetch",
                    ),
                ],
            )

    p = PlanningService(MissingGateway(), repo, publisher)  # type: ignore[arg-type]
    ws, task = _make_task()
    await repo.save(ws, expected_version=None)
    await repo.save(task, expected_version=None)
    with pytest.raises(PlanTemplateError):
        await p.generate(task, PlanContext(raw_text="x", requester_id="u1"))


# ── ApprovalService ─────────────────────────────────


@pytest.mark.asyncio
async def test_approval_flow_decide(
    planning: PlanningService,
    approvals: ApprovalService,
    repo: PilotRepository,
) -> None:
    ws, task = _make_task()
    await repo.save(ws, expected_version=None)
    await repo.save(task, expected_version=None)
    plan, _ = await planning.generate(task, PlanContext(
        raw_text="hi", requester_id="u1", title="Demo",
    ))

    approval, plan = await approvals.request_plan_approval(
        plan, task, requester_id="u1",
    )
    assert approval.status == ApprovalStatus.REQUESTED
    assert plan.status == PlanStatus.APPROVAL_REQUESTED

    decided, plan = await approvals.decide_plan(
        approval, plan, task,
        decision="approve", approver_id="u2",
    )
    assert decided.status == ApprovalStatus.APPROVED
    assert plan.status == PlanStatus.APPROVED
    assert decided.decided_at is not None
    assert decided.decision_records[-1].approver_id == "u2"


@pytest.mark.asyncio
async def test_approval_supersede_for_new_plan(
    planning: PlanningService,
    approvals: ApprovalService,
    repo: PilotRepository,
) -> None:
    ws, task = _make_task()
    await repo.save(ws, expected_version=None)
    await repo.save(task, expected_version=None)
    plan, _ = await planning.generate(task, PlanContext(
        raw_text="hi", requester_id="u1", title="Demo",
    ))
    approval, plan = await approvals.request_plan_approval(
        plan, task, requester_id="u1",
    )

    superseded = await approvals.supersede_for_new_plan(plan, task)
    assert superseded.status == PlanStatus.SUPERSEDED
    refreshed = await repo.get_approval(approval.approval_id)
    assert refreshed is not None
    assert refreshed.status == ApprovalStatus.SUPERSEDED


# ── ExecutionEngine ─────────────────────────────────


@pytest.mark.asyncio
async def test_execution_demo_full_run(
    planning: PlanningService,
    approvals: ApprovalService,
    execution: ExecutionEngine,
    repo: PilotRepository,
) -> None:
    ws, task = _make_task()
    await repo.save(ws, expected_version=None)
    await repo.save(task, expected_version=None)
    plan, _ = await planning.generate(task, PlanContext(
        raw_text="hi", requester_id="u1", title="Demo",
    ))
    _, plan = await approvals.auto_approve_plan(plan, task, requester_id="u1")

    # Task: PLANNING → AWAITING_APPROVAL → APPROVED
    from agent_hub.pilot import TaskAction, transition
    task = transition(task, TaskAction.START_PLANNING)
    task = transition(task, TaskAction.REQUEST_APPROVAL)
    task = transition(task, TaskAction.APPROVE)
    await repo.save(task, expected_version=None)

    result = await execution.run_plan(task, plan)
    assert result.final_status == TaskStatus.SUCCEEDED
    assert all(e.final_status == PlanStepStatus.SUCCEEDED for e in result.executions)
    assert len(result.artifacts) >= 6


@pytest.mark.asyncio
async def test_execution_blocks_when_no_auto_approve(
    planning: PlanningService,
    approvals: ApprovalService,
    repo: PilotRepository,
    publisher: PilotEventPublisher,
    registry: SkillRegistry,
    artifacts: ArtifactStore,
) -> None:
    engine = ExecutionEngine(
        repository=repo,
        publisher=publisher,
        registry=registry,
        artifacts=artifacts,
        approvals=approvals,
        auto_approve_writes=False,
    )
    ws, task = _make_task()
    await repo.save(ws, expected_version=None)
    await repo.save(task, expected_version=None)
    plan, _ = await planning.generate(task, PlanContext(
        raw_text="hi", requester_id="u1", title="Demo",
    ))
    _, plan = await approvals.auto_approve_plan(plan, task, requester_id="u1")

    from agent_hub.pilot import TaskAction, transition
    task = transition(task, TaskAction.START_PLANNING)
    task = transition(task, TaskAction.REQUEST_APPROVAL)
    task = transition(task, TaskAction.APPROVE)
    await repo.save(task, expected_version=None)

    result = await engine.run_plan(task, plan)
    assert result.final_status == TaskStatus.BLOCKED
    statuses = {e.final_status for e in result.executions}
    assert PlanStepStatus.WAITING_APPROVAL in statuses


# ── TaskOrchestrator ────────────────────────────────


@pytest.mark.asyncio
async def test_orchestrator_auto_approve_runs_to_succeeded(
    orchestrator: TaskOrchestrator, repo: PilotRepository,
) -> None:
    handle = await orchestrator.submit(TaskRequest(
        source_channel="api",
        raw_text="generate the demo deck",
        requester_id="u1",
        auto_approve=True,
        idempotency_key="run-1",
    ))
    assert handle.status == TaskStatus.SUCCEEDED
    task = await repo.get_task(handle.task_id)
    assert task is not None
    assert task.plan_id == handle.plan_id


@pytest.mark.asyncio
async def test_orchestrator_idempotency_returns_same_task(
    orchestrator: TaskOrchestrator,
) -> None:
    request = TaskRequest(
        source_channel="api",
        raw_text="x",
        requester_id="u1",
        workspace_id="ws_fixed_001",
        idempotency_key="dup-key",
        auto_approve=True,
    )
    h1 = await orchestrator.submit(request)
    h2 = await orchestrator.submit(request)
    assert h2.deduplicated is True
    assert h1.task_id == h2.task_id


@pytest.mark.asyncio
async def test_orchestrator_pending_approval_path(
    orchestrator: TaskOrchestrator, repo: PilotRepository,
) -> None:
    handle = await orchestrator.submit(TaskRequest(
        source_channel="api",
        raw_text="needs approval",
        requester_id="u1",
        auto_approve=False,
        idempotency_key="pending-1",
    ))
    assert handle.status == TaskStatus.AWAITING_APPROVAL
    task = await repo.get_task(handle.task_id)
    assert task is not None
    assert task.status == TaskStatus.AWAITING_APPROVAL
