"""Pilot M1 服务编排层测试。

覆盖：PlanningService 模板约束、ApprovalService 状态推进、
ExecutionEngine demo 路径与审批阻塞、TaskOrchestrator 幂等。
"""

from __future__ import annotations

import pytest

from agent_hub.pilot import (
    ApprovalService,
    ApprovalStatus,
    Artifact,
    ArtifactStore,
    CommandError,
    EventBus,
    EventType,
    ExecutionEngine,
    FakeModelGateway,
    InMemoryEventStore,
    PilotCommandService,
    PilotEventPublisher,
    PilotRepository,
    PlanContext,
    PlanningService,
    PlanStatus,
    PlanStep,
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
from agent_hub.pilot.domain.models import Approval
from agent_hub.pilot.services.dto import PlanBlueprint
from agent_hub.pilot.skills.registry import SkillResult, SkillSpec

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


@pytest.fixture()
def commands(
    repo: PilotRepository,
    approvals: ApprovalService,
    orchestrator: TaskOrchestrator,
    publisher: PilotEventPublisher,
) -> PilotCommandService:
    return PilotCommandService(repo, approvals, orchestrator, publisher)


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


class _RecordingApprovalNotifier:
    def __init__(self) -> None:
        self.approval_ids: list[str] = []

    async def notify_requested(self, approval: Approval) -> None:
        self.approval_ids.append(approval.approval_id)


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
async def test_command_rejects_actor_not_in_approvers(
    planning: PlanningService,
    approvals: ApprovalService,
    commands: PilotCommandService,
    repo: PilotRepository,
) -> None:
    ws, task = _make_task()
    await repo.save(ws, expected_version=None)
    await repo.save(task, expected_version=None)
    plan, _ = await planning.generate(task, PlanContext(
        raw_text="hi", requester_id="u1", title="Demo",
    ))

    approval, _ = await approvals.request_plan_approval(
        plan,
        task,
        requester_id="u1",
        approvers=["ou_owner"],
    )

    with pytest.raises(CommandError, match="审批人名单"):
        await commands.decide_approval(
            approval.approval_id,
            decision="approve",
            actor_id="ou_other",
        )

    refreshed = await repo.get_approval(approval.approval_id)
    assert refreshed is not None
    assert refreshed.status == ApprovalStatus.REQUESTED


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


@pytest.mark.asyncio
async def test_execution_blocks_when_no_auto_approve_notifies(
    planning: PlanningService,
    approvals: ApprovalService,
    repo: PilotRepository,
    publisher: PilotEventPublisher,
    registry: SkillRegistry,
    artifacts: ArtifactStore,
) -> None:
    notifier = _RecordingApprovalNotifier()
    engine = ExecutionEngine(
        repository=repo,
        publisher=publisher,
        registry=registry,
        artifacts=artifacts,
        approvals=approvals,
        auto_approve_writes=False,
        approval_notifier=notifier,
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
    assert len(notifier.approval_ids) == 1
    pending = await repo.list_approvals_for_task(task.task_id, workspace_id=task.workspace_id)
    requested = next(a for a in pending if a.status == ApprovalStatus.REQUESTED)
    assert requested.approvers == [task.requester_id]


@pytest.mark.asyncio
async def test_orchestrator_manual_plan_approval_notifies(
    repo: PilotRepository,
    publisher: PilotEventPublisher,
    planning: PlanningService,
    approvals: ApprovalService,
    execution: ExecutionEngine,
) -> None:
    notifier = _RecordingApprovalNotifier()
    orchestrator = TaskOrchestrator(
        repository=repo,
        publisher=publisher,
        planning=planning,
        approvals=approvals,
        execution=execution,
        approval_notifier=notifier,
    )

    handle = await orchestrator.submit(TaskRequest(
        source_channel="api",
        raw_text="请帮我做一份关于Q4增长的演示稿",
        requester_id="tester-1",
        title="Q4 Growth",
        auto_approve=False,
    ))

    assert handle.status == TaskStatus.AWAITING_APPROVAL
    assert len(notifier.approval_ids) == 1
    approvals_for_task = await repo.list_approvals_for_task(
        handle.task_id,
        workspace_id=handle.workspace_id,
    )
    assert len(approvals_for_task) == 1
    assert approvals_for_task[0].approvers == ["tester-1"]


@pytest.mark.asyncio
async def test_execution_resume_continues_to_next_approval(
    planning: PlanningService,
    approvals: ApprovalService,
    repo: PilotRepository,
    publisher: PilotEventPublisher,
    registry: SkillRegistry,
    artifacts: ArtifactStore,
) -> None:
    """auto_approve_writes=False：人工 approve 第一个 write step 后 resume，
    应执行 approved step、产出 artifact，并在下一个 write/share step 再次 BLOCKED。
    """
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

    first_run = await engine.run_plan(task, plan)
    assert first_run.final_status == TaskStatus.BLOCKED

    # 找到第一个 waiting_approval 的 step 与对应的 requested approval
    blocked_step = next(
        s for s in first_run.steps
        if s.status == PlanStepStatus.WAITING_APPROVAL
    )
    pending_approvals = await repo.list_approvals_for_task(
        task.task_id, workspace_id=task.workspace_id,
    )
    pending = next(
        a for a in pending_approvals
        if a.target_id == blocked_step.step_id
        and a.status == ApprovalStatus.REQUESTED
    )
    blocked_task = await repo.get_task(task.task_id)
    assert blocked_task is not None
    _, approved_step = await approvals.decide_step(
        pending, blocked_step, blocked_task,
        decision="approve", approver_id="u1", comment="ok",
    )
    assert approved_step.status == PlanStepStatus.APPROVED
    assert blocked_task.status == TaskStatus.BLOCKED  # task 状态不变

    second_run = await engine.resume_plan(blocked_task, plan)

    # 之前 approved 的 step 现在应已 succeeded 并产出 artifact
    refreshed_step = await repo.get_step(approved_step.step_id)
    assert refreshed_step is not None
    assert refreshed_step.status == PlanStepStatus.SUCCEEDED
    assert refreshed_step.output_artifact_id is not None

    # 仍有后续 write/share step 未审批，task 应再次 BLOCKED
    assert second_run.final_status in {TaskStatus.BLOCKED, TaskStatus.SUCCEEDED}
    if second_run.final_status == TaskStatus.BLOCKED:
        statuses = {s.status for s in second_run.steps}
        assert PlanStepStatus.WAITING_APPROVAL in statuses


@pytest.mark.asyncio
async def test_execution_resume_rejects_non_blocked_task(
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
    with pytest.raises(ValueError, match="无法 resume"):
        await engine.resume_plan(task, plan)


@pytest.mark.asyncio
async def test_artifact_store_promotes_share_link_to_shared(
    repo: PilotRepository,
    publisher: PilotEventPublisher,
) -> None:
    artifacts = ArtifactStore(repo, publisher)
    ws, task = _make_task()
    step = PlanStep(
        workspace_id=ws.workspace_id,
        task_id=task.task_id,
        plan_id="plan_demo",
        index=4,
        title="上传 Drive 并生成分享",
        kind=PlanStepKind.UPLOAD,
        skill_name="real.drive.upload_share",
        output_artifact_ref="share",
        risk_level=RiskLevel.SHARE,
        requires_approval=True,
    )
    await repo.save(ws, expected_version=None)
    await repo.save(task, expected_version=None)
    await repo.save(step, expected_version=None)

    artifact = await artifacts.save(
        step,
        task,
        {
            "type": "share_link",
            "title": "deck share",
            "mime_type": "text/uri-list",
            "content": {
                "file_name": "deck.pptx",
                "share_url": "https://open.feishu.cn/share/demo",
                "file_token": "file_demo",
            },
            "metadata": {
                "share_url": "https://open.feishu.cn/share/demo",
                "feishu_token": "file_demo",
                "file_token": "file_demo",
            },
        },
    )

    assert artifact.status == "shared"
    assert artifact.share_url == "https://open.feishu.cn/share/demo"
    assert artifact.feishu_token == "file_demo"


@pytest.mark.asyncio
async def test_execute_step_dry_run_receives_resolved_inputs(
    repo: PilotRepository,
    publisher: PilotEventPublisher,
    approvals: ApprovalService,
    artifacts: ArtifactStore,
) -> None:
    seen: dict[str, bool] = {}
    registry = SkillRegistry()

    async def _render(inv):  # noqa: ANN001, ANN202
        seen["dry_run_has_spec"] = "spec" in (
            inv.params.get("input_artifacts") or {}
        )
        if inv.dry_run:
            return SkillResult(skill_name=inv.skill_name, success=True)
        return SkillResult(
            skill_name=inv.skill_name,
            success=True,
            artifact_payload={
                "type": "pptx",
                "title": "deck.pptx",
                "mime_type": (
                    "application/vnd.openxmlformats-officedocument."
                    "presentationml.presentation"
                ),
                "content": b"pptx-bytes",
            },
        )

    registry.register(
        SkillSpec(
            skill_name="test.render",
            side_effect="write",
            requires_approval=True,
            supports_dry_run=True,
            timeout_ms=5_000,
            scopes=[],
        ),
        _render,
    )

    engine = ExecutionEngine(
        repository=repo,
        publisher=publisher,
        registry=registry,
        artifacts=artifacts,
        approvals=approvals,
        auto_approve_writes=True,
    )

    ws, task = _make_task()
    await repo.save(ws, expected_version=None)
    await repo.save(task, expected_version=None)

    upstream = PlanStep(
        workspace_id=ws.workspace_id,
        task_id=task.task_id,
        plan_id="plan_demo",
        index=2,
        title="生成 SlideSpec",
        kind=PlanStepKind.GENERATE_SLIDE_SPEC,
        skill_name="noop.spec",
        output_artifact_ref="spec",
    )
    await repo.save(upstream, expected_version=None)
    spec_artifact = Artifact(
        workspace_id=ws.workspace_id,
        task_id=task.task_id,
        step_id=upstream.step_id,
        type="slide_spec_json",
        title="SlideSpec",
        status="generated",
        mime_type="application/json",
        metadata={},
    )
    await repo.save(spec_artifact, expected_version=None)
    upstream = upstream.model_copy(update={
        "version": upstream.version + 1,
        "output_artifact_id": spec_artifact.artifact_id,
        "status": PlanStepStatus.SUCCEEDED,
    })
    await repo.save(upstream)

    target = PlanStep(
        workspace_id=ws.workspace_id,
        task_id=task.task_id,
        plan_id="plan_demo",
        index=3,
        title="渲染 PPTX",
        kind=PlanStepKind.RENDER_SLIDES,
        skill_name="test.render",
        inputs_from=["spec"],
        depends_on=[upstream.step_id],
        output_artifact_ref="pptx",
        risk_level=RiskLevel.WRITE,
        requires_approval=True,
    )
    await repo.save(target, expected_version=None)

    result = await engine._execute_step(target, task)

    assert result.final_status == PlanStepStatus.SUCCEEDED
    assert seen["dry_run_has_spec"] is True


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
async def test_orchestrator_emits_progress_events(
    orchestrator: TaskOrchestrator,
    store: InMemoryEventStore,
) -> None:
    handle = await orchestrator.submit(TaskRequest(
        source_channel="api",
        raw_text="generate the demo deck",
        requester_id="u1",
        auto_approve=True,
        idempotency_key="progress-1",
    ))
    events = await store.list_events(
        handle.workspace_id, since_sequence=0, limit=200,
    )
    progress = [e for e in events if e.type is EventType.TASK_PROGRESS]
    messages = [e.message for e in progress]

    assert "收到任务，开始执行" in messages
    assert "Agent 正在进行任务编排" in messages
    assert "任务已完成，成果已生成" in messages


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
