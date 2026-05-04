"""Pilot M3 文档生成链测试。

覆盖：
- ArtifactStore 内容落盘 / 内存兜底 / read_content 往返；
- StepInputResolver 把上游 Artifact 注入下游 step；
- internal.context.assemble + internal.brief.generate 端到端；
- Brief v2 修订：parent SUPERSEDED + artifact_version=2 + parent_artifact_id 链。
"""

from __future__ import annotations

import pytest

from agent_hub.pilot import (
    ApprovalService,
    ArtifactStatus,
    ArtifactStore,
    ArtifactType,
    BriefRevisionContext,
    EventBus,
    ExecutionEngine,
    FakeModelGateway,
    InMemoryEventStore,
    PilotEventPublisher,
    PilotRepository,
    PlanningService,
    PlanStatus,
    PlanStepKind,
    PlanStepStatus,
    SkillRegistry,
    StepInputResolver,
    Task,
    TemplateBriefGenerator,
    Workspace,
    WorkspaceStatus,
    register_internal_document_skills,
    transition,
)
from agent_hub.pilot.domain.enums import TaskAction
from agent_hub.pilot.domain.models import PlanStep
from agent_hub.pilot.skills.internal import BRIEF_SECTIONS


def _drive_task_to_running(task: Task) -> Task:
    t = transition(task, TaskAction.START_PLANNING)
    t = transition(t, TaskAction.REQUEST_APPROVAL)
    t = transition(t, TaskAction.APPROVE)
    return transition(t, TaskAction.START_RUNNING)


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
def publisher(
    store: InMemoryEventStore, bus: EventBus,
) -> PilotEventPublisher:
    return PilotEventPublisher(store, bus)


@pytest.fixture()
def artifacts_mem(
    repo: PilotRepository, publisher: PilotEventPublisher,
) -> ArtifactStore:
    return ArtifactStore(repo, publisher)


@pytest.fixture()
def artifacts_disk(
    repo: PilotRepository,
    publisher: PilotEventPublisher,
    tmp_path,
) -> ArtifactStore:
    return ArtifactStore(repo, publisher, artifact_dir=str(tmp_path / "a"))


def _ws_task() -> tuple[Workspace, Task]:
    ws = Workspace(
        title="m3", source_channel="api", created_by="u1",
        status=WorkspaceStatus.ACTIVE,
    )
    task = Task(
        workspace_id=ws.workspace_id,
        origin_text="给我做一份 Q2 复盘 brief",
        requester_id="u1",
    )
    return ws, task


def _step_for(
    ws: Workspace, task: Task, kind: PlanStepKind, skill: str,
) -> PlanStep:
    return PlanStep(
        workspace_id=ws.workspace_id,
        task_id=task.task_id,
        plan_id="plan-stub",
        index=0,
        title="t",
        kind=kind,
        skill_name=skill,
        status=PlanStepStatus.PENDING,
    )


def _make_internal_registry(
    artifacts: ArtifactStore,
) -> tuple[SkillRegistry, TemplateBriefGenerator]:
    registry = SkillRegistry()
    generator = TemplateBriefGenerator()
    register_internal_document_skills(
        registry,
        brief_generator=generator,
        artifact_reader=artifacts.read_content_by_id,
    )
    return registry, generator


# ── ArtifactStore ────────────────────────────────


@pytest.mark.asyncio
async def test_artifact_store_memory_roundtrip(
    repo: PilotRepository,
    artifacts_mem: ArtifactStore,
) -> None:
    ws, task = _ws_task()
    await repo.save(ws, expected_version=None)
    await repo.save(task, expected_version=None)
    step = _step_for(
        ws, task, PlanStepKind.GENERATE_DOC, "internal.brief.generate",
    )
    await repo.save(step, expected_version=None)

    artifact = await artifacts_mem.save(step, task, {
        "type": ArtifactType.PROJECT_BRIEF_MD.value,
        "title": "Brief v1",
        "mime_type": "text/markdown",
        "content": "# hello\n",
        "metadata": {"language": "zh-CN"},
    })

    assert artifact.status == ArtifactStatus.GENERATED
    assert artifact.artifact_version == 1
    assert artifact.checksum and len(artifact.checksum) == 64
    assert (artifact.storage_key or "").startswith("memory://")
    body = await artifacts_mem.read_content(artifact)
    assert body == "# hello\n"


@pytest.mark.asyncio
async def test_artifact_store_disk_roundtrip(
    repo: PilotRepository,
    artifacts_disk: ArtifactStore,
) -> None:
    ws, task = _ws_task()
    await repo.save(ws, expected_version=None)
    await repo.save(task, expected_version=None)
    step = _step_for(
        ws, task, PlanStepKind.READ_CONTEXT, "internal.context.assemble",
    )
    await repo.save(step, expected_version=None)

    payload = {"k": "v", "n": 1}
    artifact = await artifacts_disk.save(step, task, {
        "type": ArtifactType.CONTEXT_BUNDLE.value,
        "title": "ctx",
        "mime_type": "application/json",
        "content": payload,
    })
    assert artifact.uri and artifact.uri.startswith("file://")
    assert artifact.checksum
    body = await artifacts_disk.read_content(artifact)
    assert body == payload


@pytest.mark.asyncio
async def test_artifact_supersede_chain(
    repo: PilotRepository,
    artifacts_mem: ArtifactStore,
) -> None:
    ws, task = _ws_task()
    await repo.save(ws, expected_version=None)
    await repo.save(task, expected_version=None)
    step = _step_for(
        ws, task, PlanStepKind.GENERATE_DOC, "internal.brief.generate",
    )
    await repo.save(step, expected_version=None)

    v1 = await artifacts_mem.save(step, task, {
        "type": ArtifactType.PROJECT_BRIEF_MD.value,
        "title": "Brief v1",
        "mime_type": "text/markdown",
        "content": "# v1\n",
    })
    v2 = await artifacts_mem.save(step, task, {
        "type": ArtifactType.PROJECT_BRIEF_MD.value,
        "title": "Brief v2",
        "mime_type": "text/markdown",
        "content": "# v2\n",
        "parent_artifact_id": v1.artifact_id,
    })
    parent_after = await repo.get_artifact(v1.artifact_id)
    assert v2.artifact_version == 2
    assert v2.parent_artifact_id == v1.artifact_id
    assert parent_after is not None
    assert parent_after.status == ArtifactStatus.SUPERSEDED


# ── StepInputResolver ───────────────────────────


@pytest.mark.asyncio
async def test_step_input_resolver_returns_upstream_content(
    repo: PilotRepository,
    artifacts_mem: ArtifactStore,
) -> None:
    ws, task = _ws_task()
    await repo.save(ws, expected_version=None)
    await repo.save(task, expected_version=None)

    upstream = PlanStep(
        workspace_id=ws.workspace_id,
        task_id=task.task_id,
        plan_id="plan-x",
        index=0,
        title="ctx",
        kind=PlanStepKind.READ_CONTEXT,
        skill_name="internal.context.assemble",
        output_artifact_ref="ctx",
        status=PlanStepStatus.SUCCEEDED,
    )
    await repo.save(upstream, expected_version=None)
    artifact = await artifacts_mem.save(upstream, task, {
        "type": ArtifactType.CONTEXT_BUNDLE.value,
        "title": "ctx",
        "mime_type": "application/json",
        "content": {"origin_text": "hello"},
    })
    upstream = upstream.model_copy(update={
        "output_artifact_id": artifact.artifact_id,
    })
    await repo.save(upstream, expected_version=None)

    downstream = PlanStep(
        workspace_id=ws.workspace_id,
        task_id=task.task_id,
        plan_id="plan-x",
        index=1,
        title="brief",
        kind=PlanStepKind.GENERATE_DOC,
        skill_name="internal.brief.generate",
        inputs_from=["ctx"],
        depends_on=[upstream.step_id],
        status=PlanStepStatus.PENDING,
    )
    await repo.save(downstream, expected_version=None)

    resolver = StepInputResolver(repo, artifacts_mem)
    resolved, missing = await resolver.resolve(downstream)
    assert missing == []
    assert "ctx" in resolved
    assert resolved["ctx"]["artifact_id"] == artifact.artifact_id
    assert resolved["ctx"]["content"] == {"origin_text": "hello"}


@pytest.mark.asyncio
async def test_step_input_resolver_reports_missing(
    repo: PilotRepository,
    artifacts_mem: ArtifactStore,
) -> None:
    ws, task = _ws_task()
    await repo.save(ws, expected_version=None)
    await repo.save(task, expected_version=None)

    downstream = PlanStep(
        workspace_id=ws.workspace_id,
        task_id=task.task_id,
        plan_id="plan-y",
        index=0,
        title="x",
        kind=PlanStepKind.GENERATE_DOC,
        skill_name="internal.brief.generate",
        inputs_from=["ctx"],
        status=PlanStepStatus.PENDING,
    )
    await repo.save(downstream, expected_version=None)

    resolver = StepInputResolver(repo, artifacts_mem)
    resolved, missing = await resolver.resolve(downstream)
    assert resolved == {}
    assert missing == ["ctx"]


# ── 端到端：context + brief generate ─────────────


@pytest.mark.asyncio
async def test_internal_brief_generate_end_to_end(
    repo: PilotRepository,
    publisher: PilotEventPublisher,
    artifacts_mem: ArtifactStore,
) -> None:
    registry, _ = _make_internal_registry(artifacts_mem)
    approvals = ApprovalService(repo, publisher)
    engine = ExecutionEngine(
        repository=repo,
        publisher=publisher,
        registry=registry,
        artifacts=artifacts_mem,
        approvals=approvals,
    )

    ws, task = _ws_task()
    await repo.save(ws, expected_version=None)
    await repo.save(task, expected_version=None)
    running_task = _drive_task_to_running(task)
    await repo.save(running_task, expected_version=None)

    plan_id = "plan-internal"
    step_ctx = PlanStep(
        workspace_id=ws.workspace_id,
        task_id=task.task_id,
        plan_id=plan_id,
        index=0,
        title="ctx",
        kind=PlanStepKind.READ_CONTEXT,
        skill_name="internal.context.assemble",
        input_params={
            "origin_text": task.origin_text,
            "title": "ctx-bundle",
        },
        output_artifact_ref="ctx",
        status=PlanStepStatus.PENDING,
    )
    step_brief = PlanStep(
        workspace_id=ws.workspace_id,
        task_id=task.task_id,
        plan_id=plan_id,
        index=1,
        title="brief",
        kind=PlanStepKind.GENERATE_DOC,
        skill_name="internal.brief.generate",
        input_params={"title": "Q2 复盘"},
        inputs_from=["ctx"],
        output_artifact_ref="brief",
        depends_on=[step_ctx.step_id],
        status=PlanStepStatus.PENDING,
    )
    await repo.save(step_ctx, expected_version=None)
    await repo.save(step_brief, expected_version=None)

    exec_ctx = await engine._execute_step(  # noqa: SLF001
        step_ctx, running_task,
    )
    assert exec_ctx.final_status == PlanStepStatus.SUCCEEDED
    assert exec_ctx.artifact_id

    refreshed = await repo.get_step(step_brief.step_id)
    assert refreshed is not None
    exec_brief = await engine._execute_step(  # noqa: SLF001
        refreshed, running_task,
    )
    assert exec_brief.final_status == PlanStepStatus.SUCCEEDED
    brief_art = await repo.get_artifact(exec_brief.artifact_id or "")
    assert brief_art is not None
    md = await artifacts_mem.read_content(brief_art)
    assert isinstance(md, str)
    for section in BRIEF_SECTIONS:
        assert f"## {section}" in md


@pytest.mark.asyncio
async def test_brief_revision_chain(
    repo: PilotRepository,
    publisher: PilotEventPublisher,
    artifacts_mem: ArtifactStore,
) -> None:
    registry, _ = _make_internal_registry(artifacts_mem)
    approvals = ApprovalService(repo, publisher)
    engine = ExecutionEngine(
        repository=repo,
        publisher=publisher,
        registry=registry,
        artifacts=artifacts_mem,
        approvals=approvals,
    )

    ws, task = _ws_task()
    await repo.save(ws, expected_version=None)
    await repo.save(task, expected_version=None)
    running_task = _drive_task_to_running(task)
    await repo.save(running_task, expected_version=None)

    seed_step = _step_for(
        ws, task, PlanStepKind.GENERATE_DOC, "internal.brief.generate",
    )
    await repo.save(seed_step, expected_version=None)
    v1 = await artifacts_mem.save(seed_step, task, {
        "type": ArtifactType.PROJECT_BRIEF_MD.value,
        "title": "Brief v1",
        "mime_type": "text/markdown",
        "content": "# Brief v1\n## 背景\n初版背景\n",
    })

    planning = PlanningService(FakeModelGateway(), repo, publisher)
    plan, steps = await planning.generate_revision(
        running_task,
        BriefRevisionContext(
            base_brief_artifact_id=v1.artifact_id,
            revision_instruction="补充 Q2 渠道数据",
            title="Q2 复盘",
        ),
    )
    assert plan.status == PlanStatus.DRAFT
    assert {s.kind for s in steps} == {
        PlanStepKind.READ_CONTEXT, PlanStepKind.GENERATE_DOC,
    }

    ctx_step = next(s for s in steps if s.kind == PlanStepKind.READ_CONTEXT)
    revise_step = next(
        s for s in steps if s.kind == PlanStepKind.GENERATE_DOC
    )
    assert revise_step.skill_name == "internal.brief.revise"
    assert revise_step.input_params["base_brief_artifact_id"] == v1.artifact_id

    exec_ctx = await engine._execute_step(  # noqa: SLF001
        ctx_step, running_task,
    )
    assert exec_ctx.final_status == PlanStepStatus.SUCCEEDED

    refreshed = await repo.get_step(revise_step.step_id)
    assert refreshed is not None
    exec_rev = await engine._execute_step(  # noqa: SLF001
        refreshed, running_task,
    )
    assert exec_rev.final_status == PlanStepStatus.SUCCEEDED

    v2 = await repo.get_artifact(exec_rev.artifact_id or "")
    parent_after = await repo.get_artifact(v1.artifact_id)
    assert v2 is not None and parent_after is not None
    assert v2.artifact_version == 2
    assert v2.parent_artifact_id == v1.artifact_id
    assert parent_after.status == ArtifactStatus.SUPERSEDED

    md = await artifacts_mem.read_content(v2)
    assert isinstance(md, str)
    assert "## 修订记录" in md
    assert "补充 Q2 渠道数据" in md


@pytest.mark.asyncio
async def test_execution_fails_when_inputs_missing(
    repo: PilotRepository,
    publisher: PilotEventPublisher,
    artifacts_mem: ArtifactStore,
) -> None:
    registry, _ = _make_internal_registry(artifacts_mem)
    approvals = ApprovalService(repo, publisher)
    engine = ExecutionEngine(
        repository=repo,
        publisher=publisher,
        registry=registry,
        artifacts=artifacts_mem,
        approvals=approvals,
    )

    ws, task = _ws_task()
    await repo.save(ws, expected_version=None)
    await repo.save(task, expected_version=None)
    running_task = _drive_task_to_running(task)
    await repo.save(running_task, expected_version=None)

    orphan = PlanStep(
        workspace_id=ws.workspace_id,
        task_id=task.task_id,
        plan_id="plan-orphan",
        index=0,
        title="brief",
        kind=PlanStepKind.GENERATE_DOC,
        skill_name="internal.brief.generate",
        inputs_from=["ctx"],
        status=PlanStepStatus.PENDING,
    )
    await repo.save(orphan, expected_version=None)
    result = await engine._execute_step(  # noqa: SLF001
        orphan, running_task,
    )
    assert result.final_status == PlanStepStatus.FAILED
    assert "ctx" in (result.error or "")
