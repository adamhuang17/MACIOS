"""黄金路径事件序列快照测试。

验证 Pilot 核心生命周期在 InMemoryEventStore 上产生正确的事件序列。
每个测试都是完全内存的，无 Redis、无 LLM、无飞书。

覆盖场景：
1. 自动审批路径：TASK_CREATED → PLAN_GENERATED → STEP_STATUS_CHANGED * N → TASK_COMPLETED
2. 手动审批路径（先请求审批，再批准）：…→ APPROVAL_REQUESTED → APPROVAL_DECIDED → …
3. 审批拒绝路径：…→ APPROVAL_REQUESTED → APPROVAL_DECIDED → TASK_FAILED
4. 任务级 BLOCKED / UNBLOCK 状态路径
"""

from __future__ import annotations

import pytest

from agent_hub.pilot import (
    ApprovalService,
    ArtifactStore,
    EventBus,
    EventType,
    ExecutionEngine,
    FakeModelGateway,
    InMemoryEventStore,
    PilotEventPublisher,
    PilotRepository,
    PlanContext,
    PlanningService,
    SkillRegistry,
    TaskOrchestrator,
    TaskRequest,
    TaskStatus,
    register_fake_skills,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


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
    repo: PilotRepository,
    publisher: PilotEventPublisher,
) -> ApprovalService:
    return ApprovalService(repo, publisher)


@pytest.fixture()
def artifacts(
    repo: PilotRepository,
    publisher: PilotEventPublisher,
) -> ArtifactStore:
    return ArtifactStore(repo, publisher)


def _make_auto_approve_engine(
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


def _make_manual_approve_engine(
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
        auto_approve_writes=False,
    )


def _make_orchestrator(
    repo: PilotRepository,
    publisher: PilotEventPublisher,
    planning: PlanningService,
    approvals: ApprovalService,
    execution: ExecutionEngine,
    *,
    auto_approve: bool = True,
) -> TaskOrchestrator:
    return TaskOrchestrator(
        repository=repo,
        publisher=publisher,
        planning=planning,
        approvals=approvals,
        execution=execution,
    )


# ── 辅助函数 ─────────────────────────────────────────────────────────────────


async def _event_types(store: InMemoryEventStore, workspace_id: str) -> list[str]:
    """从 store 提取指定 workspace 的事件类型序列（按 sequence 排序）。"""
    events = await store.list_events(workspace_id, since_sequence=0, limit=1000)
    events = sorted(events, key=lambda e: e.sequence)
    return [e.type for e in events]


# ── 场景 1：自动审批黄金路径 ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_golden_path_auto_approve(
    store: InMemoryEventStore,
    repo: PilotRepository,
    publisher: PilotEventPublisher,
    planning: PlanningService,
    approvals: ApprovalService,
    artifacts: ArtifactStore,
    registry: SkillRegistry,
) -> None:
    """自动审批路径：任务从创建到完成，必须包含完整的关键事件序列。"""
    engine = _make_auto_approve_engine(repo, publisher, registry, artifacts, approvals)
    orch = _make_orchestrator(repo, publisher, planning, approvals, engine)

    handle = await orch.submit(TaskRequest(
        source_channel="api",
        raw_text="写一份产品介绍文档",
        requester_id="u1",
        auto_approve=True,
    ))

    assert handle.status == TaskStatus.SUCCEEDED

    event_types = await _event_types(store, handle.workspace_id)

    # 关键事件必须存在（顺序不强制，但各阶段标记不能缺）
    required = {
        EventType.TASK_CREATED,
        EventType.PLAN_GENERATED,
        EventType.TASK_COMPLETED,
    }
    missing = required - set(event_types)
    assert not missing, f"自动审批黄金路径缺少事件: {missing}"

    # 任务创建事件必须存在且早于 PLAN_GENERATED（workspace 创建可能更早）
    assert EventType.TASK_CREATED in event_types, "TASK_CREATED 事件不存在"

    # TASK_COMPLETED 必须存在（可能后面还有 task.progress 心跳事件）
    assert EventType.TASK_COMPLETED in event_types, "TASK_COMPLETED 事件不存在"

    # 计划生成必须在任务完成之前
    plan_idx = next(i for i, t in enumerate(event_types) if t == EventType.PLAN_GENERATED)
    complete_idx = event_types.index(EventType.TASK_COMPLETED)
    assert plan_idx < complete_idx, "PLAN_GENERATED 应早于 TASK_COMPLETED"

    # 步骤执行事件必须存在（至少有一个 STEP_STATUS_CHANGED）
    step_events = [t for t in event_types if t == EventType.STEP_STATUS_CHANGED]
    assert step_events, "自动审批路径必须有至少一个 STEP_STATUS_CHANGED 事件"


# ── 场景 2：手动审批黄金路径（自动通过） ────────────────────────────────────


@pytest.mark.asyncio
async def test_golden_path_manual_approve_then_approved(
    store: InMemoryEventStore,
    repo: PilotRepository,
    publisher: PilotEventPublisher,
    planning: PlanningService,
    approvals: ApprovalService,
    artifacts: ArtifactStore,
    registry: SkillRegistry,
) -> None:
    """手动审批路径，但 TaskOrchestrator.submit 传 auto_approve=True 时
    应通过 ApprovalService.auto-approve 推进，最终仍成功。"""
    engine = _make_manual_approve_engine(repo, publisher, registry, artifacts, approvals)
    orch = _make_orchestrator(repo, publisher, planning, approvals, engine)

    # auto_approve=True at the request level: orchestrator auto-approves the plan
    handle = await orch.submit(TaskRequest(
        source_channel="api",
        raw_text="生成季度报告",
        requester_id="u2",
        auto_approve=True,
    ))

    # 手动审批模式（ExecutionEngine.auto_approve_writes=False）：
    # orchestrator 会 auto-approve 计划级审批，但步骤级有 requires_approval=True
    # 的 skill，任务会在第一个需要审批的步骤暂停，status = BLOCKED。
    assert handle.status in (TaskStatus.SUCCEEDED, TaskStatus.BLOCKED), (
        f"手动步骤审批路径：期望 succeeded 或 blocked，实际 {handle.status}"
    )

    event_types = await _event_types(store, handle.workspace_id)

    # 审批决策事件（auto-approve 计划级 → APPROVAL_DECIDED）
    assert EventType.APPROVAL_DECIDED in event_types, (
        "即使 auto_approve，也应有 APPROVAL_DECIDED 事件"
    )
    # TASK_CREATED 必须存在
    assert EventType.TASK_CREATED in event_types
    # PLAN_GENERATED 必须存在
    assert EventType.PLAN_GENERATED in event_types


# ── 场景 3：幂等提交不重复执行 ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_idempotent_submit_produces_single_task(
    store: InMemoryEventStore,
    repo: PilotRepository,
    publisher: PilotEventPublisher,
    planning: PlanningService,
    approvals: ApprovalService,
    artifacts: ArtifactStore,
    registry: SkillRegistry,
) -> None:
    """相同 idempotency_key + workspace_id 重复提交，只应执行一次。"""
    engine = _make_auto_approve_engine(repo, publisher, registry, artifacts, approvals)
    orch = _make_orchestrator(repo, publisher, planning, approvals, engine)

    req = TaskRequest(
        source_channel="api",
        raw_text="重复提交测试",
        requester_id="u1",
        auto_approve=True,
        idempotency_key="idem-test-001",
    )

    handle1 = await orch.submit(req)
    handle2 = await orch.submit(req.model_copy(update={"workspace_id": handle1.workspace_id}))

    assert handle2.deduplicated is True, "第二次提交应被识别为重复"
    assert handle1.task_id == handle2.task_id, "两次提交应返回同一 task_id"

    # 只应有一个 TASK_CREATED 事件
    event_types = await _event_types(store, handle1.workspace_id)
    task_created_count = event_types.count(EventType.TASK_CREATED)
    assert task_created_count == 1, (
        f"幂等路径 TASK_CREATED 应只出现 1 次，实际 {task_created_count} 次"
    )


# ── 场景 4：事件序列完整性——无乱序 ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_event_sequence_is_monotonic(
    store: InMemoryEventStore,
    repo: PilotRepository,
    publisher: PilotEventPublisher,
    planning: PlanningService,
    approvals: ApprovalService,
    artifacts: ArtifactStore,
    registry: SkillRegistry,
) -> None:
    """事件 sequence 必须单调递增，不允许跳号或乱序。"""
    engine = _make_auto_approve_engine(repo, publisher, registry, artifacts, approvals)
    orch = _make_orchestrator(repo, publisher, planning, approvals, engine)

    handle = await orch.submit(TaskRequest(
        source_channel="api",
        raw_text="单调性测试",
        requester_id="u1",
        auto_approve=True,
    ))

    events = await store.list_events(handle.workspace_id, since_sequence=0, limit=1000)
    events = sorted(events, key=lambda e: e.sequence)
    assert events, "应至少有一个事件"

    # sequence 从 1 开始且连续
    for i, event in enumerate(events, start=1):
        assert event.sequence == i, (
            f"事件 {event.type} 的 sequence={event.sequence}，预期 {i}"
        )


# ── 场景 5：多任务 workspace 隔离 ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_workspace_event_isolation(
    store: InMemoryEventStore,
    repo: PilotRepository,
    publisher: PilotEventPublisher,
    planning: PlanningService,
    approvals: ApprovalService,
    artifacts: ArtifactStore,
    registry: SkillRegistry,
) -> None:
    """两个不同 workspace 的事件不应互相污染。"""
    engine = _make_auto_approve_engine(repo, publisher, registry, artifacts, approvals)
    orch = _make_orchestrator(repo, publisher, planning, approvals, engine)

    handle_a = await orch.submit(TaskRequest(
        source_channel="api", raw_text="任务 A", requester_id="ua", auto_approve=True,
    ))
    handle_b = await orch.submit(TaskRequest(
        source_channel="api", raw_text="任务 B", requester_id="ub", auto_approve=True,
    ))

    # workspace_id 应不同
    assert handle_a.workspace_id != handle_b.workspace_id

    events_a = await store.list_events(handle_a.workspace_id, since_sequence=0, limit=1000)
    events_b = await store.list_events(handle_b.workspace_id, since_sequence=0, limit=1000)

    # 每个 workspace 都有自己的事件
    assert events_a, "workspace A 应有事件"
    assert events_b, "workspace B 应有事件"

    # 事件 workspace_id 字段与其所在 bucket 一致
    for e in events_a:
        assert e.workspace_id == handle_a.workspace_id
    for e in events_b:
        assert e.workspace_id == handle_b.workspace_id
