"""Pilot M2 API 端点测试。

通过独立装配 ``PilotRuntime`` + 路由，避免依赖主应用的 Pipeline。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_hub.api.pilot_routes import build_pilot_router
from agent_hub.api.pilot_runtime import build_pilot_runtime
from agent_hub.config.settings import Settings


@pytest.fixture()
def app() -> Iterator[FastAPI]:
    """构造仅含 Pilot 路由的最小 FastAPI 应用。"""
    settings = Settings(
        api_keys=["test-key"],
        pilot_enabled=True,
        pilot_store_path="",  # InMemory
        pilot_demo_mode=True,
        pilot_auto_approve_writes=False,
        pilot_admin_token="admin-secret",
        public_base_url="",
        feishu_enabled=False,
        pilot_use_real_gateway=False,
        pilot_use_real_chain=False,
    )
    runtime = build_pilot_runtime(settings)

    @asynccontextmanager
    async def lifespan(_a: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await runtime.aclose()

    fa = FastAPI(lifespan=lifespan)
    fa.include_router(build_pilot_router(runtime))
    fa.state.runtime = runtime
    yield fa


@pytest.fixture()
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def _submit(client: TestClient, **overrides: object) -> dict:
    body = {
        "source_channel": "api",
        "raw_text": "请帮我做一份关于Q4增长的演示稿",
        "requester_id": "tester-1",
        "title": "Q4 Growth",
    }
    body.update(overrides)
    resp = client.post("/api/pilot/tasks", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_public_url_keeps_relative_until_configured() -> None:
    settings = Settings(public_base_url="")
    assert settings.public_url("api/pilot/tasks/t1") == "/api/pilot/tasks/t1"

    settings = Settings(public_base_url="https://demo.example.com/")
    assert (
        settings.public_url("/api/pilot/tasks/t1")
        == "https://demo.example.com/api/pilot/tasks/t1"
    )


# ── 基础读写 ────────────────────────────────────────


def test_submit_and_get_detail(client: TestClient) -> None:
    handle = _submit(client)
    assert handle["task_id"]
    assert handle["plan_id"]
    assert handle["deduplicated"] is False
    assert handle["detail_url"] == f"/api/pilot/tasks/{handle['task_id']}"

    detail = client.get(handle["detail_url"]).json()
    assert detail["task"]["task_id"] == handle["task_id"]
    assert detail["active_plan"]["plan_id"] == handle["plan_id"]
    assert detail["latest_sequence"] >= 1
    assert isinstance(detail["available_actions"], list)


def test_idempotency_dedup(client: TestClient) -> None:
    first = _submit(client, idempotency_key="dup-key-1")
    # 复用同一 workspace_id，否则 idempotency 是 per-workspace 隔离的
    second = _submit(
        client,
        idempotency_key="dup-key-1",
        workspace_id=first["workspace_id"],
    )
    assert second["task_id"] == first["task_id"]
    assert second["deduplicated"] is True


def test_list_tasks_filter(client: TestClient) -> None:
    handle = _submit(client)
    resp = client.get("/api/pilot/tasks", params={"workspace_id": handle["workspace_id"]})
    assert resp.status_code == 200
    rows = resp.json()
    assert any(r["task_id"] == handle["task_id"] for r in rows)


def test_unknown_task_returns_404(client: TestClient) -> None:
    assert client.get("/api/pilot/tasks/missing-id").status_code == 404


def test_delete_task_hides_from_list_and_detail(client: TestClient) -> None:
    import asyncio

    handle = _submit(client)
    task_id = handle["task_id"]
    workspace_id = handle["workspace_id"]

    resp = client.delete(
        f"/api/pilot/tasks/{task_id}",
        params={"actor_id": "tester-1"},
    )
    assert resp.status_code == 204, resp.text

    rows = client.get("/api/pilot/tasks").json()
    assert task_id not in {row["task_id"] for row in rows}
    assert client.get(f"/api/pilot/tasks/{task_id}").status_code == 404

    runtime = client.app.state.runtime

    async def _read_deleted_marker() -> tuple[str | None, bool]:
        task = await runtime.repository.get_task(task_id)
        events = await runtime.event_store.list_events(
            workspace_id,
            since_sequence=0,
            limit=100,
        )
        return (
            task.metadata.get("deleted_by") if task is not None else None,
            any(
                event.task_id == task_id
                and event.payload.get("deleted") is True
                for event in events
            ),
        )

    deleted_by, event_seen = asyncio.new_event_loop().run_until_complete(
        _read_deleted_marker(),
    )
    assert deleted_by == "tester-1"
    assert event_seen is True


def test_delete_unknown_task_returns_404(client: TestClient) -> None:
    resp = client.delete("/api/pilot/tasks/missing-id")
    assert resp.status_code == 404


def test_delete_running_task_returns_409(client: TestClient) -> None:
    import asyncio

    from agent_hub.pilot.domain.models import Task, Workspace

    runtime = client.app.state.runtime

    async def _seed_running_task() -> str:
        ws = Workspace(
            title="running workspace",
            source_channel="api",
            created_by="tester-1",
        )
        task = Task(
            workspace_id=ws.workspace_id,
            origin_text="running task",
            requester_id="tester-1",
            status="running",
        )
        await runtime.repository.save(ws, expected_version=None)
        await runtime.repository.save(task, expected_version=None)
        return task.task_id

    task_id = asyncio.new_event_loop().run_until_complete(_seed_running_task())
    resp = client.delete(
        f"/api/pilot/tasks/{task_id}",
        params={"actor_id": "tester-1"},
    )
    assert resp.status_code == 409


# ── 审批 ────────────────────────────────────────────


def test_pause_running_task_exposes_action_and_records_request(
    client: TestClient,
) -> None:
    import asyncio

    from agent_hub.pilot.domain.models import Task, Workspace

    runtime = client.app.state.runtime

    async def _seed_running_task() -> tuple[str, str]:
        ws = Workspace(
            title="running workspace",
            source_channel="api",
            created_by="tester-1",
        )
        task = Task(
            workspace_id=ws.workspace_id,
            origin_text="running task",
            requester_id="tester-1",
            status="running",
        )
        await runtime.repository.save(ws, expected_version=None)
        await runtime.repository.save(task, expected_version=None)
        return task.task_id, task.workspace_id

    task_id, workspace_id = asyncio.new_event_loop().run_until_complete(
        _seed_running_task(),
    )

    detail = client.get(f"/api/pilot/tasks/{task_id}")
    assert detail.status_code == 200, detail.text
    assert any(
        action["type"] == "pause_task"
        for action in detail.json()["available_actions"]
    )

    resp = client.post(
        f"/api/pilot/tasks/{task_id}/pause",
        params={"actor_id": "tester-1"},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["task_id"] == task_id
    assert payload["task_status"] == "running"
    assert payload["pause_requested"] is True

    async def _read_pause_marker() -> tuple[str | None, bool]:
        task = await runtime.repository.get_task(task_id)
        events = await runtime.event_store.list_events(
            workspace_id,
            since_sequence=0,
            limit=100,
        )
        return (
            task.metadata.get("pause_requested_by") if task is not None else None,
            any(
                event.task_id == task_id
                and event.payload.get("pause_requested") is True
                for event in events
            ),
        )

    requested_by, event_seen = asyncio.new_event_loop().run_until_complete(
        _read_pause_marker(),
    )
    assert requested_by == "tester-1"
    assert event_seen is True


def test_pause_unknown_task_returns_404(client: TestClient) -> None:
    resp = client.post("/api/pilot/tasks/missing-id/pause")
    assert resp.status_code == 404


def _wait_for_pending_approval(
    client: TestClient, task_id: str, timeout: float = 2.0,
) -> dict | None:
    """轮询 detail，直到至少一个 approval 处于 requested。"""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        detail = client.get(f"/api/pilot/tasks/{task_id}").json()
        for ap in detail["approvals"]:
            if ap["status"] == "requested":
                return ap
        if detail["task"]["status"] in {"succeeded", "failed", "canceled"}:
            return None
    return None


def test_approval_approve_runs_task(client: TestClient) -> None:
    handle = _submit(client)
    pending = _wait_for_pending_approval(client, handle["task_id"])
    assert pending is not None, "demo skill 应在第一个写动作处产生审批"

    decision = client.post(
        f"/api/pilot/approvals/{pending['approval_id']}/decision",
        json={
            "decision": "approve",
            "actor_id": "tester-1",
            "comment": "ok",
            "run_after_approval": True,
        },
    )
    assert decision.status_code == 200, decision.text
    payload = decision.json()
    assert payload["approval_status"] == "approved"
    # run_result 在 approve plan 时返回任务终态；可能在第一个 write step
    # 处再次进入 BLOCKED 等待 step 审批。
    assert payload["task_status"] in {"succeeded", "failed", "running", "blocked"}


def test_approval_reject_finalizes_task(client: TestClient) -> None:
    handle = _submit(client)
    pending = _wait_for_pending_approval(client, handle["task_id"])
    assert pending is not None

    resp = client.post(
        f"/api/pilot/approvals/{pending['approval_id']}/decision",
        json={
            "decision": "reject",
            "actor_id": "tester-1",
            "comment": "no",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["approval_status"] == "rejected"

    detail = client.get(f"/api/pilot/tasks/{handle['task_id']}").json()
    assert detail["task"]["status"] in {"failed", "canceled"}


def test_unknown_approval_returns_404(client: TestClient) -> None:
    resp = client.post(
        "/api/pilot/approvals/missing/decision",
        json={"decision": "approve", "actor_id": "x"},
    )
    assert resp.status_code == 404


# ── 续跑 ────────────────────────────────────────────


def _approve_plan_then_blocked(client: TestClient) -> dict:
    """提交任务，approve plan 进入 step 审批阶段，返回 task detail。"""
    handle = _submit(client)
    pending = _wait_for_pending_approval(client, handle["task_id"])
    assert pending is not None
    decision = client.post(
        f"/api/pilot/approvals/{pending['approval_id']}/decision",
        json={
            "decision": "approve",
            "actor_id": "tester-1",
            "run_after_approval": True,
        },
    )
    assert decision.status_code == 200
    detail = client.get(f"/api/pilot/tasks/{handle['task_id']}").json()
    return {"handle": handle, "detail": detail}


def test_step_approval_auto_resumes(client: TestClient) -> None:
    """approve plan 后停在 step 审批；approve step 应在同 task 上自动续跑。"""
    bundle = _approve_plan_then_blocked(client)
    handle = bundle["handle"]
    detail = bundle["detail"]
    assert detail["task"]["status"] == "blocked"
    step_action = next(
        a for a in detail["available_actions"] if a["type"] == "decide_step"
    )
    blocked_step_id = step_action["step_id"]

    decision = client.post(
        f"/api/pilot/approvals/{step_action['approval_id']}/decision",
        json={
            "decision": "approve",
            "actor_id": "tester-1",
            "run_after_approval": True,
        },
    )
    assert decision.status_code == 200, decision.text
    payload = decision.json()
    assert payload["approval_status"] == "approved"
    assert payload["step_status"] == "approved"
    # 续跑后任务状态应在 blocked（下一 step 审批）/succeeded/failed 之间
    assert payload["task_status"] in {"blocked", "succeeded", "failed"}
    assert payload["run_result"] is not None

    refreshed = client.get(f"/api/pilot/tasks/{handle['task_id']}").json()
    refreshed_step = next(
        s for s in refreshed["steps"] if s["step_id"] == blocked_step_id
    )
    assert refreshed_step["status"] == "succeeded"
    assert refreshed_step["output_artifact_id"]


def test_step_reject_finalizes_task_failed(client: TestClient) -> None:
    bundle = _approve_plan_then_blocked(client)
    handle = bundle["handle"]
    step_action = next(
        a for a in bundle["detail"]["available_actions"]
        if a["type"] == "decide_step"
    )

    resp = client.post(
        f"/api/pilot/approvals/{step_action['approval_id']}/decision",
        json={"decision": "reject", "actor_id": "tester-1", "comment": "no"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["approval_status"] == "rejected"
    assert payload["step_status"] == "failed"
    assert payload["task_status"] == "failed"

    detail = client.get(f"/api/pilot/tasks/{handle['task_id']}").json()
    assert detail["task"]["status"] == "failed"
    types = {a["type"] for a in detail["available_actions"]}
    assert "decide_step" not in types


def test_resume_endpoint_recovers_blocked_after_manual_approval(
    client: TestClient,
) -> None:
    """approve step 时 run_after_approval=false → blocked 持续；
    POST /tasks/{id}/resume 应在 detail 中显示 resume_task 动作并能成功续跑。
    """
    bundle = _approve_plan_then_blocked(client)
    handle = bundle["handle"]
    step_action = next(
        a for a in bundle["detail"]["available_actions"]
        if a["type"] == "decide_step"
    )

    decision = client.post(
        f"/api/pilot/approvals/{step_action['approval_id']}/decision",
        json={
            "decision": "approve",
            "actor_id": "tester-1",
            "run_after_approval": False,
        },
    )
    assert decision.status_code == 200
    payload = decision.json()
    assert payload["task_status"] == "blocked"
    assert "run_result" not in payload or payload["run_result"] is None

    detail = client.get(f"/api/pilot/tasks/{handle['task_id']}").json()
    assert detail["task"]["status"] == "blocked"
    types = {a["type"] for a in detail["available_actions"]}
    assert "resume_task" in types

    resume = client.post(f"/api/pilot/tasks/{handle['task_id']}/resume")
    assert resume.status_code == 200, resume.text
    body = resume.json()
    assert body["task_id"] == handle["task_id"]
    assert body["task_status"] in {"blocked", "succeeded", "failed"}


def test_resume_endpoint_rejects_non_blocked_task(client: TestClient) -> None:
    handle = _submit(client)
    resp = client.post(f"/api/pilot/tasks/{handle['task_id']}/resume")
    assert resp.status_code == 409


def test_resume_endpoint_unknown_task_returns_404(client: TestClient) -> None:
    resp = client.post("/api/pilot/tasks/missing/resume")
    assert resp.status_code == 404


# ── 工件 ────────────────────────────────────────────


def test_recover_step_endpoint_resumes_same_task(client: TestClient) -> None:
    import asyncio

    from agent_hub.pilot import (
        Plan,
        PlanStatus,
        PlanStep,
        PlanStepKind,
        Task,
        TaskAction,
        TaskStatus,
        Workspace,
        WorkspaceStatus,
        transition,
    )
    from agent_hub.pilot.skills.registry import (
        SkillInvocation,
        SkillResult,
        SkillSpec,
    )

    runtime = client.app.state.runtime
    calls = {"flaky": 0}

    async def ok(inv: SkillInvocation) -> SkillResult:
        return SkillResult(skill_name=inv.skill_name, success=True)

    async def flaky(inv: SkillInvocation) -> SkillResult:
        calls["flaky"] += 1
        if calls["flaky"] == 1:
            return SkillResult(
                skill_name=inv.skill_name,
                success=False,
                error="temporary failure",
            )
        return SkillResult(skill_name=inv.skill_name, success=True)

    runtime.registry.register(SkillSpec(
        skill_name="api.recover.ok",
        supports_dry_run=False,
    ), ok)
    runtime.registry.register(SkillSpec(
        skill_name="api.recover.flaky",
        supports_dry_run=False,
    ), flaky)

    async def _seed_failed_task() -> tuple[str, str]:
        ws = Workspace(
            title="api recover",
            source_channel="api",
            created_by="tester-1",
            status=WorkspaceStatus.ACTIVE,
        )
        task = Task(
            workspace_id=ws.workspace_id,
            origin_text="recover this task",
            requester_id="tester-1",
        )
        plan = Plan(
            workspace_id=ws.workspace_id,
            task_id=task.task_id,
            goal="recover failed step",
            status=PlanStatus.APPROVED,
        )
        first = PlanStep(
            workspace_id=ws.workspace_id,
            task_id=task.task_id,
            plan_id=plan.plan_id,
            index=0,
            title="first",
            kind=PlanStepKind.READ_CONTEXT,
            skill_name="api.recover.ok",
        )
        failed = PlanStep(
            workspace_id=ws.workspace_id,
            task_id=task.task_id,
            plan_id=plan.plan_id,
            index=1,
            title="flaky",
            kind=PlanStepKind.SUMMARIZE,
            skill_name="api.recover.flaky",
            depends_on=[first.step_id],
        )
        plan = plan.model_copy(update={
            "step_ids": [first.step_id, failed.step_id],
        })
        task = task.model_copy(update={"plan_id": plan.plan_id})
        task = transition(task, TaskAction.START_PLANNING)
        task = transition(task, TaskAction.REQUEST_APPROVAL)
        task = transition(task, TaskAction.APPROVE)

        await runtime.repository.save(ws, expected_version=None)
        await runtime.repository.save(task, expected_version=None)
        await runtime.repository.save(plan, expected_version=None)
        await runtime.repository.save(first, expected_version=None)
        await runtime.repository.save(failed, expected_version=None)

        first_run = await runtime.execution.run_plan(task, plan)
        assert first_run.final_status == TaskStatus.FAILED
        return task.task_id, failed.step_id

    task_id, failed_step_id = asyncio.new_event_loop().run_until_complete(
        _seed_failed_task(),
    )

    detail = client.get(f"/api/pilot/tasks/{task_id}").json()
    action_types = {a["type"] for a in detail["available_actions"]}
    assert {"retry_task", "recover_from"} <= action_types

    resp = client.post(
        f"/api/pilot/tasks/{task_id}/steps/{failed_step_id}/recover",
        json={"requester_id": "tester-1"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["task_id"] == task_id
    assert body["task_status"] == "succeeded"
    assert calls["flaky"] == 2


def test_unknown_artifact_returns_404(client: TestClient) -> None:
    assert client.get("/api/pilot/artifacts/missing").status_code == 404


def test_unknown_artifact_download_returns_404(client: TestClient) -> None:
    assert (
        client.get("/api/pilot/artifacts/missing/download").status_code == 404
    )


@pytest.fixture()
def auto_client(tmp_path) -> Iterator[TestClient]:
    """跑完整 fake demo 链路的客户端：自动审批 + 临时 artifact 目录。"""
    settings = Settings(
        api_keys=["test-key"],
        pilot_enabled=True,
        pilot_store_path="",
        pilot_demo_mode=True,
        pilot_auto_approve_writes=True,
        pilot_admin_token="admin-secret",
        public_base_url="",
        feishu_enabled=False,
        pilot_use_real_gateway=False,
        pilot_use_real_chain=False,
        pilot_artifact_dir=str(tmp_path / "artifacts"),
    )
    runtime = build_pilot_runtime(settings)

    @asynccontextmanager
    async def lifespan(_a: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await runtime.aclose()

    fa = FastAPI(lifespan=lifespan)
    fa.include_router(build_pilot_router(runtime))
    fa.state.runtime = runtime
    with TestClient(fa) as c:
        yield c


def test_artifact_download_returns_attachment(
    auto_client: TestClient,
) -> None:
    """直接通过 runtime 注入一个带 content 的 artifact 后下载。"""
    import asyncio

    from agent_hub.pilot.domain.enums import ArtifactStatus, ArtifactType
    from agent_hub.pilot.domain.models import Artifact

    runtime = auto_client.app.state.runtime
    handle = _submit(auto_client)
    detail = auto_client.get(f"/api/pilot/tasks/{handle['task_id']}").json()
    workspace_id = detail["task"]["workspace_id"]
    task_id = detail["task"]["task_id"]

    pptx_bytes = b"PK\x03\x04fake-pptx-bytes-for-test"

    async def _seed() -> str:
        artifact = Artifact(
            workspace_id=workspace_id,
            task_id=task_id,
            step_id=None,
            type=ArtifactType.PPTX,
            title="季度复盘.pptx",
            artifact_version=1,
            mime_type=(
                "application/vnd.openxmlformats-officedocument"
                ".presentationml.presentation"
            ),
            storage_key="memory://placeholder",
            status=ArtifactStatus.GENERATED,
        )
        await runtime.repository.save(artifact, expected_version=None)
        runtime.artifacts._memory_blobs[artifact.artifact_id] = pptx_bytes  # noqa: SLF001
        bound = artifact.model_copy(update={
            "storage_key": f"memory://{artifact.artifact_id}",
            "uri": f"memory://{artifact.artifact_id}",
        })
        await runtime.repository.save(bound, expected_version=artifact.version)
        return artifact.artifact_id

    artifact_id = asyncio.new_event_loop().run_until_complete(_seed())

    resp = auto_client.get(f"/api/pilot/artifacts/{artifact_id}/download")
    assert resp.status_code == 200, resp.text
    assert resp.content == pptx_bytes
    disp = resp.headers.get("content-disposition", "")
    assert "attachment" in disp.lower()
    assert "filename*=UTF-8''" in disp
    assert resp.headers.get("content-type", "").startswith(
        "application/vnd.openxmlformats-officedocument",
    )


# ── 管理员路由 ──────────────────────────────────────


def test_recover_requires_admin_token(client: TestClient) -> None:
    handle = _submit(client)
    resp = client.post(
        f"/api/pilot/tasks/{handle['task_id']}/recover_from/step-x",
        json={"requester_id": "tester-1"},
    )
    # 缺少 token 直接 401
    assert resp.status_code == 401


# ── SSE ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sse_backfill(app: FastAPI, client: TestClient) -> None:
    """直接调用 SSE 生成器，避免 TestClient 与 EventSourceResponse 的同步阻塞。"""
    from agent_hub.api.pilot_streaming import pilot_event_stream

    handle = _submit(client)
    runtime = app.state.runtime
    workspace_id = handle["workspace_id"]
    task_id = handle["task_id"]

    gen = pilot_event_stream(
        workspace_id=workspace_id,
        task_id=task_id,
        since_sequence=0,
        store=runtime.event_store,
        bus=runtime.bus,
    )
    seen_event = False
    backfill_done = False
    try:
        async for raw in gen:
            payload = json.loads(raw)
            if payload.get("type") == "event":
                seen_event = True
            elif payload.get("type") == "backfill_done":
                backfill_done = True
                break
    finally:
        await gen.aclose()
    assert seen_event
    assert backfill_done
