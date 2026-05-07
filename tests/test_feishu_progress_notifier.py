from __future__ import annotations

import asyncio
import json

import pytest

from agent_hub.connectors.feishu import FakeFeishuClient, FeishuProgressNotifier
from agent_hub.pilot import (
    EventType,
    ExecutionEvent,
    InMemoryEventStore,
    PilotRepository,
    Task,
    Workspace,
    WorkspaceStatus,
)


async def _wait_for_messages(
    client: FakeFeishuClient,
    count: int,
) -> None:
    for _ in range(10):
        if len(client.sent_messages) >= count:
            break
        await asyncio.sleep(0.01)


class _SlowFirstMessageClient(FakeFeishuClient):
    async def send_message(
        self,
        *,
        receive_id: str,
        receive_id_type: str,
        msg_type: str,
        content: str,
    ):
        if "正在执行：第一步" in content:
            await asyncio.sleep(0.05)
        return await super().send_message(
            receive_id=receive_id,
            receive_id_type=receive_id_type,
            msg_type=msg_type,
            content=content,
        )


@pytest.mark.asyncio
async def test_feishu_progress_notifier_sends_task_progress_message() -> None:
    store = InMemoryEventStore()
    repo = PilotRepository(store)
    client = FakeFeishuClient()
    notifier = FeishuProgressNotifier(repo, client, min_interval_seconds=1)

    workspace = Workspace(
        title="飞书任务",
        source_channel="feishu",
        source_conversation_id="oc_chat_1",
        feishu_chat_id="oc_chat_1",
        created_by="ou_user",
        members=["ou_user"],
        status=WorkspaceStatus.ACTIVE,
    )
    task = Task(
        workspace_id=workspace.workspace_id,
        origin_text="帮我生成 Agent-Hub 项目路演 PPT",
        requester_id="ou_user",
    )
    await repo.save(workspace, expected_version=None)
    await repo.save(task, expected_version=None)

    event = ExecutionEvent(
        workspace_id=workspace.workspace_id,
        trace_id=task.trace_id,
        task_id=task.task_id,
        type=EventType.TASK_PROGRESS,
        message="正在执行：渲染 PPTX",
        payload={"phase": "step_running", "kind": "milestone"},
    )

    notifier.handle_event(event)
    await _wait_for_messages(client, 1)
    await notifier.aclose()

    assert len(client.sent_messages) == 1
    sent = client.sent_messages[0]
    assert sent["receive_id"] == "oc_chat_1"
    payload = json.loads(sent["content"])
    assert "正在执行：渲染 PPTX" in payload["text"]
    assert "帮我生成 Agent-Hub 项目路演 PPT" in payload["text"]
    assert "阶段：步骤执行" in payload["text"]


@pytest.mark.asyncio
async def test_feishu_progress_notifier_silences_task_progress_while_approval_pending() -> None:
    store = InMemoryEventStore()
    repo = PilotRepository(store)
    client = FakeFeishuClient()
    notifier = FeishuProgressNotifier(repo, client, min_interval_seconds=1)

    workspace = Workspace(
        title="飞书任务",
        source_channel="feishu",
        source_conversation_id="oc_chat_1",
        feishu_chat_id="oc_chat_1",
        created_by="ou_user",
        members=["ou_user"],
        status=WorkspaceStatus.ACTIVE,
    )
    task = Task(
        workspace_id=workspace.workspace_id,
        origin_text="帮我生成企业发展 PPT",
        requester_id="ou_user",
    )
    await repo.save(workspace, expected_version=None)
    await repo.save(task, expected_version=None)

    notifier.handle_event(ExecutionEvent(
        workspace_id=workspace.workspace_id,
        trace_id=task.trace_id,
        task_id=task.task_id,
        approval_id="appr_demo",
        type=EventType.APPROVAL_REQUESTED,
        message="step approval requested",
    ))
    notifier.handle_event(ExecutionEvent(
        workspace_id=workspace.workspace_id,
        trace_id=task.trace_id,
        task_id=task.task_id,
        type=EventType.TASK_PROGRESS,
        message="步骤等待审批：上传 Drive 并生成分享",
        payload={"phase": "step_awaiting_approval", "kind": "milestone"},
    ))
    notifier.handle_event(ExecutionEvent(
        workspace_id=workspace.workspace_id,
        trace_id=task.trace_id,
        task_id=task.task_id,
        type=EventType.TASK_PROGRESS,
        message="任务暂停，等待审批或恢复",
        payload={"phase": "blocked", "kind": "milestone"},
    ))

    await asyncio.sleep(0.05)
    assert client.sent_messages == []

    notifier.handle_event(ExecutionEvent(
        workspace_id=workspace.workspace_id,
        trace_id=task.trace_id,
        task_id=task.task_id,
        approval_id="appr_demo",
        type=EventType.APPROVAL_DECIDED,
        message="step approval decided: approve",
        payload={"decision": "approve"},
    ))
    notifier.handle_event(ExecutionEvent(
        workspace_id=workspace.workspace_id,
        trace_id=task.trace_id,
        task_id=task.task_id,
        type=EventType.TASK_PROGRESS,
        message="任务已恢复，继续执行剩余步骤",
        payload={"phase": "execution", "kind": "milestone"},
    ))

    await _wait_for_messages(client, 1)
    await notifier.aclose()

    assert len(client.sent_messages) == 1
    payload = json.loads(client.sent_messages[0]["content"])
    assert "任务已恢复，继续执行剩余步骤" in payload["text"]


@pytest.mark.asyncio
async def test_feishu_progress_notifier_skips_low_signal_step_events() -> None:
    store = InMemoryEventStore()
    repo = PilotRepository(store)
    client = FakeFeishuClient()
    notifier = FeishuProgressNotifier(repo, client, min_interval_seconds=1)

    workspace = Workspace(
        title="飞书任务",
        source_channel="feishu",
        source_conversation_id="oc_chat_1",
        feishu_chat_id="oc_chat_1",
        created_by="ou_user",
        members=["ou_user"],
        status=WorkspaceStatus.ACTIVE,
    )
    task = Task(
        workspace_id=workspace.workspace_id,
        origin_text="帮我生成企业发展 PPT",
        requester_id="ou_user",
    )
    await repo.save(workspace, expected_version=None)
    await repo.save(task, expected_version=None)

    notifier.handle_event(ExecutionEvent(
        workspace_id=workspace.workspace_id,
        trace_id=task.trace_id,
        task_id=task.task_id,
        type=EventType.TASK_PROGRESS,
        message="准备执行：渲染 PPTX",
        payload={"phase": "step_prepare", "kind": "milestone"},
    ))
    notifier.handle_event(ExecutionEvent(
        workspace_id=workspace.workspace_id,
        trace_id=task.trace_id,
        task_id=task.task_id,
        type=EventType.TASK_PROGRESS,
        message="Agent 正在处理：渲染 PPTX",
        payload={"phase": "step_running", "kind": "milestone"},
    ))

    await asyncio.sleep(0.05)
    assert client.sent_messages == []

    notifier.handle_event(ExecutionEvent(
        workspace_id=workspace.workspace_id,
        trace_id=task.trace_id,
        task_id=task.task_id,
        type=EventType.TASK_PROGRESS,
        message="正在执行：渲染 PPTX",
        payload={"phase": "step_running", "kind": "milestone"},
    ))

    await _wait_for_messages(client, 1)
    await notifier.aclose()

    assert len(client.sent_messages) == 1
    payload = json.loads(client.sent_messages[0]["content"])
    assert "正在执行：渲染 PPTX" in payload["text"]


@pytest.mark.asyncio
async def test_feishu_progress_notifier_throttles_heartbeat_messages() -> None:
    store = InMemoryEventStore()
    repo = PilotRepository(store)
    client = FakeFeishuClient()
    notifier = FeishuProgressNotifier(repo, client, min_interval_seconds=60)

    workspace = Workspace(
        title="飞书任务",
        source_channel="feishu",
        source_conversation_id="oc_chat_1",
        feishu_chat_id="oc_chat_1",
        created_by="ou_user",
        members=["ou_user"],
        status=WorkspaceStatus.ACTIVE,
    )
    task = Task(
        workspace_id=workspace.workspace_id,
        origin_text="帮我生成企业发展 PPT",
        requester_id="ou_user",
    )
    await repo.save(workspace, expected_version=None)
    await repo.save(task, expected_version=None)

    heartbeat = ExecutionEvent(
        workspace_id=workspace.workspace_id,
        trace_id=task.trace_id,
        task_id=task.task_id,
        type=EventType.TASK_PROGRESS,
        message="Agent 仍在工作，最近活动 15 秒前",
        payload={
            "phase": "planning",
            "kind": "heartbeat",
            "interval_seconds": 15,
            "elapsed_seconds": 15,
        },
    )

    notifier.handle_event(heartbeat)
    notifier.handle_event(heartbeat)
    await _wait_for_messages(client, 1)
    await asyncio.sleep(0.05)
    await notifier.aclose()

    assert len(client.sent_messages) == 1
    payload = json.loads(client.sent_messages[0]["content"])
    assert "已持续 15 秒" in payload["text"]


@pytest.mark.asyncio
async def test_feishu_progress_notifier_preserves_order_per_task() -> None:
    store = InMemoryEventStore()
    repo = PilotRepository(store)
    client = _SlowFirstMessageClient()
    notifier = FeishuProgressNotifier(repo, client, min_interval_seconds=1)

    workspace = Workspace(
        title="飞书任务",
        source_channel="feishu",
        source_conversation_id="oc_chat_1",
        feishu_chat_id="oc_chat_1",
        created_by="ou_user",
        members=["ou_user"],
        status=WorkspaceStatus.ACTIVE,
    )
    task = Task(
        workspace_id=workspace.workspace_id,
        origin_text="帮我生成企业发展 PPT",
        requester_id="ou_user",
    )
    await repo.save(workspace, expected_version=None)
    await repo.save(task, expected_version=None)

    notifier.handle_event(ExecutionEvent(
        workspace_id=workspace.workspace_id,
        trace_id=task.trace_id,
        task_id=task.task_id,
        type=EventType.TASK_PROGRESS,
        message="正在执行：第一步",
        payload={"phase": "step_running", "kind": "milestone"},
    ))
    notifier.handle_event(ExecutionEvent(
        workspace_id=workspace.workspace_id,
        trace_id=task.trace_id,
        task_id=task.task_id,
        type=EventType.TASK_PROGRESS,
        message="步骤完成：第一步",
        payload={"phase": "step_completed", "kind": "milestone"},
    ))

    await _wait_for_messages(client, 2)
    await notifier.aclose()

    texts = [json.loads(message["content"])["text"] for message in client.sent_messages]
    assert texts[0].startswith("正在执行：第一步")
    assert texts[1].startswith("步骤完成：第一步")
