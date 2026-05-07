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
    for _ in range(10):
        if client.sent_messages:
            break
        await asyncio.sleep(0.01)
    await notifier.aclose()

    assert len(client.sent_messages) == 1
    sent = client.sent_messages[0]
    assert sent["receive_id"] == "oc_chat_1"
    payload = json.loads(sent["content"])
    assert "正在执行：渲染 PPTX" in payload["text"]
    assert "帮我生成 Agent-Hub 项目路演 PPT" in payload["text"]
    assert "阶段：步骤执行" in payload["text"]