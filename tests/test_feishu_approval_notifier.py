from __future__ import annotations

import pytest

from agent_hub.connectors.feishu import FakeFeishuClient
from agent_hub.connectors.feishu.approval_notifier import FeishuApprovalNotifier
from agent_hub.pilot import (
    Approval,
    ApprovalStatus,
    ApprovalTargetType,
    InMemoryEventStore,
    PilotRepository,
    PlanStep,
    PlanStepKind,
    RiskLevel,
    Task,
    Workspace,
    WorkspaceStatus,
)


@pytest.mark.asyncio
async def test_feishu_approval_notifier_sends_card_and_persists_message_id() -> None:
    store = InMemoryEventStore()
    repo = PilotRepository(store)
    client = FakeFeishuClient()
    notifier = FeishuApprovalNotifier(repo, client)

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
    step = PlanStep(
        workspace_id=workspace.workspace_id,
        task_id=task.task_id,
        plan_id="plan_demo",
        index=3,
        title="渲染 PPTX",
        kind=PlanStepKind.RENDER_SLIDES,
        skill_name="fake_pptx_render",
        risk_level=RiskLevel.WRITE,
        requires_approval=True,
    )
    approval = Approval(
        workspace_id=workspace.workspace_id,
        task_id=task.task_id,
        target_type=ApprovalTargetType.STEP,
        target_id=step.step_id,
        requester_id="system",
        status=ApprovalStatus.REQUESTED,
    )

    await repo.save(workspace, expected_version=None)
    await repo.save(task, expected_version=None)
    await repo.save(step, expected_version=None)
    await repo.save(approval, expected_version=None)

    await notifier.notify_requested(approval)

    assert len(client.sent_cards) == 1
    sent = client.sent_cards[0]
    assert sent["receive_id"] == "oc_chat_1"
    assert sent["card"]["header"]["title"]["content"].startswith("步骤审批")

    refreshed = await repo.get_approval(approval.approval_id)
    assert refreshed is not None
    assert refreshed.channel_message_id == sent["message_id"]