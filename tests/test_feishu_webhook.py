"""``FeishuWebhookProcessor`` 与 ``FeishuWebhookService`` 单元测试。"""

from __future__ import annotations

import base64
import hashlib
import json

import pytest

from agent_hub.connectors.feishu import (
    FakeFeishuClient,
    FeishuChatType,
    FeishuMessageType,
    FeishuWebhookError,
    FeishuWebhookOutcome,
    FeishuWebhookProcessor,
    FeishuWebhookService,
)
from agent_hub.pilot import (
    ApprovalService,
    ArtifactStore,
    EventBus,
    ExecutionEngine,
    FakeModelGateway,
    InMemoryEventStore,
    PilotEventPublisher,
    PilotRepository,
    PlanningService,
    SkillRegistry,
    TaskOrchestrator,
    register_fake_skills,
)

# ── 通用 builders ──────────────────────────────────


def _make_orchestrator() -> tuple[TaskOrchestrator, PilotRepository]:
    store = InMemoryEventStore()
    bus = EventBus()
    repo = PilotRepository(store)
    publisher = PilotEventPublisher(store, bus)
    registry = SkillRegistry()
    register_fake_skills(registry)
    gateway = FakeModelGateway()
    planning = PlanningService(gateway, repo, publisher)
    approvals = ApprovalService(repo, publisher)
    artifacts = ArtifactStore(repo, publisher)
    execution = ExecutionEngine(
        repository=repo,
        publisher=publisher,
        registry=registry,
        artifacts=artifacts,
        approvals=approvals,
        auto_approve_writes=True,
    )
    orchestrator = TaskOrchestrator(
        repository=repo,
        publisher=publisher,
        planning=planning,
        approvals=approvals,
        execution=execution,
    )
    return orchestrator, repo


def _text_message_event(
    *,
    text: str = "请帮我写一份介绍",
    chat_id: str = "oc_chat_1",
    sender_id: str = "ou_sender_1",
    chat_type: str = "p2p",
    event_id: str = "evt-001",
    message_id: str = "om_msg_1",
    mentions: list[dict] | None = None,
    token: str = "",
) -> dict:
    body = {
        "schema": "2.0",
        "header": {
            "event_id": event_id,
            "event_type": "im.message.receive_v1",
            "create_time": "1700000000",
            "token": token,
            "app_id": "cli_app",
            "tenant_key": "tenant1",
        },
        "event": {
            "sender": {
                "sender_id": {"open_id": sender_id, "user_id": "u1"},
                "sender_type": "user",
            },
            "message": {
                "message_id": message_id,
                "chat_id": chat_id,
                "chat_type": chat_type,
                "message_type": "text",
                "create_time": "1700000000",
                "content": json.dumps({"text": text}),
                "mentions": mentions or [],
            },
        },
    }
    return body


# ── Processor ──────────────────────────────────────


@pytest.mark.asyncio
async def test_processor_url_challenge_returns_challenge() -> None:
    proc = FeishuWebhookProcessor(verification_token="vt")
    result = await proc.handle_payload(
        {"type": "url_verification", "challenge": "ch-1", "token": "vt"},
    )
    assert result.outcome is FeishuWebhookOutcome.CHALLENGE
    assert result.challenge == "ch-1"


@pytest.mark.asyncio
async def test_processor_invalid_token_raises() -> None:
    proc = FeishuWebhookProcessor(verification_token="vt")
    with pytest.raises(FeishuWebhookError):
        await proc.handle_payload(
            {"type": "url_verification", "challenge": "x", "token": "wrong"},
        )


@pytest.mark.asyncio
async def test_processor_normalize_text_message() -> None:
    proc = FeishuWebhookProcessor(verification_token="vt")
    result = await proc.handle_payload(_text_message_event(token="vt"))
    assert result.outcome is FeishuWebhookOutcome.ACCEPTED
    assert result.message is not None
    assert result.message.text == "请帮我写一份介绍"
    assert result.message.chat_id == "oc_chat_1"
    assert result.message.message_type is FeishuMessageType.TEXT
    assert result.message.chat_type is FeishuChatType.P2P
    assert result.message.tenant_key == "tenant1"
    assert result.message.event_id == "evt-001"


@pytest.mark.asyncio
async def test_processor_dedup_event_id() -> None:
    proc = FeishuWebhookProcessor(verification_token="vt")
    payload = _text_message_event(token="vt")
    first = await proc.handle_payload(payload)
    again = await proc.handle_payload(payload)
    assert first.outcome is FeishuWebhookOutcome.ACCEPTED
    assert again.outcome is FeishuWebhookOutcome.DUPLICATE


@pytest.mark.asyncio
async def test_processor_group_chat_requires_mention() -> None:
    proc = FeishuWebhookProcessor(
        verification_token="vt",
        bot_open_id="ou_bot",
        require_mention_in_group=True,
    )
    no_mention = _text_message_event(
        token="vt",
        chat_type="group",
        event_id="evt-no-mention",
        message_id="om_2",
    )
    result = await proc.handle_payload(no_mention)
    assert result.outcome is FeishuWebhookOutcome.IGNORED

    with_mention = _text_message_event(
        token="vt",
        chat_type="group",
        event_id="evt-with-mention",
        message_id="om_3",
        text="@_user_1 你好，这是一条测试内容",
        mentions=[{"id": {"open_id": "ou_bot"}}],
    )
    result = await proc.handle_payload(with_mention)
    assert result.outcome is FeishuWebhookOutcome.ACCEPTED
    assert result.message is not None
    assert result.message.bot_mentioned is True


@pytest.mark.asyncio
async def test_processor_strips_bot_mention_key_from_text() -> None:
    proc = FeishuWebhookProcessor(
        verification_token="vt",
        bot_open_id="ou_bot",
        require_mention_in_group=True,
    )
    payload = _text_message_event(
        token="vt",
        chat_type="group",
        event_id="evt-strip-mention",
        message_id="om_strip",
        text="@_user_1 你好，这是一条测试内容，你需要回复我“我很好”",
        mentions=[{"key": "@_user_1", "id": {"open_id": "ou_bot"}}],
    )

    result = await proc.handle_payload(payload)

    assert result.outcome is FeishuWebhookOutcome.ACCEPTED
    assert result.message is not None
    assert result.message.text == "你好，这是一条测试内容，你需要回复我“我很好”"


@pytest.mark.asyncio
async def test_processor_trigger_keyword_filter() -> None:
    proc = FeishuWebhookProcessor(
        verification_token="vt",
        trigger_keywords={"PPT"},
    )
    miss = await proc.handle_payload(
        _text_message_event(token="vt", text="hello", event_id="evt-miss"),
    )
    hit = await proc.handle_payload(
        _text_message_event(token="vt", text="帮我做一份 ppt", event_id="evt-hit"),
    )
    assert miss.outcome is FeishuWebhookOutcome.IGNORED
    assert hit.outcome is FeishuWebhookOutcome.ACCEPTED


@pytest.mark.asyncio
async def test_processor_unknown_event_type_ignored() -> None:
    proc = FeishuWebhookProcessor(verification_token="vt")
    payload = {
        "schema": "2.0",
        "header": {
            "event_id": "evt-other",
            "event_type": "im.chat.member_added",
            "token": "vt",
        },
        "event": {},
    }
    result = await proc.handle_payload(payload)
    assert result.outcome is FeishuWebhookOutcome.IGNORED


@pytest.mark.asyncio
async def test_processor_decrypts_encrypted_payload() -> None:
    cryptography = pytest.importorskip("cryptography")  # noqa: F841
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    encrypt_key = "my-encrypt-key"
    payload = _text_message_event(token="vt", event_id="evt-encrypted")
    plaintext = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    pad_len = 16 - (len(plaintext) % 16)
    plaintext += bytes([pad_len]) * pad_len
    iv = b"0123456789abcdef"
    key = hashlib.sha256(encrypt_key.encode("utf-8")).digest()
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    enc = cipher.encryptor()
    ciphertext = iv + enc.update(plaintext) + enc.finalize()
    envelope = {"encrypt": base64.b64encode(ciphertext).decode("ascii")}

    proc = FeishuWebhookProcessor(verification_token="vt", encrypt_key=encrypt_key)
    result = await proc.handle_payload(envelope)
    assert result.outcome is FeishuWebhookOutcome.ACCEPTED
    assert result.message is not None


# ── Service ────────────────────────────────────────


@pytest.mark.asyncio
async def test_service_creates_task_and_acks() -> None:
    orchestrator, repo = _make_orchestrator()
    fake_client = FakeFeishuClient()
    proc = FeishuWebhookProcessor(verification_token="vt")
    service = FeishuWebhookService(
        processor=proc,
        orchestrator=orchestrator,
        repository=repo,
        client=fake_client,
    )

    result = await service.handle(_text_message_event(token="vt"))
    assert result.outcome is FeishuWebhookOutcome.ACCEPTED
    assert result.handle is not None
    assert result.ack_message_id is not None
    assert len(fake_client.sent_messages) == 1
    sent = fake_client.sent_messages[0]
    assert sent["receive_id"] == "oc_chat_1"
    assert sent["msg_type"] == "text"
    assert "请帮我写一份介绍" in sent["content"] or "正在规划" in sent["content"]

    # workspace 被记录为 feishu chat
    workspaces = await repo.list_workspaces()
    assert any(w.feishu_chat_id == "oc_chat_1" for w in workspaces)


@pytest.mark.asyncio
async def test_service_reuses_workspace_for_same_chat() -> None:
    orchestrator, repo = _make_orchestrator()
    proc = FeishuWebhookProcessor(verification_token="vt")
    service = FeishuWebhookService(
        processor=proc,
        orchestrator=orchestrator,
        repository=repo,
        client=None,
    )

    a = await service.handle(_text_message_event(token="vt", event_id="ea"))
    b = await service.handle(
        _text_message_event(token="vt", event_id="eb", message_id="om_2"),
    )
    assert a.handle is not None and b.handle is not None
    assert a.handle.workspace_id == b.handle.workspace_id


@pytest.mark.asyncio
async def test_service_returns_challenge_outcome() -> None:
    orchestrator, repo = _make_orchestrator()
    proc = FeishuWebhookProcessor(verification_token="vt")
    service = FeishuWebhookService(
        processor=proc,
        orchestrator=orchestrator,
        repository=repo,
        client=None,
    )
    result = await service.handle(
        {"type": "url_verification", "challenge": "abc", "token": "vt"},
    )
    assert result.outcome is FeishuWebhookOutcome.CHALLENGE
    assert result.challenge == "abc"


# ── Card callback (M5 审批) ────────────────────────


def _card_action_event(
    *,
    approval_id: str = "ap_001",
    decision: str = "approve",
    operator_open_id: str = "ou_approver",
    comment: str = "ok",
    event_id: str = "evt-card-1",
    event_type: str = "card.action.trigger",
) -> dict:
    return {
        "schema": "2.0",
        "header": {
            "event_id": event_id,
            "event_type": event_type,
            "create_time": "1700000000",
            "token": "",
            "app_id": "cli_app",
            "tenant_key": "tenant1",
        },
        "event": {
            "operator": {
                "open_id": operator_open_id,
                "user_id": "",
                "union_id": "",
                "tenant_key": "tenant1",
            },
            "action": {
                "tag": "button",
                "value": {
                    "approval_id": approval_id,
                    "decision": decision,
                    "comment": comment,
                },
            },
            "context": {
                "open_chat_id": "oc_chat",
                "open_message_id": "om_msg",
            },
        },
    }


@pytest.mark.asyncio
async def test_service_card_callback_invokes_decide_approval() -> None:
    """卡片审批回调应调用 PilotCommandService.decide_approval。"""
    from unittest.mock import AsyncMock

    orchestrator, repo = _make_orchestrator()
    proc = FeishuWebhookProcessor()
    commands = AsyncMock()
    commands.decide_approval = AsyncMock(
        return_value={"approval_status": "approved"}
    )
    service = FeishuWebhookService(
        processor=proc,
        orchestrator=orchestrator,
        repository=repo,
        client=None,
        commands=commands,
    )

    result = await service.handle(_card_action_event())
    assert result.outcome is FeishuWebhookOutcome.CARD_CALLBACK
    assert result.approval_decision == {"approval_status": "approved"}
    commands.decide_approval.assert_awaited_once_with(
        "ap_001",
        decision="approve",
        actor_id="ou_approver",
        comment="ok",
    )


@pytest.mark.asyncio
async def test_service_card_callback_accepts_p2_event_type() -> None:
    """长连接 SDK 推送的 ``p2.card.action.trigger`` 也应被识别。"""
    from unittest.mock import AsyncMock

    orchestrator, repo = _make_orchestrator()
    proc = FeishuWebhookProcessor()
    commands = AsyncMock()
    commands.decide_approval = AsyncMock(return_value={"status": "ok"})
    service = FeishuWebhookService(
        processor=proc,
        orchestrator=orchestrator,
        repository=repo,
        client=None,
        commands=commands,
    )

    result = await service.handle(
        _card_action_event(event_type="p2.card.action.trigger"),
    )
    assert result.outcome is FeishuWebhookOutcome.CARD_CALLBACK
    commands.decide_approval.assert_awaited_once()


@pytest.mark.asyncio
async def test_service_card_callback_decide_failure_rejected() -> None:
    """decide_approval 抛异常时返回 REJECTED 并带 reason。"""
    from unittest.mock import AsyncMock

    orchestrator, repo = _make_orchestrator()
    proc = FeishuWebhookProcessor()
    commands = AsyncMock()
    commands.decide_approval = AsyncMock(side_effect=RuntimeError("not found"))
    service = FeishuWebhookService(
        processor=proc,
        orchestrator=orchestrator,
        repository=repo,
        client=None,
        commands=commands,
    )

    result = await service.handle(_card_action_event(event_id="evt-card-fail"))
    assert result.outcome is FeishuWebhookOutcome.REJECTED
    assert result.reason is not None
    assert "not found" in result.reason


# ── 审批后更新卡片终态 ─────────────────────────────


@pytest.mark.asyncio
async def test_card_callback_updates_card_to_final_state() -> None:
    """approve 回调后，应调用 client.update_card 覆盖原卡片为终态。"""
    from unittest.mock import AsyncMock

    from agent_hub.pilot import (
        Approval,
        ApprovalStatus,
        ApprovalTargetType,
    )

    orchestrator, repo = _make_orchestrator()
    # 预置一个 REQUESTED 状态的审批，带 channel_message_id
    approval = Approval(
        workspace_id="ws_test",
        task_id="task_test",
        target_type=ApprovalTargetType.PLAN,
        target_id="plan_test",
        requester_id="ou_requester",
        approvers=["ou_approver"],
        reason="生成 PPT 计划审批",
        preview="共 3 步，含写入操作",
        status=ApprovalStatus.REQUESTED,
        channel_message_id="om_card_msg_1",
    )
    await repo.save(approval, expected_version=None)

    # commands mock：decide_approval 成功即可；update_card 路径由 service 自己走 repo
    commands = AsyncMock()
    commands.decide_approval = AsyncMock(return_value={"approval_status": "approved"})

    fake_client = FakeFeishuClient()
    proc = FeishuWebhookProcessor()
    service = FeishuWebhookService(
        processor=proc,
        orchestrator=orchestrator,
        repository=repo,
        client=fake_client,
        commands=commands,
    )

    result = await service.handle_event_dict(
        _card_action_event(
            approval_id=approval.approval_id,
            decision="approve",
            operator_open_id="ou_approver",
            comment="LGTM",
        )
    )

    assert result.outcome is FeishuWebhookOutcome.CARD_CALLBACK
    # 卡片终态被 update
    assert len(fake_client.updated_cards) == 1
    updated = fake_client.updated_cards[0]
    assert updated["message_id"] == "om_card_msg_1"
    card = updated["card"]
    assert card["header"]["template"] == "green"           # approve → 绿色
    assert "已批准" in card["elements"][0]["text"]["content"]
    assert "ou_approver" in card["elements"][0]["text"]["content"]
    assert "LGTM" in card["elements"][0]["text"]["content"]


@pytest.mark.asyncio
async def test_card_callback_reject_updates_card_red() -> None:
    """reject 回调后卡片终态应为红色。"""
    from unittest.mock import AsyncMock

    from agent_hub.pilot import (
        Approval,
        ApprovalStatus,
        ApprovalTargetType,
    )

    orchestrator, repo = _make_orchestrator()
    approval = Approval(
        workspace_id="ws_test2",
        task_id="task_test2",
        target_type=ApprovalTargetType.PLAN,
        target_id="plan_test2",
        requester_id="ou_requester",
        approvers=["ou_approver"],
        reason="风险步骤审批",
        status=ApprovalStatus.REQUESTED,
        channel_message_id="om_card_msg_2",
    )
    await repo.save(approval, expected_version=None)

    commands = AsyncMock()
    commands.decide_approval = AsyncMock(return_value={"approval_status": "rejected"})

    fake_client = FakeFeishuClient()
    proc = FeishuWebhookProcessor()
    service = FeishuWebhookService(
        processor=proc,
        orchestrator=orchestrator,
        repository=repo,
        client=fake_client,
        commands=commands,
    )

    result = await service.handle_event_dict(
        _card_action_event(
            approval_id=approval.approval_id,
            decision="reject",
            operator_open_id="ou_approver",
            comment="",
            event_id="evt-reject-1",
        )
    )

    assert result.outcome is FeishuWebhookOutcome.CARD_CALLBACK
    assert len(fake_client.updated_cards) == 1
    card = fake_client.updated_cards[0]["card"]
    assert card["header"]["template"] == "red"
    assert "已拒绝" in card["elements"][0]["text"]["content"]


@pytest.mark.asyncio
async def test_card_callback_no_channel_message_id_skips_update() -> None:
    """approval 没有 channel_message_id 时，不调用 update_card。"""
    from unittest.mock import AsyncMock

    orchestrator, repo = _make_orchestrator()
    proc = FeishuWebhookProcessor()
    commands = AsyncMock()
    commands.decide_approval = AsyncMock(return_value={"status": "approved"})
    fake_client = FakeFeishuClient()
    service = FeishuWebhookService(
        processor=proc,
        orchestrator=orchestrator,
        repository=repo,
        client=fake_client,
        commands=commands,
    )

    result = await service.handle_event_dict(
        _card_action_event(approval_id="no_such_approval", event_id="evt-no-card")
    )

    assert result.outcome is FeishuWebhookOutcome.CARD_CALLBACK
    # 没找到 approval / 没有 channel_message_id → 不调用 update_card
    assert len(fake_client.updated_cards) == 0

