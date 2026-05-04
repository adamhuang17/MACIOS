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
        mentions=[{"id": {"open_id": "ou_bot"}}],
    )
    result = await proc.handle_payload(with_mention)
    assert result.outcome is FeishuWebhookOutcome.ACCEPTED
    assert result.message is not None
    assert result.message.bot_mentioned is True


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
