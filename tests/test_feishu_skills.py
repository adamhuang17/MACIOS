"""``register_feishu_skills`` 注册的技能契约与 invoke 行为测试。"""

from __future__ import annotations

import json

import pytest

from agent_hub.connectors.feishu import FakeFeishuClient
from agent_hub.pilot.skills import (
    SkillInvocation,
    SkillRegistry,
    register_feishu_skills,
)


def _build_registry(
    client: FakeFeishuClient,
    *,
    default_receive_open_id: str = "",
) -> SkillRegistry:
    registry = SkillRegistry()

    async def reader(_artifact_id: str) -> str:
        return "fake-content"

    register_feishu_skills(
        registry,
        client,
        artifact_reader=reader,
        default_folder_token="folder-x",
        default_receive_open_id=default_receive_open_id,
    )
    return registry


def test_skill_specs_have_correct_metadata() -> None:
    client = FakeFeishuClient()
    registry = _build_registry(client)

    fetch = registry.must_get("feishu.im.fetch_recent").spec
    assert fetch.side_effect == "read"
    assert fetch.requires_approval is False

    create = registry.must_get("feishu.docs.create_doc").spec
    assert create.side_effect == "write"
    assert create.requires_approval is True

    upload = registry.must_get("feishu.drive.upload_file").spec
    assert upload.side_effect == "write"
    assert upload.requires_approval is True

    send = registry.must_get("feishu.im.send_message").spec
    assert send.side_effect == "share"
    assert send.requires_approval is True

    for name in (
        "feishu.im.fetch_recent",
        "feishu.docs.create_doc",
        "feishu.drive.upload_file",
        "feishu.im.send_message",
    ):
        spec = registry.must_get(name).spec
        assert spec.scopes == ["feishu"]
        assert spec.supports_dry_run is True


@pytest.mark.asyncio
async def test_dry_run_does_not_call_client() -> None:
    client = FakeFeishuClient()
    registry = _build_registry(client)

    result = await registry.dry_run(
        SkillInvocation(
            skill_name="feishu.docs.create_doc",
            params={"title": "demo"},
        )
    )
    assert result.success is True
    assert result.dry_run is True
    assert result.output["would_create"] is True
    assert client.created_docs == []


@pytest.mark.asyncio
async def test_create_doc_invokes_client() -> None:
    client = FakeFeishuClient()
    registry = _build_registry(client)

    result = await registry.invoke(
        SkillInvocation(
            skill_name="feishu.docs.create_doc",
            params={"title": "Hello"},
        )
    )
    assert result.success is True
    assert result.artifact_payload is not None
    assert result.artifact_payload["type"] == "feishu_doc"
    assert client.created_docs[0]["title"] == "Hello"
    assert client.created_docs[0]["folder_token"] == "folder-x"


@pytest.mark.asyncio
async def test_create_doc_can_share_to_multiple_open_ids() -> None:
    client = FakeFeishuClient()
    registry = _build_registry(client)

    result = await registry.invoke(
        SkillInvocation(
            skill_name="feishu.docs.create_doc",
            params={
                "title": "共享方案",
                "folder_token": "folder-shared",
                "share_recipient_open_ids": ["ou_leader", "ou_sub_1", "ou_sub_2"],
            },
        )
    )

    assert result.success is True
    document_id = result.output["document_id"]
    assert client.created_docs[0]["folder_token"] == "folder-shared"
    assert client.shared_files == [
        {
            "file_token": document_id,
            "member_open_id": "ou_leader",
            "perm": "edit",
            "need_notification": True,
            "file_type": "docx",
        },
        {
            "file_token": document_id,
            "member_open_id": "ou_sub_1",
            "perm": "edit",
            "need_notification": True,
            "file_type": "docx",
        },
        {
            "file_token": document_id,
            "member_open_id": "ou_sub_2",
            "perm": "edit",
            "need_notification": True,
            "file_type": "docx",
        },
    ]
    assert result.artifact_payload["metadata"]["shared_open_ids"] == [
        "ou_leader",
        "ou_sub_1",
        "ou_sub_2",
    ]


@pytest.mark.asyncio
async def test_upload_file_uses_input_artifact_content() -> None:
    client = FakeFeishuClient()
    registry = _build_registry(client)

    result = await registry.invoke(
        SkillInvocation(
            skill_name="feishu.drive.upload_file",
            params={
                "file_name": "brief.md",
                "input_artifacts": {
                    "brief": {
                        "artifact_id": "art-1",
                        "type": "project_brief_md",
                        "content": "# Hello\nbody",
                    }
                },
            },
        )
    )
    assert result.success is True
    assert result.artifact_payload is not None
    assert result.output["byte_size"] > 0
    upload_call = client.uploaded_files[0]
    assert upload_call["file_name"] == "brief.md"
    assert upload_call["parent_node"] == "folder-x"


@pytest.mark.asyncio
async def test_upload_file_falls_back_to_artifact_reader() -> None:
    client = FakeFeishuClient()
    registry = _build_registry(client)

    result = await registry.invoke(
        SkillInvocation(
            skill_name="feishu.drive.upload_file",
            params={"file_name": "x.txt", "artifact_id": "abc"},
        )
    )
    assert result.success is True
    assert client.uploaded_files[0]["size"] == len(b"fake-content")


@pytest.mark.asyncio
async def test_send_message_encodes_text_payload() -> None:
    client = FakeFeishuClient()
    registry = _build_registry(client)

    result = await registry.invoke(
        SkillInvocation(
            skill_name="feishu.im.send_message",
            params={"chat_id": "oc_1", "text": "你好"},
        )
    )
    assert result.success is True
    assert result.artifact_payload is not None
    sent = client.sent_messages[0]
    assert sent["receive_id"] == "oc_1"
    payload = json.loads(sent["content"])
    assert payload == {"text": "你好"}


@pytest.mark.asyncio
async def test_send_message_builds_summary_from_share_artifact() -> None:
    client = FakeFeishuClient()
    registry = _build_registry(client)

    result = await registry.invoke(
        SkillInvocation(
            skill_name="feishu.im.send_message",
            params={
                "chat_id": "oc_1",
                "title": "自我介绍 PPT 已完成",
                "input_artifacts": {
                    "share": {
                        "artifact_id": "art-share",
                        "type": "share_link",
                        "title": "deck.pptx share link",
                        "content": {
                            "file_name": "deck.pptx",
                            "share_url": "https://open.feishu.cn/share/demo",
                            "file_token": "file_demo",
                        },
                        "metadata": {"file_token": "file_demo"},
                    }
                },
            },
        )
    )

    assert result.success is True
    assert result.artifact_payload is not None
    payload = json.loads(client.sent_messages[0]["content"])
    assert payload["text"] == (
        "自我介绍 PPT 已完成\n"
        "分享链接：https://open.feishu.cn/share/demo\n"
        "（飞书已单独推送一条「与我共享」通知，可直接点击打开）"
    )
    assert result.artifact_payload["content"] == payload["text"]


@pytest.mark.asyncio
async def test_send_message_uses_default_open_id_when_chat_id_missing() -> None:
    client = FakeFeishuClient()
    registry = _build_registry(client, default_receive_open_id="ou_admin")

    result = await registry.invoke(
        SkillInvocation(
            skill_name="feishu.im.send_message",
            params={
                "title": "PPT 已完成",
                "text": "结果已生成",
            },
        )
    )

    assert result.success is True, result.error
    assert result.output["receive_id"] == "ou_admin"
    assert result.output["receive_id_type"] == "open_id"
    sent = client.sent_messages[0]
    assert sent["receive_id"] == "ou_admin"
    assert sent["receive_id_type"] == "open_id"


@pytest.mark.asyncio
async def test_fetch_recent_returns_context_bundle() -> None:
    client = FakeFeishuClient()
    # 注入一些历史消息
    from datetime import UTC, datetime

    from agent_hub.connectors.feishu import FeishuMessageRecord

    client.history["oc_1"] = [
        FeishuMessageRecord(
            message_id="om_a",
            chat_id="oc_1",
            sender_id="ou_x",
            message_type="text",
            text="我们要做飞书 PPT",
            create_time=datetime.now(UTC),
        )
    ]
    registry = _build_registry(client)
    result = await registry.invoke(
        SkillInvocation(
            skill_name="feishu.im.fetch_recent",
            params={"chat_id": "oc_1", "page_size": 10},
        )
    )
    assert result.success is True
    bundle = result.artifact_payload["content"]  # type: ignore[index]
    assert bundle["messages"][0]["text"] == "我们要做飞书 PPT"
    assert bundle["context_scope"]["chat_id"] == "oc_1"


@pytest.mark.asyncio
async def test_send_message_validates_required_params() -> None:
    client = FakeFeishuClient()
    registry = _build_registry(client)
    result = await registry.invoke(
        SkillInvocation(skill_name="feishu.im.send_message", params={}),
    )
    assert result.success is False
    assert "chat_id" in (result.error or "")
