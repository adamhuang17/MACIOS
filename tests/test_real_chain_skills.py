from __future__ import annotations

import pytest

from agent_hub.connectors.feishu import FakeFeishuClient
from agent_hub.pilot.skills.real_chain import register_real_chain_skills
from agent_hub.pilot.skills.registry import SkillInvocation, SkillRegistry


@pytest.mark.asyncio
async def test_upload_share_includes_share_metadata_and_auto_shares() -> None:
    client = FakeFeishuClient()
    registry = SkillRegistry()

    async def reader(_artifact_id: str) -> bytes:
        return b"pptx-content"

    register_real_chain_skills(
        registry,
        feishu_client=client,
        artifact_reader=reader,
        default_folder_token="folder-x",
        admin_open_id="ou_admin",
    )

    result = await registry.invoke(
        SkillInvocation(
            skill_name="real.drive.upload_share",
            params={
                "file_name": "deck.pptx",
                "input_artifacts": {
                    "pptx": {
                        "artifact_id": "art-pptx",
                        "type": "pptx",
                        "content": b"pptx-content",
                    }
                },
            },
        )
    )

    assert result.success is True
    assert result.artifact_payload is not None
    metadata = result.artifact_payload["metadata"]
    content = result.artifact_payload["content"]
    assert metadata["folder_token"] == "folder-x"
    assert metadata["feishu_token"] == content["file_token"]
    assert metadata["share_url"] == content["share_url"]
    assert content["share_url"].startswith("https://fake.feishu.local/file/")
    assert client.shared_files == [
        {
            "file_token": content["file_token"],
            "member_open_id": "ou_admin",
            "perm": "edit",
            "need_notification": True,
        }
    ]


@pytest.mark.asyncio
async def test_upload_share_falls_back_to_drive_url_template() -> None:
    """飞书 Drive upload_all 不返回 download_url 时，应基于 file_token 拼出可访问链接。"""
    from agent_hub.connectors.feishu.client import FeishuFileUploaded

    class _NoUrlClient(FakeFeishuClient):
        async def upload_file(self, *, file_name, content, parent_type="explorer", parent_node=None):  # type: ignore[override]
            base = await super().upload_file(
                file_name=file_name,
                content=content,
                parent_type=parent_type,
                parent_node=parent_node,
            )
            # 模拟真实飞书：download_url 字段缺失/为空
            return FeishuFileUploaded(
                file_token=base.file_token,
                file_name=base.file_name,
                size=base.size,
                folder_token=base.folder_token,
                download_url="",
            )

    client = _NoUrlClient()
    registry = SkillRegistry()

    async def reader(_artifact_id: str) -> bytes:
        return b"pptx-content"

    register_real_chain_skills(
        registry,
        feishu_client=client,
        artifact_reader=reader,
        default_folder_token="folder-x",
        admin_open_id="",
        drive_url_template="https://example.feishu.cn/file/{file_token}",
    )

    result = await registry.invoke(
        SkillInvocation(
            skill_name="real.drive.upload_share",
            params={
                "file_name": "deck.pptx",
                "input_artifacts": {
                    "pptx": {"artifact_id": "art-pptx", "type": "pptx", "content": b"pptx-content"}
                },
            },
        )
    )

    assert result.success is True
    content = result.artifact_payload["content"]
    metadata = result.artifact_payload["metadata"]
    expected = f"https://example.feishu.cn/file/{content['file_token']}"
    assert content["share_url"] == expected
    assert metadata["share_url"] == expected
    assert result.artifact_payload["uri"] == expected


@pytest.mark.asyncio
async def test_upload_share_accepts_brief_upstream() -> None:
    """DOC_ONLY 链路里上游是 brief.md，upload_share 应能直接消费。"""
    client = FakeFeishuClient()
    registry = SkillRegistry()

    async def reader(_artifact_id: str) -> bytes:
        return b""

    register_real_chain_skills(
        registry,
        feishu_client=client,
        artifact_reader=reader,
        default_folder_token="folder-x",
    )

    result = await registry.invoke(
        SkillInvocation(
            skill_name="real.drive.upload_share",
            params={
                "file_name": "intro.md",
                "input_artifacts": {
                    "brief": {
                        "artifact_id": "art-brief",
                        "type": "project_brief_md",
                        "content": "# Hello\n自我介绍",
                    }
                },
            },
        )
    )

    assert result.success is True, result.error
    assert client.uploaded_files[0]["file_name"] == "intro.md"
    assert client.uploaded_files[0]["size"] == len("# Hello\n自我介绍".encode("utf-8"))