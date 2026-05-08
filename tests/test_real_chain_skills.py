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
    assert client.uploaded_files[0]["size"] == len("# Hello\n自我介绍".encode())


# ── real.slide.render_pptx 真实渲染回归 ─────────────


@pytest.mark.asyncio
async def test_render_pptx_produces_real_zip_pptx() -> None:
    """真实链路应输出一个合法的 zip/PPTX 字节流，而非 PPTX_STUB。"""
    import io
    import zipfile

    registry = SkillRegistry()
    register_real_chain_skills(registry, feishu_client=None, artifact_reader=None)

    spec = {
        "title": "季度复盘",
        "slide_count": 2,
        "slides": [
            {"title": "封面", "bullets": ["议题"]},
            {"title": "结果", "bullets": ["指标 A", "指标 B"]},
        ],
    }
    result = await registry.invoke(
        SkillInvocation(
            skill_name="real.slide.render_pptx",
            params={
                "title": "季度复盘",
                "input_artifacts": {
                    "spec": {
                        "artifact_id": "art-spec",
                        "type": "slide_spec",
                        "content": spec,
                    }
                },
            },
        )
    )

    assert result.success is True, result.error
    payload = result.artifact_payload
    assert payload is not None
    pptx_bytes = payload["content"]
    # 1) PPTX 是 zip 容器，前两字节必为 "PK"，且不能是 PPTX_STUB
    assert pptx_bytes[:2] == b"PK"
    assert not pptx_bytes.startswith(b"PPTX_STUB")
    # 2) 能被 zipfile 解开，且包含 [Content_Types].xml
    with zipfile.ZipFile(io.BytesIO(pptx_bytes)) as zf:
        names = zf.namelist()
        assert "[Content_Types].xml" in names
        assert any(n.startswith("ppt/slides/") for n in names)
    # 3) metadata 不应是 stub
    assert payload["metadata"]["is_stub"] is False
    assert payload["metadata"]["renderer"] == "python-pptx"
    assert result.output["is_stub"] is False


@pytest.mark.asyncio
async def test_render_pptx_strict_when_dependency_missing(monkeypatch) -> None:
    """模拟 python-pptx 缺失：默认严格抛错，不再悄悄交付 stub。"""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name.startswith("pptx"):
            raise ImportError(f"simulated missing: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.delenv("PILOT_ALLOW_PPTX_STUB", raising=False)

    registry = SkillRegistry()
    register_real_chain_skills(registry, feishu_client=None, artifact_reader=None)

    result = await registry.invoke(
        SkillInvocation(
            skill_name="real.slide.render_pptx",
            params={
                "title": "x",
                "input_artifacts": {
                    "spec": {
                        "artifact_id": "a",
                        "type": "slide_spec",
                        "content": {"slides": [{"title": "t", "bullets": []}]},
                    }
                },
            },
        )
    )
    assert result.success is False
    assert "python-pptx" in (result.error or "")


@pytest.mark.asyncio
async def test_render_pptx_explicit_stub_fallback(monkeypatch) -> None:
    """显式 PILOT_ALLOW_PPTX_STUB=true 时，缺依赖应降级为 stub 并标注。"""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name.startswith("pptx"):
            raise ImportError(f"simulated missing: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.setenv("PILOT_ALLOW_PPTX_STUB", "true")

    registry = SkillRegistry()
    register_real_chain_skills(registry, feishu_client=None, artifact_reader=None)

    result = await registry.invoke(
        SkillInvocation(
            skill_name="real.slide.render_pptx",
            params={
                "title": "x",
                "input_artifacts": {
                    "spec": {
                        "artifact_id": "a",
                        "type": "slide_spec",
                        "content": {"slides": [{"title": "t", "bullets": []}]},
                    }
                },
            },
        )
    )
    assert result.success is True
    payload = result.artifact_payload
    assert payload is not None
    assert payload["content"].startswith(b"PPTX_STUB::")
    assert payload["metadata"]["is_stub"] is True
    assert payload["metadata"]["renderer"] == "pptx-stub"
