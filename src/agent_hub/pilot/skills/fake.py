"""M1 端到端 Fake 技能集合。

无需任何外部依赖即可跑通 collect_context → generate_brief →
generate_slidespec → render_pptx → upload_share → send_summary
完整链路；用于演示、回归测试与离线 demo mode。
"""

from __future__ import annotations

from agent_hub.pilot.domain.enums import ArtifactType
from agent_hub.pilot.skills.registry import (
    SkillInvocation,
    SkillRegistry,
    SkillResult,
    SkillSpec,
)


async def _fake_im_fetch(inv: SkillInvocation) -> SkillResult:
    text = inv.params.get("origin_text", "")
    return SkillResult(
        skill_name=inv.skill_name,
        success=True,
        output={"summary": f"context for: {text[:64]}"},
        artifact_payload={
            "type": ArtifactType.CONTEXT_BUNDLE.value,
            "title": "Context Bundle",
            "mime_type": "application/json",
            "metadata": {"origin_text": text},
        } if not inv.dry_run else None,
    )


async def _fake_brief_generate(inv: SkillInvocation) -> SkillResult:
    title = inv.params.get("title", "Project Brief")
    return SkillResult(
        skill_name=inv.skill_name,
        success=True,
        output={"title": title},
        artifact_payload={
            "type": ArtifactType.PROJECT_BRIEF_MD.value,
            "title": title,
            "mime_type": "text/markdown",
            "metadata": {"sections": ["背景", "目标", "受众", "交付", "下一步"]},
        } if not inv.dry_run else None,
    )


async def _fake_slidespec_generate(inv: SkillInvocation) -> SkillResult:
    return SkillResult(
        skill_name=inv.skill_name,
        success=True,
        output={"slides": 8},
        artifact_payload={
            "type": ArtifactType.SLIDE_SPEC_JSON.value,
            "title": "SlideSpec",
            "mime_type": "application/json",
            "metadata": {"slide_count": 8},
        } if not inv.dry_run else None,
    )


async def _fake_pptx_render(inv: SkillInvocation) -> SkillResult:
    return SkillResult(
        skill_name=inv.skill_name,
        success=True,
        output={"path": "/tmp/fake.pptx"},
        artifact_payload={
            "type": ArtifactType.PPTX.value,
            "title": "Deck.pptx",
            "uri": "file:///tmp/fake.pptx",
            "mime_type": (
                "application/vnd.openxmlformats-officedocument"
                ".presentationml.presentation"
            ),
        } if not inv.dry_run else None,
    )


async def _fake_drive_upload(inv: SkillInvocation) -> SkillResult:
    return SkillResult(
        skill_name=inv.skill_name,
        success=True,
        output={"drive_token": "fake_token_xyz"},
        artifact_payload={
            "type": ArtifactType.DRIVE_FILE.value,
            "title": "Deck.pptx (drive)",
            "uri": "https://fake.feishu/drive/fake_token_xyz",
            "metadata": {"feishu_token": "fake_token_xyz"},
        } if not inv.dry_run else None,
    )


async def _fake_slides_share(inv: SkillInvocation) -> SkillResult:
    return SkillResult(
        skill_name=inv.skill_name,
        success=True,
        output={"share_url": "https://fake.feishu/share/abc"},
        artifact_payload={
            "type": ArtifactType.SHARE_LINK.value,
            "title": "Share Link",
            "uri": "https://fake.feishu/share/abc",
            "metadata": {"share_url": "https://fake.feishu/share/abc"},
        } if not inv.dry_run else None,
    )


async def _fake_im_send_summary(inv: SkillInvocation) -> SkillResult:
    return SkillResult(
        skill_name=inv.skill_name,
        success=True,
        output={"sent": True},
        artifact_payload={
            "type": ArtifactType.SUMMARY.value,
            "title": "Task Summary",
            "mime_type": "text/plain",
            "metadata": {"channel": inv.params.get("channel", "fake_im")},
        } if not inv.dry_run else None,
    )


_FAKE_SPECS: list[tuple[SkillSpec, object]] = [
    (
        SkillSpec(
            skill_name="fake_im_fetch",
            description="拉取群聊上下文（fake）。",
            side_effect="read",
            requires_approval=False,
            supports_dry_run=True,
            timeout_ms=5_000,
        ),
        _fake_im_fetch,
    ),
    (
        SkillSpec(
            skill_name="fake_brief_generate",
            description="生成 Project Brief（fake）。",
            side_effect="read",
            requires_approval=False,
            supports_dry_run=True,
            timeout_ms=10_000,
        ),
        _fake_brief_generate,
    ),
    (
        SkillSpec(
            skill_name="fake_slidespec_generate",
            description="生成 SlideSpec（fake）。",
            side_effect="read",
            requires_approval=False,
            supports_dry_run=True,
            timeout_ms=10_000,
        ),
        _fake_slidespec_generate,
    ),
    (
        SkillSpec(
            skill_name="fake_pptx_render",
            description="渲染 PPTX 文件（fake）。",
            side_effect="write",
            requires_approval=True,
            supports_dry_run=True,
            timeout_ms=15_000,
        ),
        _fake_pptx_render,
    ),
    (
        SkillSpec(
            skill_name="fake_drive_upload",
            description="上传到飞书 Drive（fake）。",
            side_effect="write",
            requires_approval=True,
            supports_dry_run=True,
            timeout_ms=15_000,
        ),
        _fake_drive_upload,
    ),
    (
        SkillSpec(
            skill_name="fake_slides_share",
            description="生成分享链接（fake）。",
            side_effect="share",
            requires_approval=True,
            supports_dry_run=True,
            timeout_ms=10_000,
        ),
        _fake_slides_share,
    ),
    (
        SkillSpec(
            skill_name="fake_im_send_summary",
            description="把结果回传到群聊（fake）。",
            side_effect="share",
            requires_approval=True,
            supports_dry_run=True,
            timeout_ms=10_000,
        ),
        _fake_im_send_summary,
    ),
]


def register_fake_skills(registry: SkillRegistry) -> None:
    """把 M1 默认 fake 技能注册到指定 registry。"""
    for spec, func in _FAKE_SPECS:
        registry.register(spec, func)  # type: ignore[arg-type]


__all__ = ["register_fake_skills"]
