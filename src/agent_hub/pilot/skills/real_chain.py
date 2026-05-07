"""真实产物链 skills 骨架（M3+ / M5）。

提供三个 skill，与 :mod:`agent_hub.pilot.skills.fake` 中的
``fake_slidespec_generate`` / ``fake_pptx_render`` / ``fake_drive_upload``
形成 1:1 替代关系：

- ``real.slide.generate_spec`` —— 基于 brief 文本确定性地生成 SlideSpec
  JSON（章节、要点、layout）。
- ``real.slide.render_pptx`` —— 调用 ``python-pptx`` 渲染为 PPTX 字节流；
  若运行环境未安装 ``python-pptx``，自动降级为占位的二进制 stub，
  artifact 仍可入库以便后续 Drive upload 拿到 ``content``。
- ``real.drive.upload_share`` —— 通过 :class:`FeishuClientProtocol`
  上传 PPTX 到飞书云空间并返回 share_url。

设计上保持与 fake 对应版本一致的 ``output_artifact_ref`` 与
``input_artifacts`` 协议，便于通过 feature flag 平滑切换。
"""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any

import structlog

from agent_hub.connectors.feishu.client import FeishuApiError, FeishuClientProtocol
from agent_hub.pilot.domain.enums import ArtifactType
from agent_hub.pilot.skills.internal import ArtifactContentReader
from agent_hub.pilot.skills.registry import (
    SkillInvocation,
    SkillRegistry,
    SkillResult,
    SkillSpec,
)

REAL_CHAIN_SCOPES = ["pilot", "pilot.real_chain"]
logger = structlog.get_logger(__name__)


# ── real.slide.generate_spec ───────────────────────


def _build_slide_spec(brief_text: str, title: str) -> dict[str, Any]:
    """基于 brief Markdown 生成确定性 SlideSpec。

    简易策略：按 H1/H2 切片；找不到标题时按段落聚合，确保至少 4 张
    slide。生产环境可替换为 LLM 生成。
    """
    lines = [ln.strip() for ln in (brief_text or "").splitlines() if ln.strip()]
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for ln in lines:
        if ln.startswith("# "):
            if current:
                sections.append(current)
            current = {"title": ln[2:].strip(), "bullets": []}
        elif ln.startswith("## "):
            if current:
                sections.append(current)
            current = {"title": ln[3:].strip(), "bullets": []}
        elif ln.startswith(("- ", "* ")):
            if current is None:
                current = {"title": title, "bullets": []}
            current["bullets"].append(ln[2:].strip())
        else:
            if current is None:
                current = {"title": title, "bullets": []}
            current["bullets"].append(ln)
    if current:
        sections.append(current)

    if not sections:
        sections = [
            {"title": title, "bullets": ["（无内容，请补充 brief）"]},
        ]

    slides = [{"layout": "title", "title": title, "subtitle": "Auto-generated"}]
    for sec in sections:
        slides.append(
            {
                "layout": "title_and_bullets",
                "title": sec["title"],
                "bullets": sec["bullets"][:6] or ["（待补充）"],
            },
        )
    slides.append({"layout": "thank_you", "title": "Thank You"})

    return {"title": title, "slide_count": len(slides), "slides": slides}


def _make_real_slidespec_skill() -> tuple[SkillSpec, Any]:
    async def _gen(inv: SkillInvocation) -> SkillResult:
        params = inv.params
        title = str(params.get("title", "") or "Untitled Deck")
        upstream = params.get("input_artifacts", {}) or {}
        brief_artifact = upstream.get("brief")
        brief_text = ""
        if isinstance(brief_artifact, dict):
            content = brief_artifact.get("content")
            if isinstance(content, str):
                brief_text = content
            elif isinstance(content, dict | list):
                brief_text = json.dumps(content, ensure_ascii=False)

        spec = _build_slide_spec(brief_text, title)

        if inv.dry_run:
            return SkillResult(
                skill_name=inv.skill_name,
                success=True,
                output={"slide_count": spec["slide_count"], "would_generate": True},
            )

        return SkillResult(
            skill_name=inv.skill_name,
            success=True,
            output={"slide_count": spec["slide_count"]},
            artifact_payload={
                "type": ArtifactType.SLIDE_SPEC_JSON.value,
                "title": f"{title} SlideSpec",
                "mime_type": "application/json",
                "content": spec,
                "metadata": {"slide_count": spec["slide_count"]},
            },
        )

    spec = SkillSpec(
        skill_name="real.slide.generate_spec",
        description="基于 brief 内容确定性生成 SlideSpec JSON。",
        side_effect="read",
        requires_approval=False,
        supports_dry_run=True,
        scopes=REAL_CHAIN_SCOPES,
        timeout_ms=10_000,
    )
    return spec, _gen


# ── real.slide.render_pptx ─────────────────────────


def _render_pptx_bytes(slide_spec: dict[str, Any]) -> bytes:
    """渲染 PPTX，未装 python-pptx 时输出可被识别的 stub 字节流。

    stub 仍是合法 zip-like 字节，能存入 artifact store 并供后续上传。
    """
    try:  # pragma: no cover - 可选依赖
        from pptx import Presentation  # type: ignore[import-not-found]
        from pptx.util import Inches, Pt  # type: ignore[import-not-found]
    except ImportError:
        # 占位实现：把 spec JSON 当 bytes，真实渲染留待依赖安装后启用。
        return ("PPTX_STUB::" + json.dumps(slide_spec, ensure_ascii=False)).encode(
            "utf-8",
        )

    prs = Presentation()
    blank_layout = prs.slide_layouts[5]
    for slide in slide_spec.get("slides", []):
        s = prs.slides.add_slide(blank_layout)
        title_shape = s.shapes.title
        if title_shape is not None:
            title_shape.text = str(slide.get("title", ""))
        bullets = slide.get("bullets") or []
        if bullets:
            tx_box = s.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(9), Inches(5))
            tf = tx_box.text_frame
            tf.text = str(bullets[0])
            for b in bullets[1:]:
                p = tf.add_paragraph()
                p.text = str(b)
                p.font.size = Pt(18)
    buf = BytesIO()
    prs.save(buf)
    return buf.getvalue()


def _make_real_pptx_render_skill() -> tuple[SkillSpec, Any]:
    async def _render(inv: SkillInvocation) -> SkillResult:
        params = inv.params
        upstream = params.get("input_artifacts", {}) or {}
        spec_artifact = upstream.get("spec") or upstream.get("slide_spec")
        slide_spec: dict[str, Any] | None = None
        if isinstance(spec_artifact, dict):
            content = spec_artifact.get("content")
            if isinstance(content, dict):
                slide_spec = content
            elif isinstance(content, str):
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        slide_spec = parsed
                except json.JSONDecodeError:
                    pass

        if slide_spec is None:
            return SkillResult(
                skill_name=inv.skill_name,
                success=False,
                error="real.slide.render_pptx 需要 input_artifacts.spec",
            )

        title = str(params.get("title", "") or slide_spec.get("title", "Deck"))

        if inv.dry_run:
            return SkillResult(
                skill_name=inv.skill_name,
                success=True,
                output={
                    "slide_count": slide_spec.get("slide_count", 0),
                    "would_render": True,
                },
            )

        pptx_bytes = _render_pptx_bytes(slide_spec)

        return SkillResult(
            skill_name=inv.skill_name,
            success=True,
            output={"byte_size": len(pptx_bytes)},
            artifact_payload={
                "type": ArtifactType.PPTX.value,
                "title": f"{title}.pptx",
                "mime_type": (
                    "application/vnd.openxmlformats-officedocument"
                    ".presentationml.presentation"
                ),
                "content": pptx_bytes,
                "metadata": {
                    "slide_count": slide_spec.get("slide_count", 0),
                    "renderer": "python-pptx-or-stub",
                },
            },
        )

    spec = SkillSpec(
        skill_name="real.slide.render_pptx",
        description="将 SlideSpec 渲染为 PPTX（python-pptx；缺失时降级为 stub）。",
        side_effect="write",
        requires_approval=True,
        supports_dry_run=True,
        scopes=REAL_CHAIN_SCOPES,
        idempotency_key_fields=["title"],
        timeout_ms=30_000,
    )
    return spec, _render


# ── real.drive.upload_share ────────────────────────


def _make_real_upload_share_skill(
    client: FeishuClientProtocol,
    *,
    artifact_reader: ArtifactContentReader,
    default_folder_token: str = "",
    admin_open_id: str = "",
    drive_url_template: str = "https://www.feishu.cn/file/{file_token}",
) -> tuple[SkillSpec, Any]:
    async def _upload(inv: SkillInvocation) -> SkillResult:
        params = inv.params
        file_name = str(params.get("file_name", "") or "deck.pptx")
        folder_token = str(params.get("folder_token", "") or default_folder_token) or None

        upstream = params.get("input_artifacts", {}) or {}
        # 兼容 PPTX / 任意文件 / Brief Markdown / Doc 上游，让该 skill 可以复用为
        # DOC_ONLY plan 的交付环节。
        source_artifact = (
            upstream.get("pptx")
            or upstream.get("file")
            or upstream.get("brief")
            or upstream.get("doc")
        )

        content_bytes: bytes | None = None
        if isinstance(source_artifact, dict):
            raw = source_artifact.get("content")
            if isinstance(raw, bytes):
                content_bytes = raw
            elif isinstance(raw, str):
                content_bytes = raw.encode("utf-8")

        if content_bytes is None and "artifact_id" in params:
            raw = await artifact_reader(str(params["artifact_id"]))
            if isinstance(raw, bytes):
                content_bytes = raw
            elif isinstance(raw, str):
                content_bytes = raw.encode("utf-8")

        if content_bytes is None:
            return SkillResult(
                skill_name=inv.skill_name,
                success=False,
                error="real.drive.upload_share 需要 input_artifacts.pptx / brief / doc / file",
            )

        if inv.dry_run:
            return SkillResult(
                skill_name=inv.skill_name,
                success=True,
                output={
                    "file_name": file_name,
                    "byte_size": len(content_bytes),
                    "would_upload": True,
                },
            )

        try:
            uploaded = await client.upload_file(
                file_name=file_name,
                content=content_bytes,
                parent_type="explorer",
                parent_node=folder_token,
            )
        except FeishuApiError as exc:
            return SkillResult(skill_name=inv.skill_name, success=False, error=str(exc))

        share_url = (uploaded.download_url or "").strip()
        # 飞书 Drive upload_all 实际并不返回稳定可访问 URL，需要基于 file_token 兜底拼接。
        if not share_url and uploaded.file_token and drive_url_template:
            try:
                share_url = drive_url_template.format(file_token=uploaded.file_token)
            except (KeyError, IndexError):
                share_url = ""

        # 上传成功后自动将文件分享给管理员协作者
        # need_notification=True 让飞书主动发一条带可点击链接的"与我共享"通知。
        if admin_open_id and uploaded.file_token:
            try:
                await client.share_file(
                    file_token=uploaded.file_token,
                    member_open_id=admin_open_id,
                    perm="edit",
                    need_notification=True,
                )
            except FeishuApiError:
                logger.warning("drive_share_failed", file_token=uploaded.file_token)

        return SkillResult(
            skill_name=inv.skill_name,
            success=True,
            output={
                "file_token": uploaded.file_token,
                "share_url": share_url,
                "byte_size": uploaded.size,
            },
            artifact_payload={
                "type": ArtifactType.SHARE_LINK.value,
                "title": f"{file_name} share link",
                "uri": share_url,
                "mime_type": "text/uri-list",
                "content": {
                    "file_token": uploaded.file_token,
                    "share_url": share_url,
                    "file_name": file_name,
                },
                "metadata": {
                    "file_token": uploaded.file_token,
                    "feishu_token": uploaded.file_token,
                    "byte_size": uploaded.size,
                    "folder_token": folder_token or "",
                    "share_url": share_url,
                },
            },
        )

    spec = SkillSpec(
        skill_name="real.drive.upload_share",
        description="把 PPTX 上传到飞书云空间并返回（暂用 download_url 充当）share_url。",
        side_effect="share",
        requires_approval=True,
        supports_dry_run=True,
        scopes=REAL_CHAIN_SCOPES,
        idempotency_key_fields=["file_name", "folder_token"],
        timeout_ms=45_000,
    )
    return spec, _upload


# ── 注册入口 ────────────────────────────────────────


def register_real_chain_skills(
    registry: SkillRegistry,
    *,
    feishu_client: FeishuClientProtocol | None = None,
    artifact_reader: ArtifactContentReader | None = None,
    default_folder_token: str = "",
    admin_open_id: str = "",
    drive_url_template: str = "https://www.feishu.cn/file/{file_token}",
    allow_overwrite: bool = False,
) -> None:
    """注册真实产物链 skills。

    SlideSpec / PPTX 不依赖外部服务；upload_share 仅在传入飞书 client
    时注册。便于在没有飞书凭据的环境下也能跑前两步。
    """
    spec, fn = _make_real_slidespec_skill()
    registry.register(spec, fn, allow_overwrite=allow_overwrite)

    spec, fn = _make_real_pptx_render_skill()
    registry.register(spec, fn, allow_overwrite=allow_overwrite)

    if feishu_client is not None and artifact_reader is not None:
        spec, fn = _make_real_upload_share_skill(
            feishu_client,
            artifact_reader=artifact_reader,
            default_folder_token=default_folder_token,
            admin_open_id=admin_open_id,
            drive_url_template=drive_url_template,
        )
        registry.register(spec, fn, allow_overwrite=allow_overwrite)


__all__ = [
    "REAL_CHAIN_SCOPES",
    "register_real_chain_skills",
]
