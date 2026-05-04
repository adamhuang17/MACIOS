"""M3 内部文档生成技能。

技能仍然遵守 SkillRegistry 的契约：
- 不持有状态、不直接读写仓库；
- 通过闭包注入 ``BriefGenerator`` / ``ArtifactContentReader``；
- 输入 artifact 由 ExecutionEngine 的 StepInputResolver 注入到
  ``params["input_artifacts"]`` 中。

输出 Artifact 约定（payload 关键字段）：
- ``type``: 强类型枚举字符串，如 ``"context_bundle"`` / ``"project_brief_md"``；
- ``title`` / ``mime_type`` / ``metadata``: 元数据；
- ``content``: 真正写盘的内容（``str`` / ``dict``）；
- ``parent_artifact_id`` / ``artifact_version``: 用于版本链。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from agent_hub.pilot.domain.enums import ArtifactType
from agent_hub.pilot.skills.registry import (
    SkillInvocation,
    SkillRegistry,
    SkillResult,
    SkillSpec,
)

# ── 协议 ─────────────────────────────────────────────


class BriefGenerator(Protocol):
    """根据 ContextBundle 生成 Project Brief Markdown。"""

    async def generate(
        self,
        *,
        title: str,
        context_bundle: dict[str, Any] | str | None,
        language: str = "zh-CN",
    ) -> str: ...

    async def revise(
        self,
        *,
        base_brief: str,
        revision_instruction: str,
        context_bundle: dict[str, Any] | str | None,
        language: str = "zh-CN",
    ) -> str: ...


ArtifactContentReader = Callable[[str], Awaitable[Any]]
"""根据 ``artifact_id`` 异步读取内容；通常绑定到 ``ArtifactStore``。"""


# ── 默认实现：模板 BriefGenerator ────────────────────


BRIEF_SECTIONS: tuple[str, ...] = (
    "背景",
    "目标",
    "受众",
    "约束",
    "输入",
    "洞察",
    "交付",
    "风险",
    "下一步",
)


class TemplateBriefGenerator:
    """确定性模板生成器，作为离线 / demo / 单测兜底。

    云端 LLM 接入版本可作为 ``BriefGenerator`` 的另一实现，在
    PilotRuntime 通过依赖注入替换。
    """

    sections: tuple[str, ...] = BRIEF_SECTIONS

    async def generate(
        self,
        *,
        title: str,
        context_bundle: dict[str, Any] | str | None,
        language: str = "zh-CN",  # noqa: ARG002
    ) -> str:
        origin, sources = _extract_context(context_bundle)
        lines: list[str] = [f"# {title}", ""]
        if origin:
            lines.extend([
                f"> 原始需求：{origin.strip()}",
                "",
            ])
        for section in self.sections:
            lines.append(f"## {section}")
            lines.append(_default_section_text(section, origin, sources))
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    async def revise(
        self,
        *,
        base_brief: str,
        revision_instruction: str,
        context_bundle: dict[str, Any] | str | None,
        language: str = "zh-CN",  # noqa: ARG002
    ) -> str:
        origin, _ = _extract_context(context_bundle)
        marker = "<!-- revision -->"
        revision_block = (
            f"\n{marker}\n"
            f"## 修订记录\n"
            f"- 指令：{revision_instruction.strip()}\n"
        )
        if origin:
            revision_block += f"- 原始需求：{origin.strip()[:120]}\n"
        if marker in base_brief:
            head, _, _tail = base_brief.partition(marker)
            return head.rstrip() + revision_block
        return base_brief.rstrip() + "\n" + revision_block


def _extract_context(
    context_bundle: dict[str, Any] | str | None,
) -> tuple[str, list[str]]:
    if context_bundle is None:
        return "", []
    if isinstance(context_bundle, str):
        return context_bundle, []
    origin = str(context_bundle.get("origin_text", "") or "")
    sources = [str(x) for x in context_bundle.get("source_refs", []) or []]
    return origin, sources


def _default_section_text(
    section: str,
    origin: str,
    sources: list[str],
) -> str:
    snippet = origin.strip().splitlines()[0] if origin.strip() else ""
    src_note = (
        f"\n\n来源：{', '.join(sources[:3])}" if sources else ""
    )
    if not snippet:
        return f"待补充：{section} 暂无明确输入。{src_note}"
    return f"基于「{snippet[:80]}」补全 {section} 描述。{src_note}"


# ── 技能实现 ────────────────────────────────────────


def _make_context_assemble_skill() -> tuple[SkillSpec, Any]:
    async def _assemble(inv: SkillInvocation) -> SkillResult:
        params = inv.params
        origin = str(params.get("origin_text", "") or "")
        attachments = list(params.get("attachments", []) or [])
        markdown_refs = list(params.get("markdown_refs", []) or [])
        context_scope = dict(params.get("context_scope", {}) or {})

        # input_artifacts 中由 StepInputResolver 注入；M3 默认空，
        # M4 接入飞书后会注入历史消息 / 附件文本。
        upstream = params.get("input_artifacts", {}) or {}
        prior_messages: list[Any] = []
        for ref, art in upstream.items():
            content = art.get("content")
            if content is None:
                continue
            prior_messages.append({
                "ref": ref,
                "artifact_id": art.get("artifact_id"),
                "type": art.get("type"),
                "summary": _summarize_for_context(content),
            })

        bundle: dict[str, Any] = {
            "origin_text": origin,
            "messages": prior_messages,
            "attachments": attachments,
            "markdown_docs": markdown_refs,
            "source_refs": [
                *(f"attachment:{a}" for a in attachments),
                *(f"markdown:{m}" for m in markdown_refs),
            ],
            "context_scope": context_scope,
        }

        if inv.dry_run:
            return SkillResult(
                skill_name=inv.skill_name,
                success=True,
                output={
                    "scope": context_scope,
                    "attachment_count": len(attachments),
                    "markdown_count": len(markdown_refs),
                    "would_assemble": True,
                },
            )

        return SkillResult(
            skill_name=inv.skill_name,
            success=True,
            output={
                "message_count": len(prior_messages),
                "attachment_count": len(attachments),
            },
            artifact_payload={
                "type": ArtifactType.CONTEXT_BUNDLE.value,
                "title": params.get("title", "Context Bundle"),
                "mime_type": "application/json",
                "content": bundle,
                "metadata": {
                    "source_message_id": params.get("source_message_id"),
                    "context_scope": context_scope,
                    "section_counts": {
                        "messages": len(prior_messages),
                        "attachments": len(attachments),
                        "markdown_docs": len(markdown_refs),
                    },
                },
            },
        )

    spec = SkillSpec(
        skill_name="internal.context.assemble",
        description=(
            "把 origin_text + 附件 + Markdown 引用拼装为 ContextBundle Artifact。"
        ),
        side_effect="read",
        requires_approval=False,
        supports_dry_run=True,
        timeout_ms=10_000,
    )
    return spec, _assemble


def _make_brief_generate_skill(
    brief_generator: BriefGenerator,
) -> tuple[SkillSpec, Any]:
    async def _generate(inv: SkillInvocation) -> SkillResult:
        params = inv.params
        title = str(params.get("title", "Project Brief"))
        language = str(params.get("language", "zh-CN"))

        upstream = params.get("input_artifacts", {}) or {}
        ctx_artifact = upstream.get("ctx") or upstream.get("context_bundle")
        ctx_content = ctx_artifact.get("content") if ctx_artifact else None
        source_artifact_ids = [
            art.get("artifact_id")
            for art in upstream.values()
            if art.get("artifact_id")
        ]

        if inv.dry_run:
            return SkillResult(
                skill_name=inv.skill_name,
                success=True,
                output={
                    "title": title,
                    "language": language,
                    "ctx_present": ctx_artifact is not None,
                    "would_generate": True,
                },
            )

        markdown = await brief_generator.generate(
            title=title,
            context_bundle=ctx_content,
            language=language,
        )
        return SkillResult(
            skill_name=inv.skill_name,
            success=True,
            output={"title": title, "char_count": len(markdown)},
            artifact_payload={
                "type": ArtifactType.PROJECT_BRIEF_MD.value,
                "title": f"{title}",
                "mime_type": "text/markdown",
                "content": markdown,
                "metadata": {
                    "template": "default_project_brief_v1",
                    "sections": list(BRIEF_SECTIONS),
                    "source_artifact_ids": source_artifact_ids,
                    "language": language,
                },
            },
        )

    spec = SkillSpec(
        skill_name="internal.brief.generate",
        description="基于 ContextBundle 生成 Project Brief Markdown。",
        side_effect="read",
        requires_approval=False,
        supports_dry_run=True,
        timeout_ms=15_000,
    )
    return spec, _generate


def _make_summary_generate_skill() -> tuple[SkillSpec, Any]:
    """``internal.summary.generate``：把 ContextBundle 浓缩为 Markdown 摘要 Artifact。

    主要服务于 ``PlanProfile.SUMMARY_ONLY``：当用户只需要"会话/事项摘要"
    而非完整 Brief 时使用，纯本地、无副作用。
    """

    async def _summarize(inv: SkillInvocation) -> SkillResult:
        params = inv.params
        title = str(params.get("title", "Conversation Summary"))
        upstream = params.get("input_artifacts", {}) or {}
        ctx_artifact = upstream.get("ctx") or upstream.get("context_bundle")
        ctx_content = ctx_artifact.get("content") if ctx_artifact else None

        if inv.dry_run:
            return SkillResult(
                skill_name=inv.skill_name,
                success=True,
                output={"title": title, "ctx_present": ctx_artifact is not None},
            )

        markdown = _render_summary_markdown(title, ctx_content)
        return SkillResult(
            skill_name=inv.skill_name,
            success=True,
            output={"title": title, "char_count": len(markdown)},
            artifact_payload={
                "type": ArtifactType.SUMMARY.value,
                "title": title,
                "mime_type": "text/markdown",
                "content": markdown,
                "metadata": {"template": "default_summary_v1"},
            },
        )

    spec = SkillSpec(
        skill_name="internal.summary.generate",
        description="把 ContextBundle 浓缩为 Markdown 摘要 Artifact。",
        side_effect="read",
        requires_approval=False,
        supports_dry_run=True,
        timeout_ms=10_000,
    )
    return spec, _summarize


def _render_summary_markdown(
    title: str,
    context_bundle: dict[str, Any] | str | None,
) -> str:
    origin, sources = _extract_context(context_bundle)
    lines = [f"# {title}", ""]
    if origin:
        first_line = origin.strip().splitlines()[0]
        lines.extend([f"- 主要诉求：{first_line[:200]}", ""])
    messages: list[Any] = []
    if isinstance(context_bundle, dict):
        messages = list(context_bundle.get("messages", []) or [])
    if messages:
        lines.append("## 关键消息")
        for msg in messages[:5]:
            summary = str(msg.get("summary") if isinstance(msg, dict) else msg)
            lines.append(f"- {summary[:160]}")
        lines.append("")
    if sources:
        lines.append("## 参考来源")
        for ref in sources[:5]:
            lines.append(f"- {ref}")
        lines.append("")
    if len(lines) <= 2:
        lines.extend(["（暂无可摘要的内容。）", ""])
    return "\n".join(lines).rstrip() + "\n"


def _make_brief_revise_skill(
    brief_generator: BriefGenerator,
    artifact_reader: ArtifactContentReader,
) -> tuple[SkillSpec, Any]:
    async def _revise(inv: SkillInvocation) -> SkillResult:
        params = inv.params
        base_id = params.get("base_brief_artifact_id")
        instruction = str(params.get("revision_instruction", "") or "").strip()
        language = str(params.get("language", "zh-CN"))
        if not base_id:
            return SkillResult(
                skill_name=inv.skill_name,
                success=False,
                error="missing required param: base_brief_artifact_id",
            )
        if not instruction:
            return SkillResult(
                skill_name=inv.skill_name,
                success=False,
                error="missing required param: revision_instruction",
            )

        upstream = params.get("input_artifacts", {}) or {}
        ctx_artifact = upstream.get("ctx") or upstream.get("context_bundle")
        ctx_content = ctx_artifact.get("content") if ctx_artifact else None

        base_content = await artifact_reader(str(base_id))
        if not isinstance(base_content, str):
            return SkillResult(
                skill_name=inv.skill_name,
                success=False,
                error=f"base brief artifact {base_id} 内容不可读",
            )

        if inv.dry_run:
            return SkillResult(
                skill_name=inv.skill_name,
                success=True,
                output={
                    "base_brief_artifact_id": base_id,
                    "instruction_preview": instruction[:80],
                    "would_revise": True,
                },
            )

        revised = await brief_generator.revise(
            base_brief=base_content,
            revision_instruction=instruction,
            context_bundle=ctx_content,
            language=language,
        )
        title = str(params.get("title") or "Project Brief (revised)")
        return SkillResult(
            skill_name=inv.skill_name,
            success=True,
            output={
                "base_brief_artifact_id": base_id,
                "char_count": len(revised),
            },
            artifact_payload={
                "type": ArtifactType.PROJECT_BRIEF_MD.value,
                "title": title,
                "mime_type": "text/markdown",
                "content": revised,
                "parent_artifact_id": str(base_id),
                "metadata": {
                    "template": "default_project_brief_v1",
                    "sections": list(BRIEF_SECTIONS),
                    "revision_instruction": instruction,
                    "language": language,
                },
            },
        )

    spec = SkillSpec(
        skill_name="internal.brief.revise",
        description=(
            "基于 base brief 与修订指令生成 Brief v2，链路通过 "
            "parent_artifact_id 与 artifact_version 维护。"
        ),
        side_effect="read",
        requires_approval=False,
        supports_dry_run=True,
        timeout_ms=15_000,
    )
    return spec, _revise


def _summarize_for_context(content: Any) -> str:  # noqa: ANN401
    if isinstance(content, str):
        return content[:200]
    if isinstance(content, dict):
        try:
            import json
            return json.dumps(content, ensure_ascii=False)[:200]
        except (TypeError, ValueError):
            return str(content)[:200]
    return str(content)[:200]


# ── 注册入口 ────────────────────────────────────────


def register_internal_document_skills(
    registry: SkillRegistry,
    *,
    brief_generator: BriefGenerator | None = None,
    artifact_reader: ArtifactContentReader | None = None,
    allow_overwrite: bool = False,
) -> BriefGenerator:
    """把 M3 文档技能注册到 ``registry``。

    Returns:
        实际使用的 ``BriefGenerator`` 实例（默认为
        :class:`TemplateBriefGenerator`），方便上层做断言或替换。
    """
    generator = brief_generator or TemplateBriefGenerator()

    spec, fn = _make_context_assemble_skill()
    registry.register(spec, fn, allow_overwrite=allow_overwrite)

    spec, fn = _make_brief_generate_skill(generator)
    registry.register(spec, fn, allow_overwrite=allow_overwrite)

    spec, fn = _make_summary_generate_skill()
    registry.register(spec, fn, allow_overwrite=allow_overwrite)

    if artifact_reader is not None:
        spec, fn = _make_brief_revise_skill(generator, artifact_reader)
        registry.register(spec, fn, allow_overwrite=allow_overwrite)

    return generator


__all__ = [
    "BRIEF_SECTIONS",
    "ArtifactContentReader",
    "BriefGenerator",
    "TemplateBriefGenerator",
    "register_internal_document_skills",
]
