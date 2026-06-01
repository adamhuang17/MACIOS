"""ModelGateway：Plan 生成入口。

M1 提供 :class:`TemplatePlanGateway`（旧名 ``FakeModelGateway``，作为别名保留）
和 :class:`OpenAIModelGateway`。

- TemplatePlanGateway：根据 :class:`PlanProfile` 与 ``skill_mode`` 选择不同的硬模板，
  让 START_TASK 不再被强制走六步；同时支持 ``fake`` / ``real_chain`` 两套技能名。
- OpenAIModelGateway：调用 OpenAI 兼容 API（含智谱 GLM）走 JSON mode，失败降级。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_hub.core.openai_compat import provider_extra_body
from agent_hub.pilot.domain.enums import PlanStepKind, RiskLevel
from agent_hub.pilot.services.dto import (
    PlanBlueprint,
    PlanProfile,
    PlanStepDraft,
)

logger = logging.getLogger(__name__)


SkillMode = Literal["fake", "real_chain"]


class PlanContext(BaseModel):
    """提供给模型生成 plan 时的最小上下文。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_text: str
    requester_id: str
    title: str | None = None
    attachments: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    profile: PlanProfile | None = None
    source_channel: str | None = None
    source_conversation_id: str | None = None
    source_message_id: str | None = None


@runtime_checkable
class ModelGateway(Protocol):
    """ModelGateway 协议；M1 只要求实现 ``complex_plan``。"""

    async def complex_plan(self, ctx: PlanContext) -> PlanBlueprint: ...


# ── Profile 推导（关键词 → PlanProfile） ────────────────

_DECK_KEYWORDS: tuple[str, ...] = ("ppt", "演示", "幻灯", "deck", "slides", "汇报", "报告")
_DOC_KEYWORDS: tuple[str, ...] = (
    "brief", "文档", "方案", "doc", "起草", "撰写", "写一份", "写一个",
)
_SUMMARY_KEYWORDS: tuple[str, ...] = ("总结", "摘要", "纪要", "summary", "回顾")
_REVISION_KEYWORDS: tuple[str, ...] = ("修改", "重写", "修订", "再改", "调整一下")


def derive_plan_profile(ctx: PlanContext) -> PlanProfile:
    """按显式 profile > metadata > raw_text 关键词 > 默认值的优先级派生 profile。"""
    if ctx.profile is not None:
        return ctx.profile
    meta_profile = ctx.metadata.get("plan_profile") if ctx.metadata else None
    if isinstance(meta_profile, PlanProfile):
        return meta_profile
    if isinstance(meta_profile, str):
        try:
            return PlanProfile(meta_profile)
        except ValueError:
            pass
    text = (ctx.raw_text or "").lower()
    has_base_artifact = bool(
        ctx.metadata
        and (
            ctx.metadata.get("base_brief_artifact_id")
            or ctx.metadata.get("base_artifact_id")
        )
    )
    if has_base_artifact and any(k in text for k in _REVISION_KEYWORDS):
        return PlanProfile.REVISION
    if any(k in text for k in _SUMMARY_KEYWORDS) and not any(
        k in text for k in _DECK_KEYWORDS
    ):
        return PlanProfile.SUMMARY_ONLY
    if any(k in text for k in _DECK_KEYWORDS):
        return PlanProfile.DECK_FULL
    if any(k in text for k in _DOC_KEYWORDS):
        return PlanProfile.DOC_ONLY
    return PlanProfile.DECK_FULL


# ── Skill 名称矩阵 ─────────────────────────────────────

_SKILL_NAMES: dict[SkillMode, dict[str, str]] = {
    "fake": {
        "context": "fake_im_fetch",
        "brief": "fake_brief_generate",
        "brief_revise": "fake_brief_generate",
        "spec": "fake_slidespec_generate",
        "pptx": "fake_pptx_render",
        "docx": "",
        "upload": "fake_drive_upload",
        "summary": "fake_im_send_summary",
    },
    "real_chain": {
        "context": "internal.context.assemble",
        "brief": "internal.brief.generate",
        "brief_revise": "internal.brief.revise",
        "spec": "real.slide.generate_spec",
        "pptx": "real.slide.render_pptx",
        "docx": "real.doc.render_docx",
        "upload": "real.drive.upload_share",
        "summary": "feishu.im.send_message",
    },
}


class TemplatePlanGateway:
    """确定性、profile 化的 plan 生成器。

    根据 :class:`PlanProfile` 选择不同的 step 模板：

    - ``DECK_FULL``    → 6 步（READ_CONTEXT/GENERATE_DOC/GENERATE_SLIDE_SPEC/
      RENDER_SLIDES/UPLOAD/SUMMARIZE），与历史 FakeModelGateway 对齐；
    - ``DOC_ONLY``     → 2 步（READ_CONTEXT → GENERATE_DOC）；
    - ``SUMMARY_ONLY`` → 2 步（READ_CONTEXT → SUMMARIZE）；
    - ``REVISION``     → 2 步（READ_CONTEXT → GENERATE_DOC[brief.revise]）。

    ``skill_mode`` 决定使用 ``fake_*`` 还是 ``internal.* / real.* / feishu.*`` 真实技能名，
    便于产线开关 ``pilot_use_real_chain`` 时无需改动 PlanningService 校验。
    """

    def __init__(self, *, skill_mode: SkillMode = "fake") -> None:
        if skill_mode not in _SKILL_NAMES:
            raise ValueError(f"unsupported skill_mode: {skill_mode}")
        self._skill_mode: SkillMode = skill_mode

    @property
    def skill_mode(self) -> SkillMode:
        return self._skill_mode

    async def complex_plan(self, ctx: PlanContext) -> PlanBlueprint:
        profile = derive_plan_profile(ctx)
        skills = _SKILL_NAMES[self._skill_mode]
        title = (ctx.title or ctx.raw_text[:32] or "Untitled Task").strip()
        chat_id = ctx.source_conversation_id or ""

        if profile is PlanProfile.DECK_FULL:
            steps = self._build_deck_full_steps(ctx, title, skills, chat_id)
            goal = f"为「{title}」生成一份可分享的演示稿"
            assumptions = ["输入文本完整", "Drive 写权限可用"]
        elif profile is PlanProfile.DOC_ONLY:
            steps = self._build_doc_only_steps(ctx, title, skills, chat_id)
            goal = f"为「{title}」生成 Brief 文档"
            assumptions = ["输入文本完整"]
        elif profile is PlanProfile.SUMMARY_ONLY:
            steps = self._build_summary_only_steps(ctx, title, skills, chat_id)
            goal = f"为「{title}」生成会话摘要"
            assumptions = ["上下文中已包含可摘要的消息"]
        elif profile is PlanProfile.REVISION:
            steps = self._build_revision_steps(ctx, title, skills)
            goal = f"按指令修订「{title}」的 Brief"
            base_id = (ctx.metadata or {}).get("base_brief_artifact_id") or (
                ctx.metadata or {}
            ).get("base_artifact_id", "")
            assumptions = [f"base_brief_artifact_id={base_id}"]
        else:  # pragma: no cover - PlanProfile 已枚举完
            raise ValueError(f"unknown profile: {profile}")

        return PlanBlueprint(
            goal=goal,
            assumptions=assumptions,
            steps=steps,
            profile=profile,
            metadata={
                "gateway": "template",
                "skill_mode": self._skill_mode,
            },
        )

    # ── Profile-specific builders ─────────────────────

    @staticmethod
    def _build_deck_full_steps(
        ctx: PlanContext,
        title: str,
        skills: dict[str, str],
        chat_id: str,
    ) -> list[PlanStepDraft]:
        context_skill = _context_skill_name(ctx, skills)
        steps = [
            PlanStepDraft(
                ref="ctx",
                title="收集上下文",
                kind=PlanStepKind.READ_CONTEXT,
                skill_name=context_skill,
                input_params=_context_input_params(ctx, chat_id),
                output_artifact_ref="ctx",
                risk_level=RiskLevel.READ,
            ),
            PlanStepDraft(
                ref="brief",
                title="生成 Brief",
                kind=PlanStepKind.GENERATE_DOC,
                skill_name=skills["brief"],
                input_params={"title": f"{title} Brief"},
                inputs_from=["ctx"],
                output_artifact_ref="brief",
                depends_on=["ctx"],
                risk_level=RiskLevel.READ,
            ),
            PlanStepDraft(
                ref="spec",
                title="生成 SlideSpec",
                kind=PlanStepKind.GENERATE_SLIDE_SPEC,
                skill_name=skills["spec"],
                input_params={"title": title},
                inputs_from=["brief"],
                output_artifact_ref="spec",
                depends_on=["brief"],
                risk_level=RiskLevel.READ,
            ),
            PlanStepDraft(
                ref="pptx",
                title="渲染 PPTX",
                kind=PlanStepKind.RENDER_SLIDES,
                skill_name=skills["pptx"],
                input_params={"title": title},
                inputs_from=["spec"],
                output_artifact_ref="pptx",
                depends_on=["spec"],
                risk_level=RiskLevel.WRITE,
                requires_approval=True,
            ),
            PlanStepDraft(
                ref="share",
                title="上传 Drive 并生成分享",
                kind=PlanStepKind.UPLOAD,
                skill_name=skills["upload"],
                input_params={
                    "file_name": f"{_safe_file_stem(title)}.pptx",
                    **_upload_input_params(ctx),
                },
                inputs_from=["pptx"],
                output_artifact_ref="share",
                depends_on=["pptx"],
                risk_level=RiskLevel.SHARE,
                requires_approval=True,
            ),
            PlanStepDraft(
                ref="summary",
                title="私发生成结果",
                kind=PlanStepKind.SUMMARIZE,
                skill_name=skills["summary"],
                input_params=_summary_delivery_params(ctx, chat_id, title),
                inputs_from=["share"],
                output_artifact_ref="summary",
                depends_on=["share"],
                risk_level=RiskLevel.SHARE,
                requires_approval=True,
            ),
        ]
        return steps

    @staticmethod
    def _build_doc_only_steps(
        ctx: PlanContext,
        title: str,
        skills: dict[str, str],
        chat_id: str = "",
    ) -> list[PlanStepDraft]:
        context_skill = _context_skill_name(ctx, skills)
        steps = [
            PlanStepDraft(
                ref="ctx",
                title="收集上下文",
                kind=PlanStepKind.READ_CONTEXT,
                skill_name=context_skill,
                input_params=_context_input_params(ctx, chat_id),
                output_artifact_ref="ctx",
                risk_level=RiskLevel.READ,
            ),
            PlanStepDraft(
                ref="brief",
                title="生成 Brief",
                kind=PlanStepKind.GENERATE_DOC,
                skill_name=skills["brief"],
                input_params={"title": f"{title} Brief"},
                inputs_from=["ctx"],
                output_artifact_ref="brief",
                depends_on=["ctx"],
                risk_level=RiskLevel.READ,
            ),
        ]

        docx_skill = skills.get("docx") or ""
        if docx_skill:
            steps.extend([
                PlanStepDraft(
                    ref="doc",
                    title="渲染 Word",
                    kind=PlanStepKind.GENERATE_DOC,
                    skill_name=docx_skill,
                    input_params={"title": title},
                    inputs_from=["brief"],
                    output_artifact_ref="doc",
                    depends_on=["brief"],
                    risk_level=RiskLevel.WRITE,
                    requires_approval=True,
                ),
                PlanStepDraft(
                    ref="share",
                    title="上传 Drive 并生成分享",
                    kind=PlanStepKind.UPLOAD,
                    skill_name=skills["upload"],
                    input_params={
                        "file_name": f"{_safe_file_stem(title)}.docx",
                        **_upload_input_params(ctx),
                    },
                    inputs_from=["doc"],
                    output_artifact_ref="share",
                    depends_on=["doc"],
                    risk_level=RiskLevel.SHARE,
                    requires_approval=True,
                ),
            ])
        else:
            steps.append(
                PlanStepDraft(
                    ref="share",
                    title="上传 Drive 并生成分享",
                    kind=PlanStepKind.UPLOAD,
                    skill_name=skills["upload"],
                    input_params={
                        "file_name": f"{_safe_file_stem(title)}.md",
                        **_upload_input_params(ctx),
                    },
                    inputs_from=["brief"],
                    output_artifact_ref="share",
                    depends_on=["brief"],
                    risk_level=RiskLevel.SHARE,
                    requires_approval=True,
                ),
            )

        steps.append(
            PlanStepDraft(
                ref="summary",
                title="私发生成结果",
                kind=PlanStepKind.SUMMARIZE,
                skill_name=skills["summary"],
                input_params=_summary_delivery_params(ctx, chat_id, title),
                inputs_from=["share"],
                output_artifact_ref="summary",
                depends_on=["share"],
                risk_level=RiskLevel.SHARE,
                requires_approval=True,
            ),
        )
        return steps

    @staticmethod
    def _build_summary_only_steps(
        ctx: PlanContext,
        title: str,
        skills: dict[str, str],
        chat_id: str,
    ) -> list[PlanStepDraft]:
        context_skill = _context_skill_name(ctx, skills)
        return [
            PlanStepDraft(
                ref="ctx",
                title="收集上下文",
                kind=PlanStepKind.READ_CONTEXT,
                skill_name=context_skill,
                input_params=_context_input_params(ctx, chat_id),
                output_artifact_ref="ctx",
                risk_level=RiskLevel.READ,
            ),
            PlanStepDraft(
                ref="summary",
                title="生成摘要",
                kind=PlanStepKind.SUMMARIZE,
                skill_name="internal.summary.generate"
                if skills["context"].startswith("internal.")
                else skills["summary"],
                input_params={"title": f"{title} 摘要", "chat_id": chat_id},
                inputs_from=["ctx"],
                output_artifact_ref="summary",
                depends_on=["ctx"],
                risk_level=RiskLevel.READ,
            ),
        ]

    @staticmethod
    def _build_revision_steps(
        ctx: PlanContext,
        title: str,
        skills: dict[str, str],
    ) -> list[PlanStepDraft]:
        meta = ctx.metadata or {}
        base_id = str(
            meta.get("base_brief_artifact_id")
            or meta.get("base_artifact_id")
            or "",
        )
        instruction = str(meta.get("revision_instruction") or ctx.raw_text or "")
        chat_id = ctx.source_conversation_id or ""
        context_skill = _context_skill_name(ctx, skills)
        return [
            PlanStepDraft(
                ref="ctx",
                title="重新收集上下文",
                kind=PlanStepKind.READ_CONTEXT,
                skill_name=context_skill,
                input_params=_context_input_params(ctx, chat_id),
                output_artifact_ref="ctx",
                risk_level=RiskLevel.READ,
            ),
            PlanStepDraft(
                ref="brief",
                title="重写 Brief",
                kind=PlanStepKind.GENERATE_DOC,
                skill_name=skills["brief_revise"],
                input_params={
                    "title": f"{title} (revised)",
                    "base_brief_artifact_id": base_id,
                    "revision_instruction": instruction,
                },
                inputs_from=["ctx"],
                output_artifact_ref="brief",
                depends_on=["ctx"],
                risk_level=RiskLevel.READ,
            ),
        ]


def _context_skill_name(ctx: PlanContext, skills: dict[str, str]) -> str:
    if (
        ctx.source_channel == "feishu"
        and ctx.source_conversation_id
        and skills["context"] == "internal.context.assemble"
    ):
        return "feishu.im.fetch_recent"
    return skills["context"]


def _context_input_params(ctx: PlanContext, chat_id: str) -> dict[str, Any]:
    metadata = dict(ctx.metadata or {})
    params: dict[str, Any] = {
        "origin_text": ctx.raw_text,
        "attachments": list(ctx.attachments),
        "source_message_id": ctx.source_message_id,
        "context_scope": {
            "channel": ctx.source_channel or "",
            "chat_id": chat_id,
            "message_id": ctx.source_message_id or "",
        },
    }
    if ctx.source_channel == "feishu" and chat_id:
        params["chat_id"] = chat_id
        params["page_size"] = int(metadata.get("feishu_context_page_size") or 20)
    return params


def _safe_file_stem(title: str) -> str:
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", (title or "").strip())
    stem = re.sub(r"\s+", " ", stem).strip(" .")
    return stem[:80] or "pilot-output"


def _default_upload_file_name(title: str, inputs_from: list[str]) -> str:
    refs = set(inputs_from or [])
    if {"doc", "docx"} & refs:
        suffix = ".docx"
    elif "brief" in refs:
        suffix = ".md"
    else:
        suffix = ".pptx"
    return f"{_safe_file_stem(title)}{suffix}"


def _upload_input_params(ctx: PlanContext) -> dict[str, Any]:
    open_id = _feishu_requester_open_id(ctx)
    metadata = dict(ctx.metadata or {})
    params: dict[str, Any] = {}
    if open_id:
        params["member_open_id"] = open_id
        params["share_recipient_open_id"] = open_id

    folder_token = _feishu_target_folder_token(ctx)
    if folder_token:
        params["folder_token"] = folder_token

    shared_open_ids = _collect_open_ids(
        metadata.get("feishu_share_recipient_open_ids"),
        metadata.get("feishu_shared_open_ids"),
        metadata.get("share_recipient_open_ids"),
        metadata.get("shared_open_ids"),
    )
    if _feishu_file_visibility(ctx) == "shared" and shared_open_ids:
        params["share_recipient_open_ids"] = shared_open_ids
    return params


def _summary_delivery_params(
    ctx: PlanContext,
    chat_id: str,
    title: str,
) -> dict[str, Any]:
    receive_id, receive_id_type = _feishu_private_receive(ctx, chat_id)
    params: dict[str, Any] = {
        "chat_id": receive_id,
        "receive_id": receive_id,
        "receive_id_type": receive_id_type,
        "title": f"{title} 已完成",
    }
    if ctx.source_channel == "feishu":
        params["delivery_mode"] = "private"
        params["source_chat_id"] = chat_id
    return params


def _feishu_private_receive(ctx: PlanContext, fallback_chat_id: str) -> tuple[str, str]:
    metadata = dict(ctx.metadata or {})
    if ctx.source_channel == "feishu":
        receive_id = str(
            metadata.get("feishu_private_receive_id")
            or metadata.get("feishu_requester_open_id")
            or ctx.requester_id
            or ""
        ).strip()
        receive_id_type = str(
            metadata.get("feishu_private_receive_id_type")
            or metadata.get("feishu_sender_id_type")
            or "open_id"
        ).strip() or "open_id"
        if receive_id:
            return receive_id, receive_id_type
    return fallback_chat_id, "chat_id"


def _feishu_requester_open_id(ctx: PlanContext) -> str:
    metadata = dict(ctx.metadata or {})
    explicit = str(metadata.get("feishu_requester_open_id") or "").strip()
    if explicit:
        return explicit
    sender_type = str(
        metadata.get("feishu_private_receive_id_type")
        or metadata.get("feishu_sender_id_type")
        or ""
    )
    if ctx.source_channel == "feishu" and sender_type in ("", "open_id"):
        return str(
            metadata.get("feishu_private_receive_id")
            or metadata.get("feishu_sender_id")
            or ctx.requester_id
            or ""
        ).strip()
    return ""


def _feishu_file_visibility(ctx: PlanContext) -> str:
    metadata = dict(ctx.metadata or {})
    explicit = str(
        metadata.get("feishu_file_visibility")
        or metadata.get("file_visibility")
        or metadata.get("drive_visibility")
        or ""
    ).strip().lower()
    if explicit in {"shared", "share", "public", "team"}:
        return "shared"
    if explicit in {"private", "personal"}:
        return "private"
    text = (ctx.raw_text or "").lower()
    if any(k in text for k in ("共享文件", "共享文档", "共享给", "分享给")):
        return "shared"
    return "private"


def _feishu_target_folder_token(ctx: PlanContext) -> str:
    metadata = dict(ctx.metadata or {})
    explicit = str(
        metadata.get("feishu_folder_token") or metadata.get("folder_token") or ""
    ).strip()
    if explicit:
        return explicit
    visibility = _feishu_file_visibility(ctx)
    if visibility == "shared":
        return str(
            metadata.get("feishu_shared_folder_token")
            or metadata.get("shared_folder_token")
            or ""
        ).strip()
    return str(
        metadata.get("feishu_private_folder_token")
        or metadata.get("private_folder_token")
        or ""
    ).strip()


def _collect_open_ids(*values: object) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in _iter_open_ids(value):
            if item and item not in seen:
                seen.add(item)
                result.append(item)
    return result


def _iter_open_ids(value: object):  # noqa: ANN202
    if value is None:
        return
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            if parsed is not None:
                yield from _iter_open_ids(parsed)
                return
        for part in raw.replace("；", ",").replace(";", ",").split(","):
            item = part.strip()
            if item:
                yield item
        return
    if isinstance(value, dict):
        for key in ("open_id", "member_open_id", "user_id"):
            if key in value:
                yield from _iter_open_ids(value[key])
        return
    try:
        iterator = iter(value)
    except TypeError:
        return
    for item in iterator:
        yield from _iter_open_ids(item)


# 兼容别名：M1~M4 期间所有调用方使用的名字。
FakeModelGateway = TemplatePlanGateway


# ── 真实 LLM Gateway ─────────────────────────────────


_OPENAI_SYSTEM_PROMPT_TEMPLATE = """\
你是 Pilot 的规划器。请根据用户输入和指定的 plan profile，生成一份 PlanBlueprint JSON。

# Plan profile 与必含的 PlanStepKind
- deck_full:    READ_CONTEXT, GENERATE_DOC, GENERATE_SLIDE_SPEC, RENDER_SLIDES, UPLOAD, SUMMARIZE
- doc_only:     READ_CONTEXT, GENERATE_DOC, UPLOAD, SUMMARIZE（real_chain 时必须包含 real.doc.render_docx 渲染 Word）
- summary_only: READ_CONTEXT, SUMMARIZE
- revision:     READ_CONTEXT, GENERATE_DOC

当前 profile: {profile}

# 可用 skill 白名单
{skill_allowlist}

# 输出 JSON Schema（必须严格符合）
{{
  "goal": "<一句话目标>",
  "assumptions": ["<假设1>", "..."],
  "profile": "{profile}",
  "metadata": {{"gateway": "openai", "skill_mode": "{skill_mode}"}},
  "steps": [
    {{
      "ref": "<draft 内部 id，如 ctx/brief/...>",
      "title": "<step 名>",
      "kind": "<PlanStepKind 字符串>",
      "skill_name": "<上面白名单内的 skill>",
      "input_params": {{}},
      "inputs_from": ["<上一 step 的 output_artifact_ref>"],
      "output_artifact_ref": "<本 step 产出 ref>",
      "depends_on": ["<前置 step.ref>"],
      "risk_level": "read|write|share|admin",
      "requires_approval": true,
      "timeout_ms": 30000,
      "max_retries": 0
    }}
  ]
}}

# 硬约束
1. 必须只输出 JSON，不要 markdown 代码块、不要解释文字。
2. step.ref 必须唯一，depends_on / inputs_from 必须只引用已出现的 ref。
3. 写/分享类操作（write/share/admin）必须 requires_approval=true。
4. 不要使用未在白名单中的 skill_name；也不要凭空增减 PlanStepKind。
"""


def _format_skill_allowlist(skill_mode: SkillMode) -> str:
    skills = _SKILL_NAMES[skill_mode]
    return "\n".join(f"- {role}: {name}" for role, name in skills.items())


class OpenAIModelGateway:
    """真实 ModelGateway：用 OpenAI 兼容接口（含智谱 GLM）生成 PlanBlueprint。

    设计：

    - 通过 ``response_format={"type": "json_object"}`` 让模型直接返回 JSON；
    - 解析失败 / 网络失败 / pydantic 校验失败时降级到 ``fallback`` Gateway；
    - 仅在 ``Settings.pilot_use_real_gateway`` 为 True 时由 ``build_pilot_runtime`` 装配。
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str = "",
        model: str = "glm-4-flash",
        skill_mode: SkillMode = "fake",
        timeout: float = 30.0,
        fallback: ModelGateway | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._skill_mode: SkillMode = skill_mode
        self._timeout = timeout
        self._fallback: ModelGateway = fallback or TemplatePlanGateway(
            skill_mode=skill_mode,
        )

    async def complex_plan(self, ctx: PlanContext) -> PlanBlueprint:
        if not self._api_key:
            return await self._fallback.complex_plan(ctx)
        try:
            return await self._call_real(ctx)
        except Exception as exc:  # noqa: BLE001 - 真实链路失败必须降级
            logger.warning("openai_gateway.fallback", extra={"error": str(exc)})
            return await self._fallback.complex_plan(ctx)

    async def _call_real(self, ctx: PlanContext) -> PlanBlueprint:
        """调用 OpenAI 兼容 chat.completions，强 JSON mode 输出 PlanBlueprint。

        Raises:
            Exception: 网络错误、JSON 解析错误或 PlanBlueprint 校验错误；由
                :meth:`complex_plan` 捕获后降级到 ``fallback``。
        """
        # 延迟 import，避免在没启用真实 gateway 的部署里强依赖 openai SDK。
        from openai import AsyncOpenAI

        profile = derive_plan_profile(ctx)
        system_prompt = _OPENAI_SYSTEM_PROMPT_TEMPLATE.format(
            profile=profile.value,
            skill_mode=self._skill_mode,
            skill_allowlist=_format_skill_allowlist(self._skill_mode),
        )
        user_payload = {
            "title": ctx.title,
            "raw_text": ctx.raw_text,
            "requester_id": ctx.requester_id,
            "attachments": list(ctx.attachments),
            "source_channel": ctx.source_channel,
            "source_conversation_id": ctx.source_conversation_id,
            "metadata": dict(ctx.metadata or {}),
        }
        client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url or None,
        )
        response = await client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            response_format={"type": "json_object"},
            extra_body=provider_extra_body(self._base_url, self._model) or None,
            timeout=self._timeout,
        )
        content = response.choices[0].message.content or ""
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"openai gateway returned non-JSON: {exc}") from exc
        try:
            blueprint = PlanBlueprint.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"openai gateway blueprint invalid: {exc}") from exc
        if self._skill_mode == "real_chain":
            issue = _real_chain_blueprint_issue(blueprint)
            if issue:
                raise ValueError(f"openai gateway blueprint incomplete: {issue}")
        # 强制把真实 metadata 写入，避免模型省略。
        merged_meta = {
            **dict(blueprint.metadata),
            "gateway": "openai",
            "skill_mode": self._skill_mode,
            "model": self._model,
        }
        if self._skill_mode == "real_chain":
            blueprint = _apply_real_chain_defaults(blueprint, ctx)
        blueprint = _apply_feishu_private_delivery(blueprint, ctx)
        return blueprint.model_copy(update={"metadata": merged_meta})


def _real_chain_blueprint_issue(blueprint: PlanBlueprint) -> str:
    skill_names = {step.skill_name for step in blueprint.steps}
    if blueprint.profile is PlanProfile.DECK_FULL:
        required = {
            "real.slide.generate_spec",
            "real.slide.render_pptx",
            "real.drive.upload_share",
        }
        missing = sorted(required - skill_names)
        if missing:
            return f"missing deck skills: {missing}"
    if blueprint.profile is PlanProfile.DOC_ONLY:
        required = {"internal.brief.generate", "real.doc.render_docx", "real.drive.upload_share"}
        missing = sorted(required - skill_names)
        if missing:
            return f"missing doc skills: {missing}"
    return ""


def _apply_real_chain_defaults(
    blueprint: PlanBlueprint,
    ctx: PlanContext,
) -> PlanBlueprint:
    title = (ctx.title or ctx.raw_text[:32] or "Untitled Task").strip()
    new_steps: list[PlanStepDraft] = []
    for step in blueprint.steps:
        input_params = dict(step.input_params)
        if step.skill_name in {
            "real.slide.generate_spec",
            "real.slide.render_pptx",
            "real.doc.render_docx",
        }:
            input_params.setdefault("title", title)
        elif step.skill_name == "real.drive.upload_share":
            input_params.setdefault(
                "file_name",
                _default_upload_file_name(title, step.inputs_from),
            )
        new_steps.append(step.model_copy(update={"input_params": input_params}))
    return blueprint.model_copy(update={"steps": new_steps})


def _apply_feishu_private_delivery(
    blueprint: PlanBlueprint,
    ctx: PlanContext,
) -> PlanBlueprint:
    if ctx.source_channel != "feishu":
        return blueprint

    chat_id = ctx.source_conversation_id or ""
    title = (ctx.title or ctx.raw_text[:32] or "Untitled Task").strip()
    new_steps: list[PlanStepDraft] = []
    for step in blueprint.steps:
        input_params = dict(step.input_params)
        skill_name = step.skill_name
        if skill_name == "internal.context.assemble" and chat_id:
            skill_name = "feishu.im.fetch_recent"
            input_params = {**_context_input_params(ctx, chat_id), **input_params}
        elif skill_name == "feishu.im.fetch_recent":
            input_params = {**_context_input_params(ctx, chat_id), **input_params}
        elif skill_name == "real.drive.upload_share":
            input_params = {**input_params, **_upload_input_params(ctx)}
            input_params.setdefault(
                "file_name",
                _default_upload_file_name(title, step.inputs_from),
            )
        elif skill_name == "feishu.im.send_message":
            input_params = {**input_params, **_summary_delivery_params(ctx, chat_id, title)}
        new_steps.append(step.model_copy(update={
            "skill_name": skill_name,
            "input_params": input_params,
        }))
    return blueprint.model_copy(update={"steps": new_steps})


__all__ = [
    "FakeModelGateway",
    "ModelGateway",
    "OpenAIModelGateway",
    "PlanContext",
    "SkillMode",
    "TemplatePlanGateway",
    "derive_plan_profile",
]
