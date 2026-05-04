"""Pilot Ingress：在飞书入站消息和 TaskOrchestrator 之间做意图分流。

设计目标：

- 解决 M4 后所有飞书消息都被强制塞进 "生成 PPT" 任务链路的问题。
- 在不改动 ``AgentPipeline`` 与 ``TaskOrchestrator`` 的前提下，
  根据 *轻量启发式 + 关键词* 区分如下意图：

    * ``IGNORE``         —— 空消息 / 无关闲聊；直接丢弃。
    * ``ORDINARY_QA``    —— 普通问答；走 ``AgentPipeline`` 直接回复。
    * ``PROGRESS_QUERY`` —— 询问任务进度 / 链接；走 ``PilotQueryService``。
    * ``START_TASK``     —— 真正的文档 / PPT / 汇报任务；走 Pilot Orchestrator。

- 所有分类逻辑都是确定性规则，便于测试与解释；后续可在不破坏接口的
  情况下接入 LLM Router 做兜底分类。

入口仅依赖 :class:`FeishuInboundMessage` 这一规范化 DTO，因此长连接
和 HTTP webhook 共用同一套分流路径。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import structlog

from agent_hub.connectors.feishu.models import (
    FeishuChatType,
    FeishuInboundMessage,
    FeishuMessageType,
)
from agent_hub.pilot.services.dto import PlanProfile
from agent_hub.pilot.services.model_gateway import PlanContext, derive_plan_profile

if TYPE_CHECKING:
    from agent_hub.core.pipeline import AgentPipeline
    from agent_hub.pilot.services.queries import PilotQueryService

logger = structlog.get_logger(__name__)


# ── 关键词词典（保持单一职责，避免和 webhook trigger keyword 混淆） ──

# 任务类关键词：命中即认为用户希望 Pilot 真正"做点东西"。
TASK_KEYWORDS: tuple[str, ...] = (
    "ppt",
    "演示稿",
    "汇报",
    "报告",
    "方案",
    "文档",
    "doc",
    "brief",
    "纪要",
    "整理",
    "生成",
    "做一份",
    "做一个",
    "写一份",
    "写一个",
    "起草",
    "编写",
    "梳理",
    "总结一下",
)

# 进度查询关键词：必须同时命中"进度词" + "任务词"或带 ?/？，避免误伤。
PROGRESS_HINT_KEYWORDS: tuple[str, ...] = (
    "进度",
    "做到哪",
    "做完了",
    "做完没",
    "状态",
    "结果呢",
    "链接",
    "产出",
    "搞完",
    "怎么样了",
)

PROGRESS_SUBJECT_KEYWORDS: tuple[str, ...] = (
    "任务",
    "工作",
    "ppt",
    "文档",
    "汇报",
    "报告",
    "方案",
    "brief",
    "刚才",
    "上面",
)


class IngressIntent(StrEnum):
    IGNORE = "ignore"
    ORDINARY_QA = "ordinary_qa"
    PROGRESS_QUERY = "progress_query"
    START_TASK = "start_task"


@dataclass(frozen=True, slots=True)
class IngressDecision:
    """Ingress 分流结果。

    - ``intent`` 决定下游路径；
    - ``reply_text`` 在 QA / 进度查询 / IGNORE 场景中由 ingress 直接产出
      （为 ``None`` 时连接器不发消息，仅记日志）；
    - ``matched_keywords`` 仅用于可观测性。
    """

    intent: IngressIntent
    reply_text: str | None = None
    plan_profile: PlanProfile | None = None
    matched_keywords: tuple[str, ...] = ()
    reason: str | None = None


class PilotIngressService:
    """飞书入站消息的意图分流器。

    依赖均为可选：缺少 ``pipeline`` 时 ``ORDINARY_QA`` 退化为 IGNORE，
    缺少 ``queries`` 时 ``PROGRESS_QUERY`` 退化为提示性文案，主流程都
    不会因此抛错。这样可以在测试和最小部署中渐进开启能力。
    """

    def __init__(
        self,
        *,
        pipeline: AgentPipeline | None = None,
        queries: PilotQueryService | None = None,
        repository_lookup=None,  # noqa: ANN001 — 异步可调用 ``(chat_id) -> Workspace | None``
    ) -> None:
        self._pipeline = pipeline
        self._queries = queries
        self._lookup_workspace = repository_lookup

    # ── 分类 ─────────────────────────────────────────

    def classify(self, message: FeishuInboundMessage) -> IngressDecision:
        """无副作用地把消息映射到 :class:`IngressIntent`。"""
        text = (message.text or "").strip()
        # 非文本消息（图片 / 文件 / 卡片回调等）直接交给 Task 路径，由
        # Pilot 决定是否需要附件 brief；Ingress 不替它做解读。
        if message.message_type is not FeishuMessageType.TEXT:
            return IngressDecision(
                intent=IngressIntent.START_TASK,
                plan_profile=self._derive_profile(message),
                reason="non_text_message",
            )
        if not text:
            return IngressDecision(intent=IngressIntent.IGNORE, reason="empty_text")

        lower = text.lower()

        progress_hits = tuple(k for k in PROGRESS_HINT_KEYWORDS if k in lower)
        if progress_hits:
            subject_hits = tuple(
                k for k in PROGRESS_SUBJECT_KEYWORDS if k in lower
            )
            if subject_hits or text.endswith(("?", "？")):
                return IngressDecision(
                    intent=IngressIntent.PROGRESS_QUERY,
                    matched_keywords=progress_hits + subject_hits,
                )

        task_hits = tuple(k for k in TASK_KEYWORDS if k in lower)
        if task_hits:
            return IngressDecision(
                intent=IngressIntent.START_TASK,
                plan_profile=self._derive_profile(message),
                matched_keywords=task_hits,
            )

        # 群聊里只在被 @ 时才走 QA，避免无端打扰；webhook 已做 mention 过滤，
        # 这里只是兜底保险。
        if (
            message.chat_type is FeishuChatType.GROUP
            and not message.bot_mentioned
        ):
            return IngressDecision(
                intent=IngressIntent.IGNORE,
                reason="group_without_mention",
            )

        return IngressDecision(intent=IngressIntent.ORDINARY_QA)

    # ── 异步 dispatch（可生成 reply_text） ──────────

    async def dispatch(self, message: FeishuInboundMessage) -> IngressDecision:
        """在分类基础上为 QA / 进度查询补齐 ``reply_text``。

        - START_TASK / IGNORE 直接返回 :meth:`classify` 结果。
        - ORDINARY_QA：调用 ``AgentPipeline.run`` 取回回复文本。
        - PROGRESS_QUERY：到 ``PilotQueryService`` 找最近一个任务做摘要。
        """
        decision = self.classify(message)
        intent = decision.intent

        if intent is IngressIntent.ORDINARY_QA:
            reply = await self._answer_qa(message)
            return IngressDecision(
                intent=intent,
                reply_text=reply,
                plan_profile=decision.plan_profile,
                matched_keywords=decision.matched_keywords,
                reason=decision.reason,
            )
        if intent is IngressIntent.PROGRESS_QUERY:
            reply = await self._answer_progress(message)
            return IngressDecision(
                intent=intent,
                reply_text=reply,
                plan_profile=decision.plan_profile,
                matched_keywords=decision.matched_keywords,
                reason=decision.reason,
            )
        return decision

    # ── 内部 ─────────────────────────────────────────

    async def _answer_qa(self, message: FeishuInboundMessage) -> str:
        if self._pipeline is None:
            logger.info(
                "pilot.ingress.qa_unavailable",
                chat_id=message.chat_id,
            )
            return "（普通问答能力暂未启用，请稍后再试。）"
        # 延迟 import 避免与 pilot 模块互相加载。
        from agent_hub.core.enums import UserRole
        from agent_hub.core.models import TaskInput, UserContext

        user_ctx = UserContext(
            user_id=message.sender_id or "feishu-user",
            role=UserRole.USER,
            channel="feishu",
            session_id=message.chat_id,
            group_id=(
                message.chat_id
                if message.chat_type is FeishuChatType.GROUP
                else None
            ),
            is_private=message.chat_type is FeishuChatType.P2P,
        )
        task_input = TaskInput(
            user_context=user_ctx,
            raw_message=message.text or "",
            attachments=[],
            is_at_bot=message.bot_mentioned,
        )
        try:
            output = await self._pipeline.run(task_input)
        except Exception as exc:  # noqa: BLE001 — QA 失败兜底
            logger.warning(
                "pilot.ingress.qa_failed",
                chat_id=message.chat_id,
                error=str(exc),
            )
            return "（抱歉，问答暂时失败，请稍后再试。）"
        return output.response or "（暂无回答内容。）"

    @staticmethod
    def _derive_profile(message: FeishuInboundMessage) -> PlanProfile:
        return derive_plan_profile(
            PlanContext(
                raw_text=message.text or "",
                requester_id=message.sender_id,
                title=(message.text or "")[:32] or None,
                attachments=[a.file_key for a in message.attachments],
                source_channel="feishu",
                source_conversation_id=message.chat_id,
                source_message_id=message.message_id,
                metadata={
                    "feishu_message_type": message.message_type.value,
                    "feishu_chat_type": message.chat_type.value,
                },
            )
        )

    async def _answer_progress(self, message: FeishuInboundMessage) -> str:
        if self._queries is None or self._lookup_workspace is None:
            return "（进度查询能力暂未启用。）"
        try:
            workspace = await self._lookup_workspace(message.chat_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "pilot.ingress.progress_lookup_failed",
                chat_id=message.chat_id,
                error=str(exc),
            )
            return "（暂时查不到该会话的任务记录。）"
        if workspace is None:
            return "本会话还没有创建过任务，告诉我你想做什么我可以开始。"
        try:
            tasks = await self._queries.list_tasks(
                workspace_id=workspace.workspace_id,
                limit=1,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "pilot.ingress.progress_list_failed",
                workspace_id=workspace.workspace_id,
                error=str(exc),
            )
            return "（任务进度查询暂时不可用。）"
        if not tasks:
            return "本会话还没有进行中的任务。"
        latest = tasks[0]
        return (
            f"最新任务「{latest.title}」当前状态 {latest.status.value}，"
            f"待审批 {latest.pending_approvals} 项，产物 {latest.artifact_count} 个。"
        )


__all__ = [
    "IngressDecision",
    "IngressIntent",
    "PilotIngressService",
    "PROGRESS_HINT_KEYWORDS",
    "PROGRESS_SUBJECT_KEYWORDS",
    "TASK_KEYWORDS",
]
