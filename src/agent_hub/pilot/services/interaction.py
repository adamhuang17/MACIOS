"""InteractionService：通道无关的统一入口编排器。

把「用户说了什么」映射到「应该走哪条能力」，并调用对应下游服务：

- 普通问答     → ``AgentPipeline.run()``（保留 ReAct + 记忆）
- 进度查询     → ``PilotQueryService.list_tasks()``
- 复杂任务     → ``TaskOrchestrator.submit()``
- 忽略 / 澄清  → 轻量回复或静默

InteractionService **不**修改 AgentPipeline、TaskOrchestrator 的内部
执行逻辑；它只做「用户说了什么 → 应该走哪条能力」这一层门面决策。

飞书长连接（FeishuWebhookService）、HTTP 调试入口（/api/interactions）
以及未来的评测模拟器都调用同一个 InteractionService 实例，保证产品主路径
与评测路径一致。

关键词词典从 :mod:`agent_hub.pilot.services.ingress` 复用，避免双重维护。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from agent_hub.contracts.identity import SourceChatType
from agent_hub.contracts.interaction import InboundRequest, InteractionIntent
from agent_hub.pilot.domain.enums import TaskStatus
from agent_hub.pilot.services.dto import PlanProfile, TaskRequest
from agent_hub.pilot.services.ingress import (
    PROGRESS_HINT_KEYWORDS,
    PROGRESS_SUBJECT_KEYWORDS,
    TASK_KEYWORDS,
)
from agent_hub.pilot.services.model_gateway import PlanContext, derive_plan_profile

if TYPE_CHECKING:
    from agent_hub.core.pipeline import AgentPipeline
    from agent_hub.pilot.services.orchestrator import TaskOrchestrator
    from agent_hub.pilot.services.queries import PilotQueryService
    from agent_hub.pilot.services.repository import PilotRepository

logger = structlog.get_logger(__name__)

_SOURCE_FEISHU = "feishu"


# ── 结果 ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class InteractionResult:
    """InteractionService.handle() 的统一返回值。

    Attributes:
        intent:           分类结果。
        trace_id:         本次交互的追踪 ID，IGNORE / 轻量回复时也保证非空。
        reply_text:       ORDINARY_QA / PROGRESS_QUERY / 降级时的回复文案。
        task_id:          START_TASK 成功时创建（或幂等命中）的任务 ID。
        workspace_id:     START_TASK 时所属工作区 ID。
        plan_id:          START_TASK 时关联的 plan ID（auto_approve=True 时非空）。
        task_status:      START_TASK 时任务的即时状态。
        events_url:       SSE 事件流 URL（由外层路由填充，此处留空）。
        decision_reason:  分类 / 降级原因，供日志和评测可观测性使用。
        matched_keywords: 触发分类的关键词。
        deduplicated:     START_TASK 时是否幂等命中已有任务。
    """

    intent: InteractionIntent
    trace_id: str
    reply_text: str | None = None
    task_id: str | None = None
    workspace_id: str | None = None
    plan_id: str | None = None
    task_status: TaskStatus | None = None
    events_url: str | None = None
    decision_reason: str | None = None
    matched_keywords: tuple[str, ...] = field(default_factory=tuple)
    deduplicated: bool = False


# ── 服务 ──────────────────────────────────────────────────────────────────────


class InteractionService:
    """通道无关的统一入口编排器。

    所有依赖均为可选：缺少对应服务时自动降级并记录日志，不抛错。
    这样可以在测试和最小部署中渐进开启能力。

    Args:
        pipeline:     AgentPipeline 实例，供普通问答使用。
        queries:      PilotQueryService，供进度查询使用。
        orchestrator: TaskOrchestrator，供启动复杂任务使用。
        repository:   PilotRepository，供 workspace 查找 / 创建使用。
    """

    def __init__(
        self,
        *,
        pipeline: AgentPipeline | None = None,
        queries: PilotQueryService | None = None,
        orchestrator: TaskOrchestrator | None = None,
        repository: PilotRepository | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._queries = queries
        self._orchestrator = orchestrator
        self._repo = repository

    # ── 主入口 ────────────────────────────────────────────────────────────────

    async def handle(self, request: InboundRequest) -> InteractionResult:
        """分类入站请求并调用对应下游能力，返回统一结果。"""
        intent, keywords, reason, plan_profile = self.classify(request)

        logger.info(
            "pilot.interaction.classified",
            channel=request.source_context.channel,
            chat_id=request.source_context.chat_id,
            intent=intent.value,
            matched=list(keywords),
            reason=reason,
        )

        if intent is InteractionIntent.IGNORE:
            return InteractionResult(
                intent=intent,
                trace_id=request.trace_id,
                decision_reason=reason,
                matched_keywords=keywords,
            )

        if intent is InteractionIntent.ORDINARY_QA:
            reply = await self._answer_qa(request)
            return InteractionResult(
                intent=intent,
                trace_id=request.trace_id,
                reply_text=reply,
                matched_keywords=keywords,
                decision_reason=reason,
            )

        if intent is InteractionIntent.PROGRESS_QUERY:
            reply = await self._answer_progress(request)
            return InteractionResult(
                intent=intent,
                trace_id=request.trace_id,
                reply_text=reply,
                matched_keywords=keywords,
                decision_reason=reason,
            )

        if intent is InteractionIntent.START_TASK:
            return await self._start_task(request, plan_profile, keywords, reason)

        # CLARIFY 或未来意图：轻量回复（阶段 1 不分派）
        return InteractionResult(
            intent=intent,
            trace_id=request.trace_id,
            reply_text="（请补充更多信息后再试。）",
            decision_reason=reason,
        )

    def classify(
        self, request: InboundRequest
    ) -> tuple[InteractionIntent, tuple[str, ...], str | None, PlanProfile | None]:
        """Expose the side-effect-free intent classifier for connectors."""
        return self._classify(request)

    # ── 确定性分类 ────────────────────────────────────────────────────────────

    def _classify(
        self, request: InboundRequest
    ) -> tuple[InteractionIntent, tuple[str, ...], str | None, PlanProfile | None]:
        """无副作用地把入站请求映射到 :class:`InteractionIntent`。

        Returns:
            (intent, matched_keywords, reason, plan_profile)
        """
        text = request.text.strip()

        # 非文本消息（图片 / 文件等）直接交给 Task 路径
        if request.message_type != "text":
            return (
                InteractionIntent.START_TASK,
                (),
                "non_text_message",
                self._derive_profile(request),
            )

        if not text:
            return InteractionIntent.IGNORE, (), "empty_text", None

        lower = text.lower()

        progress_hits = tuple(k for k in PROGRESS_HINT_KEYWORDS if k in lower)
        if progress_hits:
            subject_hits = tuple(k for k in PROGRESS_SUBJECT_KEYWORDS if k in lower)
            if subject_hits or text.endswith(("?", "？")):
                return (
                    InteractionIntent.PROGRESS_QUERY,
                    progress_hits + subject_hits,
                    None,
                    None,
                )

        task_hits = tuple(k for k in TASK_KEYWORDS if k in lower)
        if task_hits:
            return (
                InteractionIntent.START_TASK,
                task_hits,
                None,
                self._derive_profile(request),
            )

        # 群聊未 @bot：忽略
        if (
            request.source_context.chat_type is SourceChatType.GROUP
            and not request.source_context.is_at_bot
        ):
            return InteractionIntent.IGNORE, (), "group_without_mention", None

        return InteractionIntent.ORDINARY_QA, (), None, None

    # ── 普通问答 ──────────────────────────────────────────────────────────────

    async def _answer_qa(self, request: InboundRequest) -> str:
        if self._pipeline is None:
            logger.info(
                "pilot.interaction.qa_unavailable",
                channel=request.source_context.channel,
            )
            return "（普通问答能力暂未启用，请稍后再试。）"

        from agent_hub.contracts.execution import TaskInput

        task_input = TaskInput(
            trace_id=request.trace_id,
            user_context=request.user_context,
            source_context=request.source_context,
            raw_message=request.text or "",
            attachments=list(request.attachments),
        )
        try:
            output = await self._pipeline.run(task_input)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "pilot.interaction.qa_failed",
                channel=request.source_context.channel,
                error=str(exc),
            )
            return "（抱歉，问答暂时失败，请稍后再试。）"
        return output.response or "（暂无回答内容。）"

    # ── 进度查询 ──────────────────────────────────────────────────────────────

    async def _answer_progress(self, request: InboundRequest) -> str:
        if self._queries is None or self._repo is None:
            return "（进度查询能力暂未启用。）"

        channel = request.source_context.channel
        chat_id = request.source_context.chat_id
        if not chat_id:
            return "（无法确定当前会话，进度查询不可用。）"

        try:
            workspace = await self._repo.find_workspace_by_conversation(
                channel, chat_id
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "pilot.interaction.progress_lookup_failed",
                channel=channel,
                chat_id=chat_id,
                error=str(exc),
            )
            return "（暂时查不到该会话的任务记录。）"

        if workspace is None:
            return "本会话还没有创建过任务，告诉我你想做什么我可以开始。"

        try:
            tasks = await self._queries.list_tasks(
                workspace_id=workspace.workspace_id, limit=1
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "pilot.interaction.progress_list_failed",
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

    # ── 启动任务 ──────────────────────────────────────────────────────────────

    async def _start_task(
        self,
        request: InboundRequest,
        plan_profile: PlanProfile | None,
        keywords: tuple[str, ...],
        reason: str | None,
    ) -> InteractionResult:
        if self._orchestrator is None or self._repo is None:
            logger.info(
                "pilot.interaction.start_task_unavailable",
                channel=request.source_context.channel,
            )
            return InteractionResult(
                intent=InteractionIntent.START_TASK,
                trace_id=request.trace_id,
                reply_text="（任务能力暂未启用，请稍后再试。）",
                matched_keywords=keywords,
                decision_reason="orchestrator_unavailable",
            )

        workspace = await self._find_or_create_workspace(request)

        idempotency_key: str | None = None
        if request.source_context.message_id:
            idempotency_key = (
                f"msg:{request.source_context.channel}"
                f":{request.source_context.message_id}"
            )

        task_request = TaskRequest(
            source_channel=request.source_context.channel,
            raw_text=request.text or "",
            requester_id=request.user_context.user_id,
            title=(request.text or "")[:32] or None,
            source_conversation_id=request.source_context.chat_id,
            source_message_id=request.source_context.message_id,
            workspace_id=workspace.workspace_id,
            attachments=list(request.attachments),
            idempotency_key=idempotency_key,
            auto_approve=True,
            plan_profile=plan_profile,
            metadata=_task_metadata_from_request(request),
        )

        try:
            handle = await self._orchestrator.submit(task_request)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "pilot.interaction.start_task_failed",
                channel=request.source_context.channel,
                error=str(exc),
            )
            return InteractionResult(
                intent=InteractionIntent.START_TASK,
                trace_id=request.trace_id,
                reply_text="（任务提交失败，请稍后再试。）",
                matched_keywords=keywords,
                decision_reason=f"submit_error:{exc!s}",
            )

        return InteractionResult(
            intent=InteractionIntent.START_TASK,
            trace_id=handle.trace_id,
            task_id=handle.task_id,
            workspace_id=handle.workspace_id,
            plan_id=handle.plan_id,
            task_status=handle.status,
            matched_keywords=keywords,
            decision_reason=reason,
            deduplicated=handle.deduplicated,
        )

    # ── Workspace 查找 / 创建 ─────────────────────────────────────────────────

    async def _find_or_create_workspace(self, request: InboundRequest):  # noqa: ANN202
        """按 (channel, conversation_id) 查找或创建 workspace。

        飞书渠道同时设置 ``feishu_chat_id``，以保证 ApprovalNotifier /
        ProgressNotifier 能找到正确的会话 ID 回推消息。
        """
        from agent_hub.pilot.domain.models import Workspace

        channel = request.source_context.channel
        chat_id = request.source_context.chat_id

        if chat_id:
            workspace = await self._repo.find_workspace_by_conversation(
                channel, chat_id
            )
            if workspace is not None:
                return workspace

        workspace = Workspace(
            title=(request.text or "")[:32] or "Untitled Workspace",
            source_channel=channel,
            source_conversation_id=chat_id,
            feishu_chat_id=chat_id if channel == _SOURCE_FEISHU else None,
            created_by=request.user_context.user_id,
            members=[request.user_context.user_id],
        )
        await self._repo.save(workspace, expected_version=None)
        return workspace

    # ── Profile 推导 ──────────────────────────────────────────────────────────

    @staticmethod
    def _derive_profile(request: InboundRequest) -> PlanProfile:
        ctx = PlanContext(
            raw_text=request.text or "",
            requester_id=request.user_context.user_id,
            title=(request.text or "")[:32] or None,
            attachments=list(request.attachments),
            source_channel=request.source_context.channel,
            source_conversation_id=request.source_context.chat_id,
            source_message_id=request.source_context.message_id,
            metadata={
                "message_type": request.message_type,
                "chat_type": request.source_context.chat_type.value,
                **_task_metadata_from_request(request),
            },
        )
        return derive_plan_profile(ctx)


def _task_metadata_from_request(request: InboundRequest) -> dict[str, object]:
    metadata: dict[str, object] = dict(request.source_context.raw or {})
    if request.source_context.channel != _SOURCE_FEISHU:
        return metadata

    sender_id = (
        request.source_context.sender_id
        or request.user_context.user_id
        or ""
    )
    sender_id_type = request.source_context.sender_id_type or "open_id"
    metadata.setdefault("feishu_source_chat_id", request.source_context.chat_id or "")
    metadata.setdefault("feishu_source_chat_type", request.source_context.chat_type.value)
    metadata.setdefault("feishu_sender_id", sender_id)
    metadata.setdefault("feishu_sender_id_type", sender_id_type)
    metadata.setdefault("feishu_delivery_mode", "private")
    if sender_id:
        metadata.setdefault("feishu_private_receive_id", sender_id)
        metadata.setdefault("feishu_private_receive_id_type", sender_id_type)
        if sender_id_type == "open_id":
            metadata.setdefault("feishu_requester_open_id", sender_id)
    return metadata


__all__ = ["InteractionResult", "InteractionService"]
