"""把规范化的飞书消息接到 ``TaskOrchestrator`` 上。

只关心三件事：

1. 通过 ``feishu_chat_id`` 复用 / 创建 :class:`Workspace`；
2. 拼装 :class:`TaskRequest` 并提交；
3. 通过 :class:`FeishuClientProtocol` 回复一条 ACK 文本，方便用户感知
   接收成功。ACK 不走审批，属于连接器自身的事务性消息。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from agent_hub.connectors.feishu.client import FeishuClientProtocol
from agent_hub.connectors.feishu.models import (
    FeishuCardCallback,
    FeishuChatType,
    FeishuInboundMessage,
    FeishuWebhookOutcome,
)
from agent_hub.connectors.feishu.webhook import (
    FeishuWebhookProcessor,
    FeishuWebhookResult,
)
from agent_hub.pilot.domain.enums import WorkspaceStatus
from agent_hub.pilot.domain.models import Workspace
from agent_hub.pilot.services.dto import PlanProfile, TaskHandle, TaskRequest
from agent_hub.pilot.services.ingress import IngressIntent, PilotIngressService

if TYPE_CHECKING:
    from agent_hub.pilot.services.commands import PilotCommandService
    from agent_hub.pilot.services.orchestrator import TaskOrchestrator
    from agent_hub.pilot.services.repository import PilotRepository

logger = structlog.get_logger(__name__)

SOURCE_CHANNEL = "feishu"


@dataclass(frozen=True, slots=True)
class FeishuAckResult:
    outcome: FeishuWebhookOutcome
    challenge: str | None = None
    handle: TaskHandle | None = None
    reason: str | None = None
    ack_message_id: str | None = None
    approval_decision: dict[str, object] | None = None
    intent: IngressIntent | None = None
    reply_message_id: str | None = None


class FeishuWebhookService:
    """webhook 入站处理服务。

    Args:
        processor: webhook 解析器（验签 + 解密 + 去重 + 规范化）。
        orchestrator: 提交 task 的入口。
        repository: 用于按 ``feishu_chat_id`` 查找已有 workspace。
        client: 飞书出站客户端，用于回 ACK；为 ``None`` 时不回复
            （单元测试或 dry-run 部署）。
        ack_template: ACK 文案模板，``{title}`` 会被替换为 task 截断
            后的标题。
        send_ack: 是否发送 ACK；关闭后只创建 task。
    """

    def __init__(
        self,
        *,
        processor: FeishuWebhookProcessor,
        orchestrator: TaskOrchestrator,
        repository: PilotRepository,
        client: FeishuClientProtocol | None = None,
        commands: PilotCommandService | None = None,
        ingress: PilotIngressService | None = None,
        ack_template: str = "已收到任务，正在规划：{title}",
        send_ack: bool = True,
    ) -> None:
        self._processor = processor
        self._orchestrator = orchestrator
        self._repo = repository
        self._client = client
        self._commands = commands
        self._ingress = ingress
        self._ack_template = ack_template
        self._send_ack = send_ack

    async def handle(self, payload: dict) -> FeishuAckResult:
        result = await self._processor.handle_payload(payload)
        return await self._process_result(result)

    async def handle_event_dict(self, payload: dict) -> FeishuAckResult:
        """长连接模式入口：跳过验签/解密，直接处理已解包事件 dict。"""
        result = await self._processor.handle_event_dict(payload)
        return await self._process_result(result)

    async def _process_result(self, result: FeishuWebhookResult) -> FeishuAckResult:
        """将 FeishuWebhookResult 路由到下游业务逻辑。"""
        if result.outcome is FeishuWebhookOutcome.CHALLENGE:
            return FeishuAckResult(
                outcome=result.outcome,
                challenge=result.challenge,
            )
        if result.outcome is FeishuWebhookOutcome.CARD_CALLBACK:
            return await self._handle_card_callback(result.card_callback)
        if result.outcome is not FeishuWebhookOutcome.ACCEPTED:
            return FeishuAckResult(outcome=result.outcome, reason=result.reason)

        message = result.message
        if message is None:  # pragma: no cover - 防御
            return FeishuAckResult(
                outcome=FeishuWebhookOutcome.IGNORED,
                reason="processor returned no message",
            )

        # 走 Ingress 分流：决定本条消息是普通问答、进度查询，还是真任务。
        if self._ingress is not None:
            decision = await self._ingress.dispatch(message)
            logger.info(
                "feishu.ingress.classified",
                chat_id=message.chat_id,
                intent=decision.intent.value,
                matched=list(decision.matched_keywords),
                reason=decision.reason,
            )
            if decision.intent is IngressIntent.IGNORE:
                return FeishuAckResult(
                    outcome=FeishuWebhookOutcome.IGNORED,
                    reason=decision.reason or "ingress_ignore",
                    intent=decision.intent,
                )
            if decision.intent in (
                IngressIntent.ORDINARY_QA,
                IngressIntent.PROGRESS_QUERY,
            ):
                reply_id = await self._send_text_reply(
                    message, decision.reply_text
                )
                return FeishuAckResult(
                    outcome=FeishuWebhookOutcome.ACCEPTED,
                    intent=decision.intent,
                    reply_message_id=reply_id,
                )
            # START_TASK fallthrough
            handle = await self._submit_task(
                message,
                plan_profile=decision.plan_profile,
            )
            ack_message_id = await self._maybe_ack(message, handle)
            return FeishuAckResult(
                outcome=FeishuWebhookOutcome.ACCEPTED,
                handle=handle,
                ack_message_id=ack_message_id,
                intent=decision.intent,
            )

        handle = await self._submit_task(message)
        ack_message_id = await self._maybe_ack(message, handle)
        return FeishuAckResult(
            outcome=FeishuWebhookOutcome.ACCEPTED,
            handle=handle,
            ack_message_id=ack_message_id,
        )

    # ── 内部 ───────────────────────────────────────

    async def _handle_card_callback(
        self,
        callback: FeishuCardCallback | None,
    ) -> FeishuAckResult:
        """M5 审批卡片回调 → PilotCommandService.decide_approval。

        骨架实现：没有注入 ``commands`` 时静默 ignore，保证后向兼容。
        后续可加入"更新卡片状态"、"跳转产物链接"等逻辑。
        """
        if callback is None:  # pragma: no cover - 防御
            return FeishuAckResult(
                outcome=FeishuWebhookOutcome.IGNORED,
                reason="empty card callback",
            )
        if self._commands is None:
            logger.info(
                "feishu.card.ignored_no_commands",
                approval_id=callback.action_value.approval_id,
            )
            return FeishuAckResult(
                outcome=FeishuWebhookOutcome.CARD_CALLBACK,
                reason="commands service not wired",
            )

        decision = callback.action_value.decision.lower()
        if decision not in ("approve", "reject"):
            return FeishuAckResult(
                outcome=FeishuWebhookOutcome.IGNORED,
                reason=f"unsupported decision: {decision}",
            )
        try:
            outcome = await self._commands.decide_approval(
                callback.action_value.approval_id,
                decision=decision,  # type: ignore[arg-type]
                actor_id=callback.operator_open_id or "feishu",
                comment=callback.action_value.comment or None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "feishu.card.decision_failed",
                approval_id=callback.action_value.approval_id,
                error=str(exc),
            )
            return FeishuAckResult(
                outcome=FeishuWebhookOutcome.REJECTED,
                reason=str(exc),
            )
        logger.info(
            "feishu.card.decision_applied",
            approval_id=callback.action_value.approval_id,
            decision=decision,
            actor_id=callback.operator_open_id,
        )
        return FeishuAckResult(
            outcome=FeishuWebhookOutcome.CARD_CALLBACK,
            approval_decision=outcome,
        )

    async def _submit_task(
        self,
        message: FeishuInboundMessage,
        *,
        plan_profile: PlanProfile | None = None,
    ) -> TaskHandle:
        workspace = await self._find_or_create_workspace(message)
        request = TaskRequest(
            source_channel=SOURCE_CHANNEL,
            raw_text=message.text or "",
            requester_id=message.sender_id,
            title=_derive_title(message),
            source_conversation_id=message.chat_id,
            source_message_id=message.message_id,
            workspace_id=workspace.workspace_id,
            tenant_key=message.tenant_key,
            attachments=[a.file_key for a in message.attachments],
            idempotency_key=_idempotency_key(message),
            auto_approve=True,
            plan_profile=plan_profile,
            metadata={
                "feishu_event_id": message.event_id,
                "feishu_chat_type": message.chat_type.value,
                "feishu_message_type": message.message_type.value,
                "feishu_app_id": message.app_id or "",
                "feishu_bot_mentioned": message.bot_mentioned,
                "plan_profile": plan_profile.value if plan_profile else "",
            },
        )
        handle = await self._orchestrator.submit(request)
        logger.info(
            "feishu.webhook.task_submitted",
            workspace_id=handle.workspace_id,
            task_id=handle.task_id,
            chat_id=message.chat_id,
            deduplicated=handle.deduplicated,
        )
        return handle

    async def _find_or_create_workspace(self, message: FeishuInboundMessage) -> Workspace:
        workspaces = await self._repo.list_workspaces()
        for ws in workspaces:
            if ws.feishu_chat_id == message.chat_id:
                return ws
        workspace = Workspace(
            tenant_key=message.tenant_key,
            title=_derive_title(message),
            source_channel=SOURCE_CHANNEL,
            source_conversation_id=message.chat_id,
            feishu_chat_id=message.chat_id,
            created_by=message.sender_id,
            members=[message.sender_id],
            status=WorkspaceStatus.ACTIVE,
            metadata={
                "feishu_chat_type": message.chat_type.value,
                "feishu_app_id": message.app_id or "",
            },
        )
        await self._repo.save(workspace, expected_version=None)
        return workspace

    async def _maybe_ack(
        self,
        message: FeishuInboundMessage,
        handle: TaskHandle,
    ) -> str | None:
        if not self._send_ack or self._client is None:
            return None
        if handle.deduplicated:
            return None
        title = _derive_title(message)
        text = self._ack_template.format(title=title)
        try:
            sent = await self._client.send_message(
                receive_id=message.chat_id,
                receive_id_type="chat_id",
                msg_type="text",
                content=json.dumps({"text": text}, ensure_ascii=False),
            )
        except Exception as exc:  # noqa: BLE001 - 防御性吞掉，避免 ACK 失败影响主流程
            logger.warning(
                "feishu.webhook.ack_failed",
                chat_id=message.chat_id,
                task_id=handle.task_id,
                error=str(exc),
            )
            return None
        return sent.message_id

    async def _send_text_reply(
        self,
        message: FeishuInboundMessage,
        text: str | None,
    ) -> str | None:
        """普通问答 / 进度查询的回复发送，失败仅打日志不影响主流程。"""
        if not text or self._client is None:
            return None
        try:
            sent = await self._client.send_message(
                receive_id=message.chat_id,
                receive_id_type="chat_id",
                msg_type="text",
                content=json.dumps({"text": text}, ensure_ascii=False),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "feishu.ingress.reply_failed",
                chat_id=message.chat_id,
                error=str(exc),
            )
            return None
        return sent.message_id


def _derive_title(message: FeishuInboundMessage) -> str:
    text = (message.text or "").strip()
    if text:
        first_line = text.splitlines()[0]
        return first_line[:32] or "飞书任务"
    if message.chat_type is FeishuChatType.P2P:
        return "飞书私聊任务"
    return "飞书群聊任务"


def _idempotency_key(message: FeishuInboundMessage) -> str:
    tenant = message.tenant_key or "unknown"
    base = message.event_id or message.message_id
    return f"feishu:{tenant}:{base}"


__all__ = [
    "FeishuAckResult",
    "FeishuWebhookService",
    "SOURCE_CHANNEL",
]


# 防 unused import 警告（FeishuWebhookResult 在 docstring 引用）。
_ = FeishuWebhookResult
