"""钉钉 webhook 解析骨架。

参考 :mod:`agent_hub.connectors.feishu.webhook`：
- 校验 ``timestamp`` + ``sign`` 头部（HMAC-SHA256，钉钉规范）；
- 处理 ``url verification`` 与 ``message`` 事件；
- 卡片回调走 ``actionCard.callback`` 事件类型。

骨架阶段仅返回 :class:`agent_hub.connectors.base.AckResult`，把
真实 schema 解析留待后续。
"""

from __future__ import annotations

from typing import Any

from agent_hub.connectors.base import AckResult, InboundOutcome


class DingTalkWebhookError(RuntimeError):
    """webhook 校验或解析错误。"""


class DingTalkWebhookProcessor:
    """钉钉 webhook 处理器（骨架）。"""

    def __init__(
        self,
        *,
        app_secret: str = "",
        bot_id: str = "",
    ) -> None:
        self._app_secret = app_secret
        self._bot_id = bot_id

    async def handle_payload(self, payload: dict[str, Any]) -> AckResult:
        if not isinstance(payload, dict):
            raise DingTalkWebhookError("payload must be a JSON object")
        # TODO: 校验签名、解析事件、规范化 InboundMessage / 卡片回调。
        event_type = str(payload.get("eventType") or payload.get("type", ""))
        if not event_type:
            return AckResult(outcome=InboundOutcome.IGNORED, reason="missing eventType")
        return AckResult(
            outcome=InboundOutcome.IGNORED,
            reason=f"dingtalk.{event_type} not implemented",
        )


__all__ = ["DingTalkWebhookError", "DingTalkWebhookProcessor"]
