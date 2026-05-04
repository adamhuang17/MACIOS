"""OpenClaw webhook 骨架。

设计目标：作为通用 webhook 适配层，把任意上游平台（自定义 IM、
工单系统等）经由 OpenClaw 网关转发的请求规范化为
:class:`agent_hub.connectors.base.InboundMessage`。

当前仅占位。
"""

from __future__ import annotations

from typing import Any

from agent_hub.connectors.base import AckResult, ConnectorWebhookService, InboundOutcome


class OpenClawWebhookError(RuntimeError):
    """OpenClaw webhook 校验或解析错误。"""


class OpenClawWebhookService(ConnectorWebhookService):
    async def handle(self, payload: dict[str, Any]) -> AckResult:
        if not isinstance(payload, dict):
            raise OpenClawWebhookError("payload must be a JSON object")
        return AckResult(
            outcome=InboundOutcome.IGNORED,
            reason="openclaw webhook not implemented",
        )


__all__ = ["OpenClawWebhookError", "OpenClawWebhookService"]
