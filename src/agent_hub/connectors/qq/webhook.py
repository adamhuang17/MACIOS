"""QQ webhook 骨架。

预留给 NoneBot OneBot v11 / 官方频道 OpenAPI 接入。
"""

from __future__ import annotations

from typing import Any

from agent_hub.connectors.base import AckResult, ConnectorWebhookService, InboundOutcome


class QQWebhookError(RuntimeError):
    """QQ webhook 校验或解析错误。"""


class QQWebhookService(ConnectorWebhookService):
    """骨架实现，待补 OneBot v11 反向 WebSocket / 官方频道签名校验。"""

    async def handle(self, payload: dict[str, Any]) -> AckResult:
        if not isinstance(payload, dict):
            raise QQWebhookError("payload must be a JSON object")
        return AckResult(
            outcome=InboundOutcome.IGNORED,
            reason="qq webhook not implemented",
        )


__all__ = ["QQWebhookError", "QQWebhookService"]
