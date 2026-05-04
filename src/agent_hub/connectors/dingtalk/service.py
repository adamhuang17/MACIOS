"""钉钉 webhook 服务骨架。

参考 :class:`agent_hub.connectors.feishu.service.FeishuWebhookService`：
负责把规范化后的入站消息提交到 :class:`TaskOrchestrator`。
当前仅占位，避免主 app 导入时报错。
"""

from __future__ import annotations

from typing import Any

from agent_hub.connectors.base import AckResult, ConnectorWebhookService, InboundOutcome
from agent_hub.connectors.dingtalk.webhook import DingTalkWebhookProcessor


class DingTalkWebhookService(ConnectorWebhookService):
    """骨架实现：转发到 processor，未实现 task 提交。"""

    def __init__(self, *, processor: DingTalkWebhookProcessor) -> None:
        self._processor = processor

    async def handle(self, payload: dict[str, Any]) -> AckResult:
        return await self._processor.handle_payload(payload)


__all__ = ["DingTalkWebhookService"]


# 防 unused 警告
_ = InboundOutcome
