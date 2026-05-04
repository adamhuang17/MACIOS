"""统一连接器协议。

为飞书 / 钉钉 / QQ / OpenClaw 等多通道入站定义最小公共契约。
现有 :mod:`agent_hub.connectors.feishu` 是参照实现，新通道在
M5+ 接入时遵循同一接口签名：

- :class:`InboundMessage` —— 协议无关的入站消息描述（文本 + 渠道
  标识 + 通道原生 message_id）；
- :class:`AckResult` —— 回 HTTP 时所需的最小信息，对应飞书的
  :class:`agent_hub.connectors.feishu.service.FeishuAckResult`；
- :class:`ConnectorWebhookService` —— 仅暴露 ``handle(payload)``，
  上层路由层只需把请求 body 转 dict 即可。

骨架阶段不强制要求老飞书代码兼容这一抽象，新连接器实现时再以
``Protocol`` 形式实现，保持单一职责。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class InboundOutcome(StrEnum):
    """跨连接器统一的处理结果。"""

    CHALLENGE = "challenge"
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    IGNORED = "ignored"
    REJECTED = "rejected"
    CARD_CALLBACK = "card_callback"


@dataclass(frozen=True, slots=True)
class InboundMessage:
    """协议无关的入站消息。"""

    channel: str
    chat_id: str
    message_id: str
    sender_id: str
    text: str = ""
    tenant_key: str | None = None
    app_id: str | None = None
    create_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AckResult:
    """连接器统一回包。"""

    outcome: InboundOutcome
    challenge: str | None = None
    reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ConnectorWebhookService(Protocol):
    """所有通道 webhook 服务的最小接口。"""

    async def handle(self, payload: dict[str, Any]) -> AckResult: ...


__all__ = [
    "AckResult",
    "ConnectorWebhookService",
    "InboundMessage",
    "InboundOutcome",
]
