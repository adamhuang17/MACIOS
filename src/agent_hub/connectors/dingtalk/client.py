"""钉钉出站客户端协议骨架。

参照 :class:`agent_hub.connectors.feishu.client.FeishuClientProtocol`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class DingTalkMessageSent:
    message_id: str
    chat_id: str


@runtime_checkable
class DingTalkClientProtocol(Protocol):
    async def send_message(
        self,
        *,
        chat_id: str,
        msg_type: str,
        content: dict[str, Any],
    ) -> DingTalkMessageSent: ...

    async def aclose(self) -> None: ...


class DingTalkApiError(RuntimeError):
    """钉钉开放平台业务错误（骨架）。"""


@dataclass(slots=True)
class FakeDingTalkClient:
    """单测占位。"""

    sent: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.sent is None:
            self.sent = []

    async def send_message(
        self,
        *,
        chat_id: str,
        msg_type: str,
        content: dict[str, Any],
    ) -> DingTalkMessageSent:
        msg_id = f"ding_{len(self.sent) + 1:08d}"
        self.sent.append(
            {
                "chat_id": chat_id,
                "msg_type": msg_type,
                "content": content,
                "message_id": msg_id,
            },
        )
        return DingTalkMessageSent(message_id=msg_id, chat_id=chat_id)

    async def aclose(self) -> None:  # pragma: no cover - no-op
        return None


__all__ = [
    "DingTalkApiError",
    "DingTalkClientProtocol",
    "DingTalkMessageSent",
    "FakeDingTalkClient",
]
