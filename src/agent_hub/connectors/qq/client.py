"""QQ 出站客户端骨架。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class QQMessageSent:
    message_id: str
    chat_id: str


@runtime_checkable
class QQClientProtocol(Protocol):
    async def send_message(
        self,
        *,
        chat_id: str,
        text: str,
    ) -> QQMessageSent: ...

    async def aclose(self) -> None: ...


@dataclass(slots=True)
class FakeQQClient:
    sent: list[dict[str, Any]] = field(default_factory=list)

    async def send_message(self, *, chat_id: str, text: str) -> QQMessageSent:
        msg_id = f"qq_{len(self.sent) + 1:08d}"
        self.sent.append({"chat_id": chat_id, "text": text, "message_id": msg_id})
        return QQMessageSent(message_id=msg_id, chat_id=chat_id)

    async def aclose(self) -> None:  # pragma: no cover - no-op
        return None


__all__ = ["FakeQQClient", "QQClientProtocol", "QQMessageSent"]
