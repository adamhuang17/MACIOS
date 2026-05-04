"""Feishu connector 边界 DTO。

设计原则：

- 飞书开放平台的原始 payload 字段宽泛（``open_chat_id`` 等可能缺省、
  附件类型多样），因此入参解析层做 *宽进*；
- 内部 DTO ``FeishuInboundMessage`` 严格 ``extra="forbid"``，只保留
  下游真正需要的字段，避免把租户敏感字段无差别透传到事件存储；
- 所有 enum 都是 :class:`StrEnum`，序列化与日志可读。
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FeishuMessageType(StrEnum):
    TEXT = "text"
    POST = "post"
    IMAGE = "image"
    FILE = "file"
    AUDIO = "audio"
    MEDIA = "media"
    STICKER = "sticker"
    INTERACTIVE = "interactive"
    SHARE_CHAT = "share_chat"
    SHARE_USER = "share_user"
    UNKNOWN = "unknown"


class FeishuChatType(StrEnum):
    P2P = "p2p"
    GROUP = "group"
    UNKNOWN = "unknown"


class FeishuAttachment(BaseModel):
    """规范化后的飞书附件描述（图片 / 文件 / 多媒体）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: FeishuMessageType
    file_key: str
    file_name: str | None = None
    mime_type: str | None = None
    size: int | None = None


class FeishuInboundMessage(BaseModel):
    """飞书入站消息的内部表示。

    任何与具体协议无关的下游服务（``FeishuWebhookService``、tests
    等）都基于本结构工作，避免依赖飞书事件 schema 细节。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    event_type: str
    tenant_key: str | None = None
    app_id: str | None = None
    chat_id: str
    chat_type: FeishuChatType = FeishuChatType.UNKNOWN
    message_id: str
    message_type: FeishuMessageType = FeishuMessageType.UNKNOWN
    sender_id: str
    sender_id_type: str = "open_id"
    text: str = ""
    mentions: list[str] = Field(default_factory=list)
    bot_mentioned: bool = False
    attachments: list[FeishuAttachment] = Field(default_factory=list)
    create_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    raw_digest: str = Field(default="")


class FeishuWebhookOutcome(StrEnum):
    """webhook 处理结果，方便测试与日志统一断言。"""

    CHALLENGE = "challenge"
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    IGNORED = "ignored"
    REJECTED = "rejected"
    CARD_CALLBACK = "card_callback"


class FeishuCardActionValue(BaseModel):
    """飞书审批卡片在 ``value`` 字段中捆绑的业务语义。

    M5 骨架：送出时由 ``feishu.card.send_approval`` 在按钮上損
    入 ``approval_id`` 与 ``decision`` （``approve`` / ``reject``），
    回调时原样返回，:class:`FeishuWebhookProcessor` 负责解析。
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    approval_id: str
    decision: str
    comment: str = ""


class FeishuCardCallback(BaseModel):
    """交互卡片 ``card.action.trigger`` 解析后的内部 DTO。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    tenant_key: str | None = None
    app_id: str | None = None
    operator_open_id: str
    chat_id: str = ""
    message_id: str = ""
    action_value: FeishuCardActionValue
    create_time: datetime = Field(default_factory=lambda: datetime.now(UTC))


__all__ = [
    "FeishuAttachment",
    "FeishuCardActionValue",
    "FeishuCardCallback",
    "FeishuChatType",
    "FeishuInboundMessage",
    "FeishuMessageType",
    "FeishuWebhookOutcome",
]


# 防 unused 警告（typing 中 Any 仅供后续扩展使用）。
_ = Any
