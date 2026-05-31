"""跨层身份与来源上下文契约。

定义：用户身份、渠道类型、消息来源上下文。
这些类型是系统之间传递"谁发出了这条请求、从哪里来"的唯一公共格式。
不依赖 pydantic 以外的任何 agent_hub 内部模块。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ── 枚举 ──────────────────────────────────────────────────────────────────────


class UserRole(StrEnum):
    """用户角色。

    ADMIN 拥有操控远端系统的权限；USER 只可发起标准请求。
    """

    ADMIN = "admin"
    USER = "user"


class ChannelType(StrEnum):
    """消息来源渠道类型。"""

    API = "api"
    WEBHOOK = "webhook"
    FEISHU = "feishu"
    UNKNOWN = "unknown"


class SourceChatType(StrEnum):
    """来源会话类型。"""

    DIRECT = "direct"
    GROUP = "group"
    CHANNEL = "channel"
    UNKNOWN = "unknown"


# ── 模型 ──────────────────────────────────────────────────────────────────────


class UserContext(BaseModel):
    """用户上下文：描述调用方身份与权限事实。"""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    role: UserRole


class SourceContext(BaseModel):
    """消息来源上下文：描述请求来自哪个渠道、会话和消息。"""

    model_config = ConfigDict(extra="forbid")

    channel: str
    account_id: str | None = None
    chat_id: str | None = None
    chat_type: SourceChatType = SourceChatType.UNKNOWN
    thread_id: str | None = None
    message_id: str | None = None
    sender_id: str | None = None
    sender_id_type: str | None = None
    team_id: str | None = None
    guild_id: str | None = None
    member_role_ids: list[str] = Field(default_factory=list)
    is_at_bot: bool = False
    is_command_prefix: bool = False
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_user_context(
        cls,
        user_context: UserContext,
        *,
        channel: str = "unknown",
        is_at_bot: bool = False,
        is_command_prefix: bool = False,
    ) -> SourceContext:
        """从身份上下文构造最小来源上下文。"""
        return cls(
            channel=channel,
            chat_type=SourceChatType.DIRECT,
            sender_id=user_context.user_id,
            is_at_bot=is_at_bot,
            is_command_prefix=is_command_prefix,
        )


__all__ = [
    "ChannelType",
    "SourceChatType",
    "SourceContext",
    "UserContext",
    "UserRole",
]
