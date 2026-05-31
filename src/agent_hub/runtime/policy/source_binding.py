"""Deterministic source-to-agent binding（规范定义位置）。

从 agent_hub.core.binding 迁移至此；core.binding 保留为 re-export 向后兼容层。

This module mirrors OpenClaw's "most-specific-wins" route binding idea in a
small Python form. It resolves inbound source facts to an agent id and a stable
session key before any LLM-driven execution suggestion is applied.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field

from agent_hub.contracts.execution import TaskInput
from agent_hub.contracts.identity import SourceChatType, SourceContext, UserContext

_TOKEN_RE = re.compile(r"[^a-zA-Z0-9_.:-]+")


class BindingRule(BaseModel):
    """One deterministic route binding rule.

    Rules may specify several fields, but the resolver only considers them in a
    fixed tier order: message/thread > chat > user >
    channel/account > channel default > main.
    """

    rule_id: str = Field(default_factory=lambda: f"binding_{uuid.uuid4().hex[:12]}")
    agent_id: str = "main"
    channel: str | None = None
    account_id: str | None = None
    message_id: str | None = None
    thread_id: str | None = None
    chat_id: str | None = None
    chat_type: SourceChatType | None = None
    user_id: str | None = None
    sender_id: str | None = None
    tool_profile: str | None = None
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class BindingResult(BaseModel):
    """Resolved route facts consumed by Pipeline and RiskPolicy."""

    agent_id: str
    route_source: str
    session_key: str
    main_session_key: str
    matched_by: str
    rule_id: str | None = None
    tool_profile: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceBindingResolver:
    """Resolve source facts to an agent and session key."""

    def __init__(
        self,
        rules: Iterable[BindingRule] | None = None,
        *,
        default_agent_id: str = "main",
    ) -> None:
        self._rules = list(rules or [])
        self._default_agent_id = default_agent_id

    @property
    def rules(self) -> list[BindingRule]:
        """Configured binding rules in evaluation order."""
        return list(self._rules)

    def resolve_task(self, task_input: TaskInput) -> BindingResult:
        """Resolve a :class:`TaskInput` to a binding result."""
        return self.resolve(task_input.source_context, task_input.user_context)

    def resolve(
        self,
        source: SourceContext,
        user_context: UserContext,
    ) -> BindingResult:
        """Resolve source/user facts by fixed most-specific-wins tiers."""
        rules = [rule for rule in self._rules if rule.enabled]
        tiers = (
            ("message", self._matches_message_or_thread),
            ("chat", self._matches_chat),
            ("user", lambda r, s, u: self._matches_user(r, s, u)),
            ("channel_account", self._matches_channel_account),
            ("channel_default", self._matches_channel_default),
        )

        for matched_by, predicate in tiers:
            for rule in rules:
                if predicate(rule, source, user_context):
                    return self._build_result(rule.agent_id, matched_by, source, rule)

        return self._build_result(self._default_agent_id, "main", source, None)

    def _build_result(
        self,
        agent_id: str,
        matched_by: str,
        source: SourceContext,
        rule: BindingRule | None,
    ) -> BindingResult:
        return BindingResult(
            agent_id=agent_id,
            route_source=matched_by,
            session_key=build_source_session_key(agent_id, source),
            main_session_key=build_main_session_key(agent_id),
            matched_by=matched_by,
            rule_id=rule.rule_id if rule else None,
            tool_profile=rule.tool_profile if rule else None,
            metadata=dict(rule.metadata) if rule else {},
        )

    @staticmethod
    def _matches_common(rule: BindingRule, source: SourceContext) -> bool:
        if rule.channel is not None and _norm(rule.channel) != _norm(source.channel):
            return False
        if rule.account_id is not None and _norm(rule.account_id) != _norm(source.account_id):
            return False
        return rule.chat_type is None or rule.chat_type == source.chat_type

    def _matches_message_or_thread(
        self,
        rule: BindingRule,
        source: SourceContext,
        user_context: UserContext,
    ) -> bool:
        _ = user_context
        if not self._matches_common(rule, source):
            return False
        if rule.message_id is not None:
            return _norm(rule.message_id) == _norm(source.message_id)
        if rule.thread_id is not None:
            return _norm(rule.thread_id) == _norm(source.thread_id)
        return False

    def _matches_chat(
        self,
        rule: BindingRule,
        source: SourceContext,
        user_context: UserContext,
    ) -> bool:
        _ = user_context
        if not self._matches_common(rule, source):
            return False
        if rule.chat_id is not None:
            return _norm(rule.chat_id) == _norm(source.chat_id)
        return False

    def _matches_user(
        self,
        rule: BindingRule,
        source: SourceContext,
        user_context: UserContext,
    ) -> bool:
        if not self._matches_common(rule, source):
            return False
        if rule.user_id is not None:
            return _norm(rule.user_id) == _norm(user_context.user_id)
        if rule.sender_id is not None:
            return _norm(rule.sender_id) == _norm(source.sender_id)
        return False

    def _matches_channel_account(
        self,
        rule: BindingRule,
        source: SourceContext,
        user_context: UserContext,
    ) -> bool:
        _ = user_context
        return (
            rule.channel is not None
            and rule.account_id is not None
            and self._matches_common(rule, source)
            and not _has_more_specific_scope(rule)
        )

    def _matches_channel_default(
        self,
        rule: BindingRule,
        source: SourceContext,
        user_context: UserContext,
    ) -> bool:
        _ = user_context
        return (
            rule.channel is not None
            and _norm(rule.channel) == _norm(source.channel)
            and rule.account_id is None
            and not _has_more_specific_scope(rule)
        )


def build_main_session_key(agent_id: str) -> str:
    """Build the stable main session key for an agent."""
    return f"agent:{_session_token(agent_id)}:main"


def build_source_session_key(agent_id: str, source: SourceContext) -> str:
    """Build a stable source-scoped session key."""
    agent = _session_token(agent_id)
    channel = _session_token(source.channel or "unknown")
    if source.thread_id:
        return f"agent:{agent}:{channel}:thread:{_session_token(source.thread_id)}"
    if source.chat_id:
        chat_kind = (
            "chat"
            if source.chat_type is SourceChatType.UNKNOWN
            else _session_token(source.chat_type.value)
        )
        return f"agent:{agent}:{channel}:{chat_kind}:{_session_token(source.chat_id)}"
    if source.sender_id:
        return f"agent:{agent}:{channel}:direct:{_session_token(source.sender_id)}"
    return build_main_session_key(agent_id)


def _has_more_specific_scope(rule: BindingRule) -> bool:
    return any((
        rule.message_id,
        rule.thread_id,
        rule.chat_id,
        rule.user_id,
        rule.sender_id,
    ))


def _norm(value: str | None) -> str:
    return (value or "").strip().lower()


def _session_token(value: str) -> str:
    token = _TOKEN_RE.sub("_", value.strip().lower())
    return token.strip("_") or "unknown"


__all__ = [
    "BindingResult",
    "BindingRule",
    "SourceBindingResolver",
    "build_main_session_key",
    "build_source_session_key",
]
