"""M5 飞书审批卡片 skills（骨架）。

提供两个 skill：

- ``feishu.card.send_approval`` —— 把 :class:`Approval` 渲染为飞书
  interactive 卡片并发送，返回 ``message_id``，写入到产物 metadata
  以便后续 ``card.action.trigger`` 回调时把审批决议写回。
- ``feishu.card.update_after_decision`` —— 决议后用最终态重绘卡片
  （置灰按钮 + 标注审批人）。

设计要点：

- 与 :mod:`agent_hub.pilot.skills.feishu` 同源：通过闭包注入
  :class:`FeishuClientProtocol`，不直接依赖 HTTP；
- 渲染逻辑放在纯函数 :func:`build_approval_card` /
  :func:`build_decision_card` 中，便于单测；
- ``action.value`` 中固定写入 ``approval_id`` / ``decision``，与
  :class:`agent_hub.connectors.feishu.models.FeishuCardActionValue`
  schema 对齐，使 webhook 回调能直接 model_validate。
"""

from __future__ import annotations

from typing import Any

from agent_hub.connectors.feishu.client import FeishuApiError, FeishuClientProtocol
from agent_hub.pilot.domain.enums import ArtifactType
from agent_hub.pilot.skills.registry import (
    SkillInvocation,
    SkillRegistry,
    SkillResult,
    SkillSpec,
)

FEISHU_CARD_SCOPES = ["feishu", "feishu.card"]


# ── 卡片渲染 ────────────────────────────────────────


def build_approval_card(
    *,
    approval_id: str,
    title: str,
    summary: str,
    detail: str = "",
) -> dict[str, Any]:
    """渲染审批申请卡片。

    返回符合飞书 interactive 卡片 schema 的 dict；按钮 ``value`` 中
    的 ``approval_id`` / ``decision`` 与 webhook 回调解析侧严格对齐。
    """
    elements: list[dict[str, Any]] = [
        {
            "tag": "div",
            "text": {"tag": "lark_md", "content": summary or "（无摘要）"},
        },
    ]
    if detail:
        elements.append(
            {"tag": "div", "text": {"tag": "lark_md", "content": detail}},
        )
    elements.append(
        {
            "tag": "action",
            "actions": [
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "✅ 批准"},
                    "type": "primary",
                    "value": {
                        "approval_id": approval_id,
                        "decision": "approve",
                    },
                },
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "❌ 拒绝"},
                    "type": "danger",
                    "value": {
                        "approval_id": approval_id,
                        "decision": "reject",
                    },
                },
            ],
        },
    )
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": "blue",
        },
        "elements": elements,
    }


def build_decision_card(
    *,
    title: str,
    summary: str,
    decision: str,
    actor_id: str,
    comment: str = "",
) -> dict[str, Any]:
    """决议后用于覆盖原卡片的最终态卡片。"""
    color = "green" if decision == "approve" else "red"
    label = "已批准" if decision == "approve" else "已拒绝"
    lines = [summary or "（无摘要）", "", f"**决议**: {label}", f"**审批人**: {actor_id}"]
    if comment:
        lines.append(f"**备注**: {comment}")
    return {
        "config": {"wide_screen_mode": True, "update_multi": True},
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": color,
        },
        "elements": [
            {"tag": "div", "text": {"tag": "lark_md", "content": "\n".join(lines)}},
        ],
    }


# ── feishu.card.send_approval ──────────────────────


def _make_send_approval_skill(client: FeishuClientProtocol) -> tuple[SkillSpec, Any]:
    async def _send(inv: SkillInvocation) -> SkillResult:
        params = inv.params
        chat_id = str(params.get("chat_id", "") or "")
        approval_id = str(params.get("approval_id", "") or "")
        title = str(params.get("title", "") or "审批请求")
        summary = str(params.get("summary", "") or "")
        detail = str(params.get("detail", "") or "")
        receive_id_type = str(params.get("receive_id_type", "chat_id") or "chat_id")
        if not chat_id or not approval_id:
            return SkillResult(
                skill_name=inv.skill_name,
                success=False,
                error="missing required param: chat_id / approval_id",
            )

        card = build_approval_card(
            approval_id=approval_id,
            title=title,
            summary=summary,
            detail=detail,
        )

        if inv.dry_run:
            return SkillResult(
                skill_name=inv.skill_name,
                success=True,
                output={
                    "chat_id": chat_id,
                    "approval_id": approval_id,
                    "would_send": True,
                    "card_preview": card,
                },
            )

        try:
            sent = await client.send_interactive_card(
                receive_id=chat_id,
                receive_id_type=receive_id_type,
                card=card,
            )
        except FeishuApiError as exc:
            return SkillResult(skill_name=inv.skill_name, success=False, error=str(exc))

        return SkillResult(
            skill_name=inv.skill_name,
            success=True,
            output={
                "message_id": sent.message_id,
                "chat_id": sent.chat_id,
                "approval_id": approval_id,
            },
            artifact_payload={
                "type": ArtifactType.SUMMARY.value,
                "title": f"飞书审批卡片：{title}",
                "mime_type": "application/json",
                "content": {
                    "approval_id": approval_id,
                    "message_id": sent.message_id,
                    "chat_id": sent.chat_id,
                    "card": card,
                },
                "metadata": {
                    "approval_id": approval_id,
                    "feishu_card_message_id": sent.message_id,
                    "feishu_chat_id": sent.chat_id,
                },
            },
        )

    spec = SkillSpec(
        skill_name="feishu.card.send_approval",
        description="把审批请求渲染为飞书 interactive 卡片并发送。",
        side_effect="share",
        requires_approval=False,
        supports_dry_run=True,
        scopes=FEISHU_CARD_SCOPES,
        idempotency_key_fields=["approval_id", "chat_id"],
        timeout_ms=15_000,
    )
    return spec, _send


# ── feishu.card.update_after_decision ──────────────


def _make_update_card_skill(client: FeishuClientProtocol) -> tuple[SkillSpec, Any]:
    async def _update(inv: SkillInvocation) -> SkillResult:
        params = inv.params
        message_id = str(params.get("message_id", "") or "")
        title = str(params.get("title", "") or "审批请求")
        summary = str(params.get("summary", "") or "")
        decision = str(params.get("decision", "") or "").lower()
        actor_id = str(params.get("actor_id", "") or "")
        comment = str(params.get("comment", "") or "")

        if not message_id or decision not in ("approve", "reject"):
            return SkillResult(
                skill_name=inv.skill_name,
                success=False,
                error="missing message_id or invalid decision",
            )

        card = build_decision_card(
            title=title,
            summary=summary,
            decision=decision,
            actor_id=actor_id,
            comment=comment,
        )

        if inv.dry_run:
            return SkillResult(
                skill_name=inv.skill_name,
                success=True,
                output={"message_id": message_id, "would_update": True, "card_preview": card},
            )

        try:
            await client.update_card(message_id=message_id, card=card)
        except FeishuApiError as exc:
            return SkillResult(skill_name=inv.skill_name, success=False, error=str(exc))

        return SkillResult(
            skill_name=inv.skill_name,
            success=True,
            output={"message_id": message_id, "decision": decision},
        )

    spec = SkillSpec(
        skill_name="feishu.card.update_after_decision",
        description="决议后覆盖飞书审批卡片为最终态。",
        side_effect="share",
        requires_approval=False,
        supports_dry_run=True,
        scopes=FEISHU_CARD_SCOPES,
        idempotency_key_fields=["message_id", "decision"],
        timeout_ms=10_000,
    )
    return spec, _update


# ── 注册入口 ────────────────────────────────────────


def register_feishu_card_skills(
    registry: SkillRegistry,
    client: FeishuClientProtocol,
    *,
    allow_overwrite: bool = False,
) -> None:
    """把 M5 飞书审批卡片技能注册到 ``registry``。"""
    spec, fn = _make_send_approval_skill(client)
    registry.register(spec, fn, allow_overwrite=allow_overwrite)

    spec, fn = _make_update_card_skill(client)
    registry.register(spec, fn, allow_overwrite=allow_overwrite)


__all__ = [
    "FEISHU_CARD_SCOPES",
    "build_approval_card",
    "build_decision_card",
    "register_feishu_card_skills",
]
