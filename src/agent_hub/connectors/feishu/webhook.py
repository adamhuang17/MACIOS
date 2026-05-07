"""飞书 webhook 验签 / 解密 / 去重 / 规范化。

设计要点：

- :meth:`FeishuWebhookProcessor.handle_payload` 是唯一对外入口，
  返回 :class:`FeishuWebhookResult`，FastAPI 路由层只负责把它映射成
  HTTP 响应；
- URL verification (challenge) 与正式事件复用同一个入口；
- ``encrypt`` 字段存在时才会触发 AES 解密，否则按明文 JSON 处理。
  AES 解密依赖 ``cryptography``，缺失时会以 :class:`FeishuWebhookError`
  失败并提示用户安装；
- ``event_id`` 的去重通过 TTL + 容量上限的 LRU dict 实现，避免无界
  内存增长；
- 规范化阶段对 ``im.message.receive_v1`` 做 best-effort 解析，其它
  事件统一返回 :attr:`FeishuWebhookOutcome.IGNORED`，让上层只处理
  自己关心的事件，不引入新的 enum。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import structlog

from agent_hub.connectors.feishu.models import (
    FeishuAttachment,
    FeishuCardActionValue,
    FeishuCardCallback,
    FeishuChatType,
    FeishuInboundMessage,
    FeishuMessageType,
    FeishuWebhookOutcome,
)

logger = structlog.get_logger(__name__)


class FeishuWebhookError(RuntimeError):
    """webhook 解析失败（含验签失败、加密解析失败、payload 非法）。"""


@dataclass(frozen=True, slots=True)
class FeishuWebhookResult:
    outcome: FeishuWebhookOutcome
    challenge: str | None = None
    message: FeishuInboundMessage | None = None
    card_callback: FeishuCardCallback | None = None
    reason: str | None = None


class _TtlLru:
    """带 TTL 与最大容量的 LRU dict，用于事件去重。"""

    def __init__(self, *, max_entries: int, ttl_seconds: int) -> None:
        self._max = max(max_entries, 16)
        self._ttl = max(ttl_seconds, 1)
        self._items: OrderedDict[str, float] = OrderedDict()
        self._lock = asyncio.Lock()

    async def add_if_absent(self, key: str) -> bool:
        """返回 True 表示首次出现；False 表示重复。"""
        now = time.monotonic()
        async with self._lock:
            self._evict(now)
            if key in self._items:
                # 重复事件刷新 LRU 顺序但仍返回 False。
                self._items.move_to_end(key)
                return False
            self._items[key] = now + self._ttl
            self._items.move_to_end(key)
            while len(self._items) > self._max:
                self._items.popitem(last=False)
            return True

    def _evict(self, now: float) -> None:
        # OrderedDict 末尾是最近写入；前缀是最早；扫一段过期即可。
        expired: list[str] = []
        for key, expires_at in self._items.items():
            if expires_at <= now:
                expired.append(key)
            else:
                break
        for key in expired:
            self._items.pop(key, None)


class FeishuWebhookProcessor:
    """承载验签 / 解密 / 去重 / 规范化的纯逻辑组件。

    Args:
        verification_token: 事件订阅校验 token；为空时跳过校验
            （仅供本地开发，**不要**在生产为空）。
        encrypt_key: AES-256 key；为空时不解密。
        bot_open_id: 用于识别 ``@bot``；为空表示不强制 mention 才触发。
        require_mention_in_group: 群聊里是否必须 ``@bot`` 才视为命中。
        trigger_keywords: 命中关键字才触发；空集合表示不过滤。
        dedup_ttl_seconds / dedup_max_entries: 事件去重 LRU 配置。
    """

    def __init__(
        self,
        *,
        verification_token: str = "",
        encrypt_key: str = "",
        bot_open_id: str = "",
        require_mention_in_group: bool = True,
        trigger_keywords: set[str] | None = None,
        dedup_ttl_seconds: int = 600,
        dedup_max_entries: int = 4096,
    ) -> None:
        self._verification_token = verification_token
        self._encrypt_key = encrypt_key
        self._bot_open_id = bot_open_id
        self._require_mention_in_group = require_mention_in_group
        self._keywords = trigger_keywords or set()
        self._dedup = _TtlLru(
            max_entries=dedup_max_entries,
            ttl_seconds=dedup_ttl_seconds,
        )

    async def handle_payload(self, payload: dict[str, Any]) -> FeishuWebhookResult:
        """处理一条 webhook 入站 payload（含验签 / 解密）。"""
        if not isinstance(payload, dict):
            raise FeishuWebhookError("payload must be a JSON object")

        if "encrypt" in payload:
            payload = self._decrypt_envelope(payload)

        # URL verification challenge（schema=v1 即 type=url_verification；
        # schema=v2 也能命中，header.event_type 不同但都会带 challenge）。
        if payload.get("type") == "url_verification" or "challenge" in payload:
            self._verify_token_v1(payload)
            challenge = str(payload.get("challenge", ""))
            return FeishuWebhookResult(
                outcome=FeishuWebhookOutcome.CHALLENGE,
                challenge=challenge,
            )

        header = payload.get("header") or {}
        if header:
            self._verify_token_v2(header)
            event_type = str(header.get("event_type", ""))
            event_id = str(header.get("event_id", ""))
            tenant_key = header.get("tenant_key")
            app_id = header.get("app_id")
        else:
            self._verify_token_v1(payload)
            event_payload = payload.get("event") or {}
            event_type = str(payload.get("event_type") or event_payload.get("type", ""))
            event_id = str(payload.get("uuid", ""))
            tenant_key = payload.get("tenant_key")
            app_id = (payload.get("app_id") or "")

        return await self._dispatch(payload, header, event_type, event_id, tenant_key, app_id)

    async def handle_event_dict(self, payload: dict[str, Any]) -> FeishuWebhookResult:
        """处理已解包的事件 dict，跳过验签与解密（供长连接模式使用）。

        WebSocket 长连接在连接层已完成鉴权，无需 token 校验和 AES 解密。
        """
        if not isinstance(payload, dict):
            raise FeishuWebhookError("payload must be a JSON object")

        header = payload.get("header") or {}
        if header:
            event_type = str(header.get("event_type", ""))
            event_id = str(header.get("event_id", ""))
            tenant_key = header.get("tenant_key")
            app_id = header.get("app_id")
        else:
            event_payload = payload.get("event") or {}
            event_type = str(payload.get("event_type") or event_payload.get("type", ""))
            event_id = str(payload.get("uuid", ""))
            tenant_key = payload.get("tenant_key")
            app_id = (payload.get("app_id") or "")

        return await self._dispatch(payload, header, event_type, event_id, tenant_key, app_id)

    async def _dispatch(
        self,
        payload: dict[str, Any],
        header: dict[str, Any],
        event_type: str,
        event_id: str,
        tenant_key: Any,  # noqa: ANN401
        app_id: Any,  # noqa: ANN401
    ) -> FeishuWebhookResult:
        """内部：路由 / 去重 / 规范化 / 过滤（已完成验签与解密）。"""
        if event_type != "im.message.receive_v1":
            if event_type in ("card.action.trigger", "p2.card.action.trigger"):
                # M5 飞书审批卡片回调：与消息事件复用 webhook URL，但走单独
                # 的解析路径，返回 CARD_CALLBACK，由 service 路由到
                # PilotCommandService.decide_approval。
                callback = _normalize_card_callback(
                    payload=payload,
                    event=payload.get("event") or {},
                    header=header,
                    event_id=event_id,
                    tenant_key=str(tenant_key) if tenant_key else None,
                    app_id=str(app_id) if app_id else None,
                )
                if callback is None:
                    return FeishuWebhookResult(
                        outcome=FeishuWebhookOutcome.IGNORED,
                        reason="card callback payload not parseable",
                    )
                if event_id and not await self._dedup.add_if_absent(event_id):
                    return FeishuWebhookResult(
                        outcome=FeishuWebhookOutcome.DUPLICATE,
                        reason="duplicate event_id",
                    )
                return FeishuWebhookResult(
                    outcome=FeishuWebhookOutcome.CARD_CALLBACK,
                    card_callback=callback,
                )
            return FeishuWebhookResult(
                outcome=FeishuWebhookOutcome.IGNORED,
                reason=f"unsupported event_type: {event_type}",
            )

        if event_id and not await self._dedup.add_if_absent(event_id):
            return FeishuWebhookResult(
                outcome=FeishuWebhookOutcome.DUPLICATE,
                reason="duplicate event_id",
            )

        message = _normalize_inbound_message(
            event=payload.get("event") or {},
            event_id=event_id,
            event_type=event_type,
            tenant_key=str(tenant_key) if tenant_key else None,
            app_id=str(app_id) if app_id else None,
        )
        if message is None:
            return FeishuWebhookResult(
                outcome=FeishuWebhookOutcome.IGNORED,
                reason="payload not parseable",
            )

        message = self._apply_mention_flag(message)

        if self._should_skip_mention(message):
            return FeishuWebhookResult(
                outcome=FeishuWebhookOutcome.IGNORED,
                reason="bot not mentioned in group chat",
            )
        if self._keywords and not self._has_trigger_keyword(message.text):
            return FeishuWebhookResult(
                outcome=FeishuWebhookOutcome.IGNORED,
                reason="no trigger keyword",
            )

        return FeishuWebhookResult(
            outcome=FeishuWebhookOutcome.ACCEPTED,
            message=message,
        )


    # ── 内部 ───────────────────────────────────────

    def _decrypt_envelope(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._encrypt_key:
            raise FeishuWebhookError("encrypted payload received but encrypt_key is empty")
        cipher_b64 = payload.get("encrypt")
        if not isinstance(cipher_b64, str):
            raise FeishuWebhookError("encrypt field must be string")
        try:
            from cryptography.hazmat.primitives.ciphers import (  # type: ignore[import-not-found]
                Cipher,
                algorithms,
                modes,
            )
        except ImportError as exc:  # pragma: no cover - optional dep
            raise FeishuWebhookError(
                "cryptography package required to decrypt feishu webhook events"
            ) from exc

        try:
            ciphertext = base64.b64decode(cipher_b64)
        except (ValueError, base64.binascii.Error) as exc:
            raise FeishuWebhookError("encrypt field is not valid base64") from exc
        if len(ciphertext) < 32:
            raise FeishuWebhookError("encrypt payload too short")
        key = hashlib.sha256(self._encrypt_key.encode("utf-8")).digest()
        iv = ciphertext[:16]
        body = ciphertext[16:]
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        plaintext_padded = decryptor.update(body) + decryptor.finalize()
        pad = plaintext_padded[-1]
        if pad < 1 or pad > 16:
            raise FeishuWebhookError("invalid pkcs7 padding")
        plaintext = plaintext_padded[:-pad]
        try:
            decoded = json.loads(plaintext.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FeishuWebhookError("decrypted payload is not valid json") from exc
        if not isinstance(decoded, dict):
            raise FeishuWebhookError("decrypted payload must be json object")
        return decoded

    def _verify_token_v1(self, payload: dict[str, Any]) -> None:
        if not self._verification_token:
            return
        token = payload.get("token")
        if token != self._verification_token:
            raise FeishuWebhookError("verification_token mismatch")

    def _verify_token_v2(self, header: dict[str, Any]) -> None:
        if not self._verification_token:
            return
        token = header.get("token")
        if token != self._verification_token:
            raise FeishuWebhookError("verification_token mismatch")

    def _apply_mention_flag(self, message: FeishuInboundMessage) -> FeishuInboundMessage:
        if not self._bot_open_id:
            return message
        bot_mentioned = self._bot_open_id in message.mentions
        if bot_mentioned == message.bot_mentioned:
            return message
        return message.model_copy(update={"bot_mentioned": bot_mentioned})

    def _should_skip_mention(self, message: FeishuInboundMessage) -> bool:
        if not self._require_mention_in_group:
            return False
        if message.chat_type != FeishuChatType.GROUP:
            return False
        if not self._bot_open_id:
            # 没配 bot_open_id 时无法判定，宽容放过。
            return False
        return not message.bot_mentioned

    def _has_trigger_keyword(self, text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        return any(kw.lower() in lowered for kw in self._keywords)


# ── 规范化 ─────────────────────────────────────────


def _normalize_card_callback(
    *,
    payload: dict[str, Any],
    event: dict[str, Any],
    header: dict[str, Any],
    event_id: str,
    tenant_key: str | None,
    app_id: str | None,
) -> FeishuCardCallback | None:
    """解析飞书卡片 ``card.action.trigger`` 事件 payload。

    M5 骨架：仅提取审批所需的核心字段，避免依赖飞书 schema 细节。
    要求 ``action.value`` 是 dict 且包含 ``approval_id`` / ``decision``。
    """
    action = event.get("action") or payload.get("action") or {}
    raw_value = action.get("value") if isinstance(action, dict) else None
    if not isinstance(raw_value, dict):
        return None
    if not raw_value.get("approval_id") or not raw_value.get("decision"):
        return None
    try:
        action_value = FeishuCardActionValue.model_validate(raw_value)
    except Exception:  # noqa: BLE001 - pydantic 校验失败统一忽略
        return None

    operator = event.get("operator") or payload.get("operator") or {}
    operator_id = ""
    if isinstance(operator, dict):
        operator_id = str(
            operator.get("open_id")
            or operator.get("union_id")
            or operator.get("user_id")
            or "",
        )

    context = event.get("context") or {}
    chat_id = str(
        (event.get("chat_id") if isinstance(event, dict) else None)
        or (context.get("open_chat_id") if isinstance(context, dict) else "")
        or "",
    )
    message_id = str(
        (event.get("message_id") if isinstance(event, dict) else None)
        or (context.get("open_message_id") if isinstance(context, dict) else "")
        or "",
    )

    return FeishuCardCallback(
        event_id=event_id,
        tenant_key=tenant_key,
        app_id=app_id,
        operator_open_id=operator_id,
        chat_id=chat_id,
        message_id=message_id,
        action_value=action_value,
    )


def _normalize_inbound_message(
    *,
    event: dict[str, Any],
    event_id: str,
    event_type: str,
    tenant_key: str | None,
    app_id: str | None,
) -> FeishuInboundMessage | None:
    message_obj = event.get("message") or {}
    sender_obj = event.get("sender") or {}
    if not message_obj or not sender_obj:
        return None

    chat_id = str(message_obj.get("chat_id", ""))
    message_id = str(message_obj.get("message_id", ""))
    if not chat_id or not message_id:
        return None

    chat_type_raw = str(message_obj.get("chat_type", "")).lower()
    if chat_type_raw == "p2p":
        chat_type = FeishuChatType.P2P
    elif chat_type_raw == "group":
        chat_type = FeishuChatType.GROUP
    else:
        chat_type = FeishuChatType.UNKNOWN

    message_type_raw = str(message_obj.get("message_type", "")).lower()
    try:
        message_type = FeishuMessageType(message_type_raw)
    except ValueError:
        message_type = FeishuMessageType.UNKNOWN

    sender_id_obj = sender_obj.get("sender_id") or {}
    sender_id = (
        sender_id_obj.get("open_id")
        or sender_id_obj.get("union_id")
        or sender_id_obj.get("user_id")
        or ""
    )
    sender_id_type = "open_id"
    if not sender_id_obj.get("open_id"):
        if sender_id_obj.get("union_id"):
            sender_id_type = "union_id"
        elif sender_id_obj.get("user_id"):
            sender_id_type = "user_id"

    mentions: list[str] = []
    mention_keys: list[str] = []
    raw_mentions = message_obj.get("mentions") or ()
    for mention in raw_mentions:
        if not isinstance(mention, dict):
            continue
        mention_key = mention.get("key")
        if mention_key:
            mention_keys.append(str(mention_key))
        mention_id = (mention.get("id") or {}).get("open_id")
        if mention_id:
            mentions.append(str(mention_id))

    text = _strip_mention_keys(
        _extract_text(message_type, message_obj.get("content")),
        mention_keys,
    )

    attachments = _extract_attachments(message_type, message_obj.get("content"))

    raw_digest = hashlib.sha256(
        json.dumps(message_obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    return FeishuInboundMessage(
        event_id=event_id,
        event_type=event_type,
        tenant_key=tenant_key,
        app_id=app_id,
        chat_id=chat_id,
        chat_type=chat_type,
        message_id=message_id,
        message_type=message_type,
        sender_id=str(sender_id),
        sender_id_type=sender_id_type,
        text=text,
        mentions=mentions,
        attachments=attachments,
        raw_digest=raw_digest,
    )


def _extract_text(message_type: FeishuMessageType, raw_content: Any) -> str:  # noqa: ANN401
    if not raw_content:
        return ""
    try:
        parsed = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
    except json.JSONDecodeError:
        return str(raw_content)
    if not isinstance(parsed, dict):
        return str(parsed)

    if message_type == FeishuMessageType.TEXT:
        return str(parsed.get("text", ""))
    if message_type == FeishuMessageType.POST:
        title = parsed.get("title") or ""
        # post 内容是一个 list[list[segment]]；只抽 text segments。
        segments: list[str] = []
        for line in parsed.get("content") or ():
            if not isinstance(line, list):
                continue
            for seg in line:
                if isinstance(seg, dict) and seg.get("tag") == "text":
                    segments.append(str(seg.get("text", "")))
        body = "\n".join(s for s in segments if s)
        return f"{title}\n{body}".strip() if title else body
    if message_type == FeishuMessageType.FILE:
        return str(parsed.get("file_name", ""))
    if message_type == FeishuMessageType.IMAGE:
        return ""
    return str(parsed)


def _strip_mention_keys(text: str, mention_keys: list[str]) -> str:
    """移除飞书文本里的 mention 占位符，避免把 @bot 当成用户意图。"""
    cleaned = text
    for key in mention_keys:
        cleaned = cleaned.replace(key, " ")
    cleaned = re.sub(r"@_user_\d+\b", " ", cleaned)
    return " ".join(cleaned.split())


def _extract_attachments(
    message_type: FeishuMessageType,
    raw_content: Any,  # noqa: ANN401
) -> list[FeishuAttachment]:
    if not raw_content or message_type not in {
        FeishuMessageType.FILE,
        FeishuMessageType.IMAGE,
        FeishuMessageType.AUDIO,
        FeishuMessageType.MEDIA,
    }:
        return []
    try:
        parsed = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, dict):
        return []
    file_key = parsed.get("file_key") or parsed.get("image_key")
    if not file_key:
        return []
    return [
        FeishuAttachment(
            kind=message_type,
            file_key=str(file_key),
            file_name=parsed.get("file_name"),
            mime_type=parsed.get("mime_type"),
            size=parsed.get("file_size"),
        )
    ]


__all__ = [
    "FeishuWebhookError",
    "FeishuWebhookProcessor",
    "FeishuWebhookResult",
]
