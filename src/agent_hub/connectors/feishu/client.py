"""飞书开放平台 HTTP 客户端。

只覆盖 M4 需要的能力，全部返回内部 DTO，不外泄 raw payload：

- ``send_message``        —— ``POST /open-apis/im/v1/messages``
- ``fetch_recent_messages`` —— ``GET  /open-apis/im/v1/messages``
- ``upload_file``          —— ``POST /open-apis/drive/v1/files/upload_all``
- ``share_file``           —— ``POST /open-apis/drive/v1/permissions/{token}/members``
- ``create_doc``           —— ``POST /open-apis/docx/v1/documents``

为了让 skills / 测试不依赖真实 HTTP，本模块同时定义
:class:`FeishuClientProtocol` 与一个内存 :class:`FakeFeishuClient`。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

import httpx
import structlog

from agent_hub.connectors.feishu.auth import FeishuAuthProvider

logger = structlog.get_logger(__name__)


class FeishuApiError(RuntimeError):
    """飞书开放平台业务错误。

    ``code`` 为开放平台返回的业务码；HTTP 层错误统一包装为
    ``code = -1``。
    """

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"feishu api error: code={code} msg={message}")
        self.code = code
        self.message = message


# ── 返回 DTO ────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FeishuMessageSent:
    message_id: str
    chat_id: str
    create_time: datetime


@dataclass(frozen=True, slots=True)
class FeishuMessageRecord:
    message_id: str
    chat_id: str
    sender_id: str
    message_type: str
    text: str
    create_time: datetime


@dataclass(frozen=True, slots=True)
class FeishuFileUploaded:
    file_token: str
    file_name: str
    size: int
    folder_token: str | None
    download_url: str | None = None


@dataclass(frozen=True, slots=True)
class FeishuDocCreated:
    document_id: str
    title: str
    revision_id: int
    url: str


@dataclass(frozen=True, slots=True)
class FeishuCardSent:
    """交互卡片发送结果。

    M5 审批卡片骨架：``card_id`` 为发送后由飞书返回的
    消息 ID，后续 `update_card` 、 `card.action.trigger` 回调均
    以该 ID 关联到本地 :class:`Approval`。
    """

    message_id: str
    chat_id: str
    create_time: datetime


# ── Protocol & Fake ────────────────────────────────


@runtime_checkable
class FeishuClientProtocol(Protocol):
    async def send_message(
        self,
        *,
        receive_id: str,
        receive_id_type: str,
        msg_type: str,
        content: str,
    ) -> FeishuMessageSent: ...

    async def fetch_recent_messages(
        self,
        *,
        chat_id: str,
        page_size: int = 20,
    ) -> list[FeishuMessageRecord]: ...

    async def upload_file(
        self,
        *,
        file_name: str,
        content: bytes,
        parent_type: str = "explorer",
        parent_node: str | None = None,
    ) -> FeishuFileUploaded: ...

    async def create_doc(
        self,
        *,
        title: str,
        folder_token: str | None = None,
    ) -> FeishuDocCreated: ...

    async def send_interactive_card(
        self,
        *,
        receive_id: str,
        receive_id_type: str,
        card: dict[str, Any],
    ) -> FeishuCardSent: ...

    async def update_card(
        self,
        *,
        message_id: str,
        card: dict[str, Any],
    ) -> None: ...

    async def share_file(
        self,
        *,
        file_token: str,
        member_open_id: str,
        perm: str = "edit",
        need_notification: bool = True,
    ) -> None: ...

    async def aclose(self) -> None: ...


@dataclass(slots=True)
class _FakeMessageEntry:
    record: FeishuMessageRecord


@dataclass(slots=True)
class FakeFeishuClient:
    """单元测试用内存实现；保留所有出站调用便于断言。

    自动在 :meth:`send_message` 时给消息分配 ``om_*`` ID、把
    ``fetch_recent_messages`` 的历史消息写到本地。
    """

    base_url: str = "https://fake.feishu.local"
    sent_messages: list[dict[str, Any]] = field(default_factory=list)
    uploaded_files: list[dict[str, Any]] = field(default_factory=list)
    created_docs: list[dict[str, Any]] = field(default_factory=list)
    sent_cards: list[dict[str, Any]] = field(default_factory=list)
    updated_cards: list[dict[str, Any]] = field(default_factory=list)
    history: dict[str, list[FeishuMessageRecord]] = field(default_factory=dict)
    _seq: int = 0

    def _next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}_{self._seq:08d}"

    async def send_message(
        self,
        *,
        receive_id: str,
        receive_id_type: str,
        msg_type: str,
        content: str,
    ) -> FeishuMessageSent:
        sent = FeishuMessageSent(
            message_id=self._next_id("om"),
            chat_id=receive_id,
            create_time=datetime.now(UTC),
        )
        self.sent_messages.append(
            {
                "receive_id": receive_id,
                "receive_id_type": receive_id_type,
                "msg_type": msg_type,
                "content": content,
                "message_id": sent.message_id,
            }
        )
        return sent

    async def fetch_recent_messages(
        self,
        *,
        chat_id: str,
        page_size: int = 20,
    ) -> list[FeishuMessageRecord]:
        records = list(self.history.get(chat_id, ()))
        # 模拟开放平台从新到旧的返回顺序。
        records.sort(key=lambda r: r.create_time, reverse=True)
        return records[:page_size]

    async def upload_file(
        self,
        *,
        file_name: str,
        content: bytes,
        parent_type: str = "explorer",
        parent_node: str | None = None,
    ) -> FeishuFileUploaded:
        token = self._next_id("file")
        uploaded = FeishuFileUploaded(
            file_token=token,
            file_name=file_name,
            size=len(content),
            folder_token=parent_node,
            download_url=f"{self.base_url}/file/{token}",
        )
        self.uploaded_files.append(
            {
                "file_name": file_name,
                "size": len(content),
                "parent_type": parent_type,
                "parent_node": parent_node,
                "file_token": token,
            }
        )
        return uploaded

    async def create_doc(
        self,
        *,
        title: str,
        folder_token: str | None = None,
    ) -> FeishuDocCreated:
        doc_id = self._next_id("doc")
        created = FeishuDocCreated(
            document_id=doc_id,
            title=title,
            revision_id=1,
            url=f"{self.base_url}/docx/{doc_id}",
        )
        self.created_docs.append(
            {
                "document_id": doc_id,
                "title": title,
                "folder_token": folder_token,
            }
        )
        return created

    shared_files: list[dict[str, Any]] = field(default_factory=list)

    async def share_file(
        self,
        *,
        file_token: str,
        member_open_id: str,
        perm: str = "edit",
        need_notification: bool = True,
    ) -> None:
        self.shared_files.append(
            {
                "file_token": file_token,
                "member_open_id": member_open_id,
                "perm": perm,
                "need_notification": need_notification,
            }
        )

    async def aclose(self) -> None:  # pragma: no cover - no-op
        return None

    async def send_interactive_card(
        self,
        *,
        receive_id: str,
        receive_id_type: str,
        card: dict[str, Any],
    ) -> FeishuCardSent:
        sent = FeishuCardSent(
            message_id=self._next_id("om_card"),
            chat_id=receive_id,
            create_time=datetime.now(UTC),
        )
        self.sent_cards.append(
            {
                "receive_id": receive_id,
                "receive_id_type": receive_id_type,
                "card": card,
                "message_id": sent.message_id,
            }
        )
        return sent

    async def update_card(
        self,
        *,
        message_id: str,
        card: dict[str, Any],
    ) -> None:
        self.updated_cards.append({"message_id": message_id, "card": card})


# ── 真实 HTTP 实现 ──────────────────────────────────


class FeishuClient:
    """基于 httpx 的飞书开放平台客户端。"""

    def __init__(
        self,
        *,
        auth: FeishuAuthProvider,
        api_base_url: str = "https://open.feishu.cn",
        request_timeout_ms: int = 12_000,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._auth = auth
        self._api_base_url = api_base_url.rstrip("/")
        self._timeout_s = request_timeout_ms / 1000.0
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=self._timeout_s)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _headers(self) -> dict[str, str]:
        token = await self._auth.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self._api_base_url}{path}"
        try:
            resp = await self._client.request(
                method,
                url,
                params=params,
                json=json_body,
                headers=await self._headers(),
            )
        except httpx.HTTPError as exc:
            raise FeishuApiError(-1, f"http error: {exc.__class__.__name__}") from exc

        if resp.status_code == 401:
            await self._auth.invalidate()
            raise FeishuApiError(401, "unauthorized")
        if resp.status_code >= 400:
            raise FeishuApiError(resp.status_code, f"http {resp.status_code}")

        try:
            payload = resp.json()
        except ValueError as exc:
            raise FeishuApiError(-1, "invalid json") from exc
        code = payload.get("code", 0)
        if code != 0:
            raise FeishuApiError(int(code), str(payload.get("msg", "")))
        return payload

    async def send_message(
        self,
        *,
        receive_id: str,
        receive_id_type: str,
        msg_type: str,
        content: str,
    ) -> FeishuMessageSent:
        body = {"receive_id": receive_id, "msg_type": msg_type, "content": content}
        payload = await self._request_json(
            "POST",
            "/open-apis/im/v1/messages",
            params={"receive_id_type": receive_id_type},
            json_body=body,
        )
        data = payload.get("data") or {}
        message_id = data.get("message_id") or ""
        chat_id = data.get("chat_id") or receive_id
        create_time = _parse_open_platform_timestamp(data.get("create_time"))
        return FeishuMessageSent(message_id=message_id, chat_id=chat_id, create_time=create_time)

    async def fetch_recent_messages(
        self,
        *,
        chat_id: str,
        page_size: int = 20,
    ) -> list[FeishuMessageRecord]:
        payload = await self._request_json(
            "GET",
            "/open-apis/im/v1/messages",
            params={
                "container_id_type": "chat",
                "container_id": chat_id,
                "page_size": min(max(page_size, 1), 50),
                "sort_type": "ByCreateTimeDesc",
            },
        )
        items = (payload.get("data") or {}).get("items") or []
        result: list[FeishuMessageRecord] = []
        for raw in items:
            message_id = raw.get("message_id") or ""
            sender = ((raw.get("sender") or {}).get("id")) or ""
            msg_type = raw.get("msg_type") or ""
            create_time = _parse_open_platform_timestamp(raw.get("create_time"))
            text = _flatten_open_platform_message(raw)
            result.append(
                FeishuMessageRecord(
                    message_id=message_id,
                    chat_id=chat_id,
                    sender_id=sender,
                    message_type=msg_type,
                    text=text,
                    create_time=create_time,
                )
            )
        return result

    async def upload_file(
        self,
        *,
        file_name: str,
        content: bytes,
        parent_type: str = "explorer",
        parent_node: str | None = None,
    ) -> FeishuFileUploaded:
        # Drive upload_all 是 multipart 接口，与其他 JSON 接口签名不同。
        url = f"{self._api_base_url}/open-apis/drive/v1/files/upload_all"
        token = await self._auth.get_token()
        files = {
            "file_name": (None, file_name),
            "parent_type": (None, parent_type),
            "parent_node": (None, parent_node or ""),
            "size": (None, str(len(content))),
            "file": (file_name, content, "application/octet-stream"),
        }
        try:
            resp = await self._client.post(
                url,
                headers={"Authorization": f"Bearer {token}"},
                files=files,
            )
        except httpx.HTTPError as exc:
            raise FeishuApiError(-1, f"http error: {exc.__class__.__name__}") from exc
        if resp.status_code >= 400:
            raise FeishuApiError(resp.status_code, f"http {resp.status_code}")
        payload = resp.json()
        if payload.get("code", 0) != 0:
            raise FeishuApiError(int(payload.get("code", 0)), str(payload.get("msg", "")))
        data = payload.get("data") or {}
        file_token = data.get("file_token") or ""
        return FeishuFileUploaded(
            file_token=file_token,
            file_name=file_name,
            size=len(content),
            folder_token=parent_node,
            download_url=data.get("download_url"),
        )

    async def share_file(
        self,
        *,
        file_token: str,
        member_open_id: str,
        perm: str = "edit",
        need_notification: bool = True,
    ) -> None:
        """为已上传的云空间文件添加协作者权限。

        ``need_notification=True`` 时飞书会主动给 ``member_open_id`` 发一条
        包含可点击链接的“与我共享”通知，这是机器人让用户拿到文件链接
        的官方推荐路径。
        """
        await self._request_json(
            "POST",
            f"/open-apis/drive/v1/permissions/{file_token}/members",
            params={
                "type": "file",
                "need_notification": "true" if need_notification else "false",
            },
            json_body={
                "member_type": "openid",
                "member_id": member_open_id,
                "perm": perm,
            },
        )

    async def create_doc(
        self,
        *,
        title: str,
        folder_token: str | None = None,
    ) -> FeishuDocCreated:
        body: dict[str, Any] = {"title": title}
        if folder_token:
            body["folder_token"] = folder_token
        payload = await self._request_json(
            "POST",
            "/open-apis/docx/v1/documents",
            json_body=body,
        )
        data = (payload.get("data") or {}).get("document") or {}
        document_id = data.get("document_id") or ""
        revision_id = int(data.get("revision_id", 1))
        url = f"https://feishu.cn/docx/{document_id}" if document_id else ""
        return FeishuDocCreated(
            document_id=document_id,
            title=title,
            revision_id=revision_id,
            url=url,
        )

    async def send_interactive_card(
        self,
        *,
        receive_id: str,
        receive_id_type: str,
        card: dict[str, Any],
    ) -> FeishuCardSent:
        """发送 interactive 卡片（``msg_type=interactive``）。

        骨架实现：复用 ``/open-apis/im/v1/messages``，以 ``content``
        携带卡片 JSON。入参 ``card`` 应是飞书卡片协议的 dict，本
        层负责序列化。
        """
        body = {
            "receive_id": receive_id,
            "msg_type": "interactive",
            "content": json.dumps(card, ensure_ascii=False),
        }
        payload = await self._request_json(
            "POST",
            "/open-apis/im/v1/messages",
            params={"receive_id_type": receive_id_type},
            json_body=body,
        )
        data = payload.get("data") or {}
        message_id = data.get("message_id") or ""
        chat_id = data.get("chat_id") or receive_id
        create_time = _parse_open_platform_timestamp(data.get("create_time"))
        return FeishuCardSent(
            message_id=message_id, chat_id=chat_id, create_time=create_time,
        )

    async def update_card(
        self,
        *,
        message_id: str,
        card: dict[str, Any],
    ) -> None:
        """更新已发送的交互卡片。

        骨架实现：调用 ``PATCH /open-apis/im/v1/messages/{message_id}``。
        使用方负责传入完整的 ``card`` JSON。
        """
        body = {
            "content": json.dumps(card, ensure_ascii=False),
        }
        await self._request_json(
            "PATCH",
            f"/open-apis/im/v1/messages/{message_id}",
            json_body=body,
        )


# ── 解析工具 ────────────────────────────────────────


def _parse_open_platform_timestamp(value: Any) -> datetime:  # noqa: ANN401
    if value is None:
        return datetime.now(UTC)
    try:
        ts = int(value)
    except (TypeError, ValueError):
        return datetime.now(UTC)
    # 飞书返回的 create_time 多为毫秒；按长度兜底。
    if ts > 10_000_000_000:
        ts //= 1000
    try:
        return datetime.fromtimestamp(ts, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return datetime.now(UTC)


def _flatten_open_platform_message(raw: dict[str, Any]) -> str:
    body = raw.get("body") or {}
    content = body.get("content")
    if not content:
        return ""
    try:
        parsed = json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError:
        return str(content)
    if isinstance(parsed, dict):
        text = parsed.get("text")
        if isinstance(text, str):
            return text
    return str(parsed)


__all__ = [
    "FakeFeishuClient",
    "FeishuApiError",
    "FeishuCardSent",
    "FeishuClient",
    "FeishuClientProtocol",
    "FeishuDocCreated",
    "FeishuFileUploaded",
    "FeishuMessageRecord",
    "FeishuMessageSent",
]
