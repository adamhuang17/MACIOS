"""M4 飞书技能：fetch_recent / create_doc / upload_file / send_message。

设计与 :mod:`agent_hub.pilot.skills.internal` 同源：

- skill 不读写实体快照、不发事件，纯能力执行；
- 通过闭包注入 :class:`FeishuClientProtocol` 与 ``ArtifactContentReader``；
- ``fetch_recent`` 只读，用于把飞书会话上下文落成 ``context_bundle``；
- ``create_doc`` / ``upload_file`` / ``send_message`` 是写 / 分享操作，
  默认 ``requires_approval=True``，由 ApprovalService 控制。

输入 ``input_artifacts`` 由 :class:`StepInputResolver` 注入，包括：

- ``brief`` —— Project Brief Markdown，用于 upload_file 写盘；
- ``ctx``   —— ContextBundle，用于 send_summary 时拼接概要。
"""

from __future__ import annotations

import json
from typing import Any

from agent_hub.connectors.feishu.client import FeishuApiError, FeishuClientProtocol
from agent_hub.pilot.domain.enums import ArtifactType
from agent_hub.pilot.skills.internal import ArtifactContentReader
from agent_hub.pilot.skills.registry import (
    SkillInvocation,
    SkillRegistry,
    SkillResult,
    SkillSpec,
)

FEISHU_SCOPES = ["feishu"]


# ── feishu.im.fetch_recent ─────────────────────────


def _make_fetch_recent_skill(client: FeishuClientProtocol) -> tuple[SkillSpec, Any]:
    async def _fetch(inv: SkillInvocation) -> SkillResult:
        params = inv.params
        chat_id = str(params.get("chat_id", "") or "")
        page_size = int(params.get("page_size", 20) or 20)
        if not chat_id:
            return SkillResult(
                skill_name=inv.skill_name,
                success=False,
                error="missing required param: chat_id",
            )

        if inv.dry_run:
            return SkillResult(
                skill_name=inv.skill_name,
                success=True,
                output={"chat_id": chat_id, "page_size": page_size, "would_fetch": True},
            )

        try:
            records = await client.fetch_recent_messages(
                chat_id=chat_id,
                page_size=page_size,
            )
        except FeishuApiError as exc:
            return SkillResult(skill_name=inv.skill_name, success=False, error=str(exc))

        messages = [
            {
                "message_id": r.message_id,
                "sender_id": r.sender_id,
                "message_type": r.message_type,
                "text": r.text,
                "create_time": r.create_time.isoformat(),
            }
            for r in records
        ]
        bundle = {
            "origin_text": str(params.get("origin_text", "") or ""),
            "messages": messages,
            "attachments": list(params.get("attachments", []) or []),
            "source_refs": [f"feishu:msg:{r.message_id}" for r in records],
            "context_scope": {"chat_id": chat_id, "page_size": page_size},
        }
        return SkillResult(
            skill_name=inv.skill_name,
            success=True,
            output={"chat_id": chat_id, "message_count": len(messages)},
            artifact_payload={
                "type": ArtifactType.CONTEXT_BUNDLE.value,
                "title": params.get("title", f"Feishu chat {chat_id} 上下文"),
                "mime_type": "application/json",
                "content": bundle,
                "metadata": {
                    "feishu_chat_id": chat_id,
                    "fetched_count": len(messages),
                },
            },
        )

    spec = SkillSpec(
        skill_name="feishu.im.fetch_recent",
        description="拉取飞书会话最近若干条消息，规范化为 ContextBundle。",
        side_effect="read",
        requires_approval=False,
        supports_dry_run=True,
        scopes=FEISHU_SCOPES,
        timeout_ms=15_000,
    )
    return spec, _fetch


# ── feishu.docs.create_doc ─────────────────────────


def _make_create_doc_skill(
    client: FeishuClientProtocol,
    *,
    default_folder_token: str = "",
) -> tuple[SkillSpec, Any]:
    async def _create(inv: SkillInvocation) -> SkillResult:
        params = inv.params
        title = str(params.get("title", "") or "Untitled Doc")
        folder_token = str(params.get("folder_token", "") or default_folder_token) or None

        if inv.dry_run:
            return SkillResult(
                skill_name=inv.skill_name,
                success=True,
                output={
                    "title": title,
                    "folder_token": folder_token,
                    "would_create": True,
                },
            )

        try:
            created = await client.create_doc(title=title, folder_token=folder_token)
        except FeishuApiError as exc:
            return SkillResult(skill_name=inv.skill_name, success=False, error=str(exc))

        return SkillResult(
            skill_name=inv.skill_name,
            success=True,
            output={
                "document_id": created.document_id,
                "url": created.url,
                "revision_id": created.revision_id,
            },
            artifact_payload={
                "type": ArtifactType.FEISHU_DOC.value,
                "title": title,
                "mime_type": "application/vnd.feishu.docx",
                "content": {
                    "document_id": created.document_id,
                    "url": created.url,
                    "title": title,
                },
                "metadata": {
                    "document_id": created.document_id,
                    "url": created.url,
                    "revision_id": created.revision_id,
                    "folder_token": folder_token or "",
                },
            },
        )

    spec = SkillSpec(
        skill_name="feishu.docs.create_doc",
        description="在飞书云文档创建新 docx，返回 document_id 与 URL。",
        side_effect="write",
        requires_approval=True,
        supports_dry_run=True,
        scopes=FEISHU_SCOPES,
        idempotency_key_fields=["title", "folder_token"],
        timeout_ms=20_000,
    )
    return spec, _create


# ── feishu.drive.upload_file ───────────────────────


def _make_upload_file_skill(
    client: FeishuClientProtocol,
    *,
    artifact_reader: ArtifactContentReader,
    default_folder_token: str = "",
) -> tuple[SkillSpec, Any]:
    async def _upload(inv: SkillInvocation) -> SkillResult:
        params = inv.params
        file_name = str(params.get("file_name", "") or "")
        if not file_name:
            return SkillResult(
                skill_name=inv.skill_name,
                success=False,
                error="missing required param: file_name",
            )
        folder_token = str(params.get("folder_token", "") or default_folder_token) or None

        upstream = params.get("input_artifacts", {}) or {}
        source_artifact = (
            upstream.get("file") or upstream.get("brief") or upstream.get("source")
        )
        explicit_artifact_id = params.get("artifact_id")

        content_bytes: bytes | None = None
        if isinstance(source_artifact, dict):
            content = source_artifact.get("content")
            content_bytes = _to_bytes(content)
        elif explicit_artifact_id:
            raw = await artifact_reader(str(explicit_artifact_id))
            content_bytes = _to_bytes(raw)
        elif "content_bytes" in params:
            value = params["content_bytes"]
            content_bytes = value if isinstance(value, bytes) else _to_bytes(value)

        if content_bytes is None:
            return SkillResult(
                skill_name=inv.skill_name,
                success=False,
                error="upload_file 需要 input_artifacts 中提供文件内容",
            )

        if inv.dry_run:
            return SkillResult(
                skill_name=inv.skill_name,
                success=True,
                output={
                    "file_name": file_name,
                    "folder_token": folder_token,
                    "byte_size": len(content_bytes),
                    "would_upload": True,
                },
            )

        try:
            uploaded = await client.upload_file(
                file_name=file_name,
                content=content_bytes,
                parent_type="explorer",
                parent_node=folder_token,
            )
        except FeishuApiError as exc:
            return SkillResult(skill_name=inv.skill_name, success=False, error=str(exc))

        return SkillResult(
            skill_name=inv.skill_name,
            success=True,
            output={
                "file_token": uploaded.file_token,
                "byte_size": uploaded.size,
                "download_url": uploaded.download_url,
            },
            artifact_payload={
                "type": ArtifactType.DRIVE_FILE.value,
                "title": file_name,
                "mime_type": "application/octet-stream",
                "content": {
                    "file_token": uploaded.file_token,
                    "file_name": file_name,
                    "download_url": uploaded.download_url,
                },
                "metadata": {
                    "file_token": uploaded.file_token,
                    "byte_size": uploaded.size,
                    "folder_token": folder_token or "",
                },
            },
        )

    spec = SkillSpec(
        skill_name="feishu.drive.upload_file",
        description="把 artifact 内容上传到飞书云空间。",
        side_effect="write",
        requires_approval=True,
        supports_dry_run=True,
        scopes=FEISHU_SCOPES,
        idempotency_key_fields=["file_name", "folder_token"],
        timeout_ms=30_000,
    )
    return spec, _upload


# ── feishu.im.send_message ─────────────────────────


def _make_send_message_skill(client: FeishuClientProtocol) -> tuple[SkillSpec, Any]:
    async def _send(inv: SkillInvocation) -> SkillResult:
        params = inv.params
        chat_id = str(params.get("chat_id", "") or "")
        text = _build_message_text(params)
        msg_type = str(params.get("msg_type", "text") or "text")
        receive_id_type = str(params.get("receive_id_type", "chat_id") or "chat_id")
        if not chat_id:
            return SkillResult(
                skill_name=inv.skill_name,
                success=False,
                error="missing required param: chat_id",
            )
        if not text:
            return SkillResult(
                skill_name=inv.skill_name,
                success=False,
                error="missing required param: text",
            )

        if msg_type == "text":
            content = json.dumps({"text": text}, ensure_ascii=False)
        else:
            # 上层已自行 encode；保持透传，方便未来支持 post / interactive。
            content = str(params.get("content") or text)

        if inv.dry_run:
            return SkillResult(
                skill_name=inv.skill_name,
                success=True,
                output={
                    "chat_id": chat_id,
                    "msg_type": msg_type,
                    "char_count": len(text),
                    "would_send": True,
                },
            )

        try:
            sent = await client.send_message(
                receive_id=chat_id,
                receive_id_type=receive_id_type,
                msg_type=msg_type,
                content=content,
            )
        except FeishuApiError as exc:
            return SkillResult(skill_name=inv.skill_name, success=False, error=str(exc))

        return SkillResult(
            skill_name=inv.skill_name,
            success=True,
            output={
                "message_id": sent.message_id,
                "chat_id": sent.chat_id,
                "char_count": len(text),
            },
            artifact_payload={
                "type": ArtifactType.SUMMARY.value,
                "title": params.get("title", "Feishu 消息推送"),
                "mime_type": "text/plain",
                "content": text,
                "metadata": {
                    "message_id": sent.message_id,
                    "chat_id": sent.chat_id,
                    "msg_type": msg_type,
                },
            },
        )

    spec = SkillSpec(
        skill_name="feishu.im.send_message",
        description="向指定飞书会话发送一条消息（默认文本）。",
        side_effect="share",
        requires_approval=True,
        supports_dry_run=True,
        scopes=FEISHU_SCOPES,
        idempotency_key_fields=["chat_id", "title"],
        timeout_ms=15_000,
    )
    return spec, _send


# ── 工具 ───────────────────────────────────────────


def _to_bytes(value: Any) -> bytes | None:  # noqa: ANN401
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, dict | list):
        try:
            return json.dumps(value, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError):
            return None
    return None


def _build_message_text(params: dict[str, Any]) -> str:
    text = str(params.get("text", "") or "").strip()
    if text:
        return text

    title = str(params.get("title", "") or "").strip()
    upstream = params.get("input_artifacts", {}) or {}
    share_artifact = upstream.get("share") or upstream.get("summary")
    brief_artifact = upstream.get("brief")

    share_title = ""
    share_url = ""
    file_token = ""
    if isinstance(share_artifact, dict):
        share_title = str(share_artifact.get("title", "") or "").strip()
        content = share_artifact.get("content")
        metadata = share_artifact.get("metadata") if isinstance(
            share_artifact.get("metadata"), dict,
        ) else {}
        if isinstance(content, dict):
            share_title = str(content.get("file_name") or share_title or "").strip()
            share_url = str(
                content.get("share_url") or content.get("download_url") or "",
            ).strip()
            file_token = str(content.get("file_token") or "").strip()
        if metadata:
            share_url = str(metadata.get("share_url") or share_url or "").strip()
            file_token = str(metadata.get("file_token") or file_token or "").strip()

    brief_title = ""
    if isinstance(brief_artifact, dict):
        brief_title = str(brief_artifact.get("title", "") or "").strip()

    lines: list[str] = []
    if title:
        lines.append(title)
    elif share_title:
        lines.append(f"已生成文件：{share_title}")
    elif brief_title:
        lines.append(f"已完成：{brief_title}")
    if share_url:
        lines.append(f"分享链接：{share_url}")
        lines.append("（飞书已单独推送一条「与我共享」通知，可直接点击打开）")
    elif file_token:
        lines.append(f"文件 token：{file_token}")
        lines.append("（飞书已单独推送一条「与我共享」通知，可直接点击打开）")

    return "\n".join(lines).strip()


# ── 注册入口 ────────────────────────────────────────


def register_feishu_skills(
    registry: SkillRegistry,
    client: FeishuClientProtocol,
    *,
    artifact_reader: ArtifactContentReader | None = None,
    default_folder_token: str = "",
    allow_overwrite: bool = False,
) -> None:
    """把 M4 飞书技能集合注册到 ``registry``。

    Args:
        registry: 目标 :class:`SkillRegistry`。
        client: 真实或 Fake 的 :class:`FeishuClientProtocol` 实现。
        artifact_reader: 上传时按 ``artifact_id`` 读取内容；缺省时
            必须由 ``input_artifacts`` 直接传 content。
        default_folder_token: drive / docs 的默认目录。
    """
    spec, fn = _make_fetch_recent_skill(client)
    registry.register(spec, fn, allow_overwrite=allow_overwrite)

    spec, fn = _make_create_doc_skill(client, default_folder_token=default_folder_token)
    registry.register(spec, fn, allow_overwrite=allow_overwrite)

    if artifact_reader is not None:
        spec, fn = _make_upload_file_skill(
            client,
            artifact_reader=artifact_reader,
            default_folder_token=default_folder_token,
        )
        registry.register(spec, fn, allow_overwrite=allow_overwrite)

    spec, fn = _make_send_message_skill(client)
    registry.register(spec, fn, allow_overwrite=allow_overwrite)


__all__ = [
    "FEISHU_SCOPES",
    "register_feishu_skills",
]
