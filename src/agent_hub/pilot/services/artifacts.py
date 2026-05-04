"""ArtifactStore：把 SkillResult 的 artifact_payload 物化为 Artifact 快照。

M1 仅保存元数据；M3 起补齐：

- 内容落盘：``content`` 字段（``str`` / ``dict`` / ``bytes``）按 mime
  写入 ``settings.pilot_artifact_dir``，记录 ``storage_key`` /
  ``checksum`` / ``uri``；目录为空时退化为内存键值兜底，方便单测。
- 版本链：``parent_artifact_id`` 指向旧 Artifact，旧 Artifact 自动
  ``SUPERSEDED``，新 Artifact ``artifact_version = parent.version + 1``
  并继承 ``parent_artifact_id``。
- 读取接口：``read_content`` 给 ExecutionEngine 的
  :class:`StepInputResolver` 使用，让下游 step 拿到上游 Artifact 内容。

Artifact 状态流：
    DRAFT → GENERATE → GENERATED          （所有产物）
    GENERATED → UPLOAD → UPLOADED         （UPLOAD kind 步骤）
    UPLOADED → SHARE → SHARED             （SHARE kind / share skill）
    GENERATED|UPLOADED|SHARED → SUPERSEDE → SUPERSEDED
                                          （存在 parent_artifact_id 的新版本）
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from agent_hub.pilot.domain.enums import (
    ArtifactAction,
    ArtifactStatus,
    ArtifactType,
    EventType,
    PlanStepKind,
)
from agent_hub.pilot.domain.errors import IllegalTransition
from agent_hub.pilot.domain.events import ExecutionEvent
from agent_hub.pilot.domain.models import Artifact, PlanStep, Task
from agent_hub.pilot.domain.state import transition

if TYPE_CHECKING:
    from agent_hub.pilot.services.event_publisher import PilotEventPublisher
    from agent_hub.pilot.services.repository import PilotRepository

logger = structlog.get_logger(__name__)


_PROMOTE_BY_KIND: dict[PlanStepKind, list[ArtifactAction]] = {
    PlanStepKind.UPLOAD: [ArtifactAction.UPLOAD],
    PlanStepKind.SHARE: [ArtifactAction.UPLOAD, ArtifactAction.SHARE],
    PlanStepKind.SUMMARIZE: [],
}


_MEMORY_SCHEME = "memory://"


class ArtifactStore:
    def __init__(
        self,
        repository: PilotRepository,
        publisher: PilotEventPublisher,
        *,
        artifact_dir: str | Path | None = None,
    ) -> None:
        self._repo = repository
        self._publisher = publisher
        self._artifact_dir = Path(artifact_dir) if artifact_dir else None
        # 内存兜底，用于未配置 artifact_dir 时的测试场景。
        self._memory_blobs: dict[str, str | bytes] = {}

    # ── 写入 ───────────────────────────────────────

    async def save(
        self,
        step: PlanStep,
        task: Task,
        payload: dict[str, Any],
    ) -> Artifact:
        parent_id: str | None = payload.get("parent_artifact_id")
        parent: Artifact | None = None
        if parent_id:
            parent = await self._repo.get_artifact(parent_id)
            if parent is None:
                msg = f"parent_artifact_id 不存在: {parent_id}"
                raise ValueError(msg)

        artifact_version = int(
            payload.get(
                "artifact_version",
                (parent.artifact_version + 1) if parent is not None else 1,
            ),
        )

        content = payload.get("content")
        mime_type = payload.get("mime_type")
        storage_key, checksum, uri, content_bytes = (
            self._materialize_content(content, mime_type)
        )

        artifact = Artifact(
            workspace_id=step.workspace_id,
            task_id=step.task_id,
            step_id=step.step_id,
            type=ArtifactType(payload["type"]),
            title=payload.get("title", "untitled"),
            artifact_version=artifact_version,
            parent_artifact_id=parent_id,
            uri=payload.get("uri") or uri,
            storage_key=payload.get("storage_key") or storage_key,
            mime_type=mime_type,
            checksum=payload.get("checksum") or checksum,
            feishu_token=payload.get("metadata", {}).get("feishu_token"),
            share_url=payload.get("metadata", {}).get("share_url"),
            metadata=dict(payload.get("metadata", {})),
            status=ArtifactStatus.DRAFT,
        )

        # 内容已在 _materialize_content 内落盘 / 写入内存；这里只需绑定 id。
        if content_bytes is not None and (storage_key or "").startswith(
            _MEMORY_SCHEME,
        ):
            self._memory_blobs[artifact.artifact_id] = (
                content_bytes if isinstance(content_bytes, bytes)
                else content_bytes
            )
            artifact = artifact.model_copy(update={
                "storage_key": f"{_MEMORY_SCHEME}{artifact.artifact_id}",
                "uri": artifact.uri or f"{_MEMORY_SCHEME}{artifact.artifact_id}",
            })

        await self._repo.save(artifact, expected_version=None)
        await self._publisher.record(ExecutionEvent(
            workspace_id=artifact.workspace_id,
            trace_id=task.trace_id,
            task_id=task.task_id,
            plan_id=step.plan_id,
            step_id=step.step_id,
            artifact_id=artifact.artifact_id,
            type=EventType.ARTIFACT_CREATED,
            message=f"artifact created: {artifact.type.value}",
            payload={
                "artifact_type": artifact.type.value,
                "artifact_version": artifact.artifact_version,
                "parent_artifact_id": artifact.parent_artifact_id,
            },
        ))

        # DRAFT → GENERATED
        artifact = transition(artifact, ArtifactAction.GENERATE)
        await self._repo.save(artifact)
        await self._emit_status(artifact, step, task)

        # 根据 step kind 继续晋升（UPLOAD / SHARE）
        for action in _PROMOTE_BY_KIND.get(step.kind, []):
            artifact = transition(artifact, action)
            await self._repo.save(artifact)
            await self._emit_status(artifact, step, task)

        # 旧版本 supersede
        if parent is not None:
            await self._supersede_parent(parent, step, task, artifact)

        return artifact

    async def _supersede_parent(
        self,
        parent: Artifact,
        step: PlanStep,
        task: Task,
        new_artifact: Artifact,
    ) -> None:
        try:
            superseded = transition(parent, ArtifactAction.SUPERSEDE)
        except IllegalTransition:
            logger.warning(
                "pilot.artifact_supersede_skipped",
                parent_id=parent.artifact_id,
                parent_status=parent.status.value,
            )
            return
        await self._repo.save(superseded)
        await self._publisher.record(ExecutionEvent(
            workspace_id=superseded.workspace_id,
            trace_id=task.trace_id,
            task_id=task.task_id,
            plan_id=step.plan_id,
            step_id=step.step_id,
            artifact_id=superseded.artifact_id,
            type=EventType.ARTIFACT_STATUS_CHANGED,
            message=f"artifact superseded by {new_artifact.artifact_id}",
            payload={
                "status": superseded.status.value,
                "superseded_by": new_artifact.artifact_id,
            },
        ))

    async def _emit_status(
        self,
        artifact: Artifact,
        step: PlanStep,
        task: Task,
    ) -> None:
        await self._publisher.record(ExecutionEvent(
            workspace_id=artifact.workspace_id,
            trace_id=task.trace_id,
            task_id=task.task_id,
            plan_id=step.plan_id,
            step_id=step.step_id,
            artifact_id=artifact.artifact_id,
            type=EventType.ARTIFACT_STATUS_CHANGED,
            message=f"artifact status: {artifact.status.value}",
            payload={"status": artifact.status.value},
        ))

    # ── 内容序列化 / 落盘 ─────────────────────────

    def _materialize_content(
        self,
        content: Any,  # noqa: ANN401 - 多态 payload
        mime_type: str | None,
    ) -> tuple[str | None, str | None, str | None, str | bytes | None]:
        """把 ``content`` 落盘或写入内存兜底。

        Returns:
            (storage_key, checksum, uri, raw_bytes_or_str)
            其中 ``raw_bytes_or_str`` 若非 None，调用方应在拿到
            ``artifact_id`` 后写入 memory_blobs（仅内存模式）。
        """
        if content is None:
            return None, None, None, None

        body, ext = self._encode(content, mime_type)
        checksum = hashlib.sha256(
            body if isinstance(body, bytes) else body.encode("utf-8"),
        ).hexdigest()

        if self._artifact_dir is None:
            # 内存兜底：调用方在 save() 中绑定 artifact_id 后再写入。
            return _MEMORY_SCHEME, checksum, None, body

        # 直接以 checksum 作为文件名前缀，避免每次重新计算路径
        file_name = f"{checksum}{ext}"
        # 真实路径在 save() 时还不知道 workspace_id？知道的——
        # 但本方法在拿到 artifact 前调用，因此先用 checksum 命名，落到
        # 全局目录下；后续可按 workspace 分桶（M5 再细化）。
        target_dir = self._artifact_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / file_name
        if isinstance(body, bytes):
            file_path.write_bytes(body)
        else:
            file_path.write_text(body, encoding="utf-8")

        storage_key = str(file_path.relative_to(self._artifact_dir.parent)) \
            if self._artifact_dir.parent != Path() \
            else str(file_path)
        uri = file_path.resolve().as_uri()
        return storage_key, checksum, uri, None

    @staticmethod
    def _encode(
        content: Any,  # noqa: ANN401 - 多态 payload
        mime_type: str | None,
    ) -> tuple[str | bytes, str]:
        if isinstance(content, bytes):
            return content, ".bin"
        if isinstance(content, str):
            ext = ".md" if (mime_type or "").startswith("text/markdown") \
                else ".txt"
            return content, ext
        if isinstance(content, dict | list):
            return json.dumps(
                content, ensure_ascii=False, indent=2, sort_keys=True,
            ), ".json"
        msg = f"不支持的 artifact content 类型: {type(content).__name__}"
        raise TypeError(msg)

    # ── 读取 ───────────────────────────────────────

    async def read_content(
        self,
        artifact: Artifact,
    ) -> str | dict[str, Any] | bytes | None:
        """读取 Artifact 内容。

        约定：
        - 内存模式：直接从 ``self._memory_blobs`` 取出原始数据；
        - 文件模式：按 ``mime_type`` 决定返回类型；
        - 没有 ``storage_key`` 的 Artifact 返回 ``None``。
        """
        key = artifact.storage_key
        if not key:
            return None

        raw: str | bytes | None
        if key.startswith(_MEMORY_SCHEME):
            blob = self._memory_blobs.get(artifact.artifact_id)
            raw = blob
        else:
            path = self._resolve_disk_path(key, artifact.uri)
            if path is None or not path.exists():
                return None
            mime = artifact.mime_type or ""
            if mime.startswith("text/") or mime in (
                "application/json", "application/x-yaml",
            ):
                raw = path.read_text(encoding="utf-8")
            else:
                raw = path.read_bytes()

        if raw is None:
            return None
        if (artifact.mime_type or "") == "application/json" and isinstance(
            raw, str,
        ):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        return raw

    def _resolve_disk_path(
        self,
        storage_key: str,
        uri: str | None,
    ) -> Path | None:
        if uri and uri.startswith("file://"):
            from urllib.parse import urlparse
            from urllib.request import url2pathname
            parsed = urlparse(uri)
            return Path(url2pathname(parsed.path))
        if self._artifact_dir is not None:
            candidate = (self._artifact_dir.parent / storage_key)
            if candidate.exists():
                return candidate
            candidate2 = self._artifact_dir / Path(storage_key).name
            if candidate2.exists():
                return candidate2
        return Path(storage_key)


    async def read_content_by_id(
        self,
        artifact_id: str,
    ) -> str | dict[str, Any] | bytes | None:
        """便捷接口：按 ``artifact_id`` 读取内容，缺失返回 ``None``。"""
        artifact = await self._repo.get_artifact(artifact_id)
        if artifact is None:
            return None
        return await self.read_content(artifact)


__all__ = ["ArtifactStore"]
