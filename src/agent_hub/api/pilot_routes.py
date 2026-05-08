"""Pilot M2 FastAPI 路由。

挂载到主 ``app``：``app.include_router(build_pilot_router(runtime))``。
路由层只做 DTO 校验、错误映射与服务转发；不直接调 ``transition()``。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette.sse import EventSourceResponse

from agent_hub.api.pilot_dto import (
    ApprovalDecisionRequest,
    ApprovalDecisionResponse,
    ArtifactView,
    RecoveryRequest,
    ResumeTaskResponse,
    SubmitTaskRequest,
    SubmitTaskResponse,
    TaskDetailView,
    TaskSummaryView,
    WorkspaceSummaryView,
)
from agent_hub.pilot.domain.enums import TaskStatus
from agent_hub.pilot.domain.errors import (
    ConcurrencyConflict,
    IllegalTransition,
    PilotDomainError,
)
from agent_hub.pilot.services.commands import CommandError
from agent_hub.pilot.services.dto import TaskHandle, TaskRequest

if TYPE_CHECKING:
    from agent_hub.api.pilot_runtime import PilotRuntime


# ── Router 工厂 ──────────────────────────────────────


def build_pilot_router(runtime: PilotRuntime) -> APIRouter:
    """构造 ``/api/pilot`` 路由。

    runtime 通过闭包注入；这样测试可以独立 build router 而无需挂全局。
    """
    router = APIRouter(prefix="/api/pilot", tags=["pilot"])

    def _require_admin(token: str | None) -> None:
        expected = runtime.settings.pilot_admin_token
        if not expected:
            raise HTTPException(status_code=403, detail="admin route disabled")
        if token != expected:
            raise HTTPException(status_code=401, detail="invalid admin token")

    # ── tasks ──────────────────────────────────────

    @router.post("/tasks", response_model=SubmitTaskResponse)
    async def submit_task(req: SubmitTaskRequest) -> SubmitTaskResponse:
        try:
            handle = await runtime.orchestrator.submit(_to_task_request(req))
        except PilotDomainError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return SubmitTaskResponse(
            workspace_id=handle.workspace_id,
            task_id=handle.task_id,
            plan_id=handle.plan_id,
            status=handle.status,
            trace_id=handle.trace_id,
            deduplicated=handle.deduplicated,
            detail_url=runtime.settings.public_url(
                f"/api/pilot/tasks/{handle.task_id}",
            ),
            events_url=runtime.settings.public_url(
                f"/api/pilot/tasks/{handle.task_id}/events",
            ),
        )

    @router.get("/tasks", response_model=list[TaskSummaryView])
    async def list_tasks(
        workspace_id: str | None = Query(default=None),  # noqa: B008
        status: TaskStatus | None = Query(default=None),  # noqa: B008
        limit: int = Query(default=50, ge=1, le=500),  # noqa: B008
    ) -> list[TaskSummaryView]:
        items = await runtime.queries.list_tasks(
            workspace_id=workspace_id, status=status, limit=limit,
        )
        return [TaskSummaryView.model_validate(s, from_attributes=True) for s in items]

    @router.get("/tasks/{task_id}", response_model=TaskDetailView)
    async def get_task(task_id: str) -> TaskDetailView:
        detail = await runtime.queries.get_task_detail(task_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"unknown task: {task_id}")
        return TaskDetailView.model_validate(detail, from_attributes=True)

    @router.post("/tasks/{task_id}/steps/{step_id}/retry",
                 response_model=SubmitTaskResponse)
    async def retry_step(
        task_id: str,
        step_id: str,
        body: RecoveryRequest,
    ) -> SubmitTaskResponse:
        try:
            handle = await runtime.commands.create_recovery_task(
                task_id,
                requester_id=body.requester_id,
                from_step_id=step_id,
                auto_approve=body.auto_approve,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except CommandError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _handle_to_response(handle, runtime)

    @router.post("/tasks/{task_id}/recover_from/{step_id}",
                 response_model=SubmitTaskResponse)
    async def recover_from(
        task_id: str,
        step_id: str,
        body: RecoveryRequest,
        x_pilot_admin_token: str | None = Header(default=None),
    ) -> SubmitTaskResponse:
        _require_admin(x_pilot_admin_token)
        try:
            handle = await runtime.commands.create_recovery_task(
                task_id,
                requester_id=body.requester_id,
                from_step_id=step_id,
                auto_approve=body.auto_approve,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except CommandError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _handle_to_response(handle, runtime)

    @router.post("/tasks/{task_id}/resume", response_model=ResumeTaskResponse)
    async def resume_task(task_id: str) -> ResumeTaskResponse:
        try:
            payload = await runtime.commands.resume_task(task_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except CommandError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except IllegalTransition as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ConcurrencyConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return ResumeTaskResponse(**payload)

    # ── workspaces ────────────────────────────────

    @router.get("/workspaces", response_model=list[WorkspaceSummaryView])
    async def list_workspaces() -> list[WorkspaceSummaryView]:
        items = await runtime.queries.list_workspaces()
        return [
            WorkspaceSummaryView.model_validate(s, from_attributes=True)
            for s in items
        ]

    @router.get("/workspaces/{workspace_id}", response_model=WorkspaceSummaryView)
    async def get_workspace(workspace_id: str) -> WorkspaceSummaryView:
        ws = await runtime.repository.get_workspace(workspace_id)
        if ws is None:
            raise HTTPException(
                status_code=404, detail=f"unknown workspace: {workspace_id}",
            )
        # 简单聚合
        tasks = await runtime.repository.list_tasks(workspace_id=workspace_id)
        events = await runtime.event_store.list_events(
            workspace_id, since_sequence=0, limit=10_000,
        )
        return WorkspaceSummaryView(
            workspace=ws,
            task_count=len(tasks),
            active_task_id=ws.active_task_id,
            latest_sequence=events[-1].sequence if events else 0,
        )

    # ── artifacts ─────────────────────────────────

    @router.get("/artifacts/{artifact_id}", response_model=ArtifactView)
    async def get_artifact(artifact_id: str) -> ArtifactView:
        art = await runtime.queries.get_artifact(artifact_id)
        if art is None:
            raise HTTPException(
                status_code=404, detail=f"unknown artifact: {artifact_id}",
            )
        return ArtifactView.model_validate(art, from_attributes=True)

    @router.get("/artifacts/{artifact_id}/content")
    async def get_artifact_content(
        artifact_id: str,
        max_bytes: int = Query(default=8192, ge=256, le=131072),  # noqa: B008
    ) -> dict[str, object]:
        """只读取 text/json artifact 的预览内容。

        - text/json 类型：返回 ``kind=text|json`` 与 ``content``；
          超长按字符截断并设置 ``truncated=true``。
        - 二进制（如 PPTX）：返回 ``kind=binary`` + 元数据，**不**返回 bytes。
        """
        art = await runtime.queries.get_artifact(artifact_id)
        if art is None:
            raise HTTPException(
                status_code=404, detail=f"unknown artifact: {artifact_id}",
            )
        content = await runtime.artifacts.read_content_by_id(artifact_id)
        if content is None:
            return {"kind": "empty", "artifact_id": artifact_id,
                    "mime_type": art.mime_type}
        if isinstance(content, dict | list):
            return {"kind": "json", "artifact_id": artifact_id,
                    "mime_type": art.mime_type or "application/json",
                    "content": content, "truncated": False}
        if isinstance(content, str):
            truncated = len(content) > max_bytes
            return {"kind": "text", "artifact_id": artifact_id,
                    "mime_type": art.mime_type or "text/plain",
                    "content": content[:max_bytes],
                    "truncated": truncated}
        # bytes：仅返回元数据，避免在 dashboard 直接传输 PPT
        return {"kind": "binary", "artifact_id": artifact_id,
                "mime_type": art.mime_type, "size_bytes": len(content),
                "share_url": art.share_url, "uri": art.uri}

    @router.get("/artifacts/{artifact_id}/download")
    async def download_artifact(artifact_id: str):  # noqa: ANN202
        """二进制 / 文件型 artifact 的真实下载入口。

        - 落盘 artifact：返回 :class:`FileResponse`，浏览器会按
          ``mime_type`` 与 ``filename`` 触发下载/预览；
        - 内存 / bytes artifact：直接 :class:`Response` 回传字节流；
        - text/json：以原文 + 合理 ``content-type`` 下载，方便取证。
        """
        from urllib.parse import quote

        art = await runtime.queries.get_artifact(artifact_id)
        if art is None:
            raise HTTPException(
                status_code=404, detail=f"unknown artifact: {artifact_id}",
            )

        # 文件名：优先用 artifact.title，去掉路径分隔符做最小净化。
        raw_name = (art.title or f"{artifact_id}").strip() or artifact_id
        safe_name = raw_name.replace("/", "_").replace("\\", "_")
        # RFC 5987：filename* 用 UTF-8 编码 + filename 兜底 ASCII。
        ascii_fallback = safe_name.encode("ascii", "ignore").decode("ascii") or "artifact"
        disposition = (
            f'attachment; filename="{ascii_fallback}"; '
            f"filename*=UTF-8''{quote(safe_name)}"
        )
        media_type = art.mime_type or "application/octet-stream"

        # 优先走磁盘路径，避免大文件读到内存。
        store = runtime.artifacts
        key = art.storage_key or ""
        if key and not key.startswith("memory://"):
            disk_path = store._resolve_disk_path(key, art.uri)  # noqa: SLF001
            if disk_path is not None and disk_path.exists():
                return FileResponse(
                    path=str(disk_path),
                    media_type=media_type,
                    filename=safe_name,
                    headers={"Content-Disposition": disposition},
                )

        content = await store.read_content_by_id(artifact_id)
        if content is None:
            raise HTTPException(
                status_code=404,
                detail=f"artifact has no content to download: {artifact_id}",
            )
        if isinstance(content, bytes):
            body: bytes = content
        elif isinstance(content, str):
            body = content.encode("utf-8")
        else:
            import json as _json

            body = _json.dumps(content, ensure_ascii=False).encode("utf-8")
            media_type = media_type or "application/json"

        return Response(
            content=body,
            media_type=media_type,
            headers={"Content-Disposition": disposition},
        )

    # ── approvals ─────────────────────────────────

    @router.post("/approvals/{approval_id}/decision",
                 response_model=ApprovalDecisionResponse)
    async def decide_approval(
        approval_id: str,
        body: ApprovalDecisionRequest,
    ) -> ApprovalDecisionResponse:
        try:
            payload = await runtime.commands.decide_approval(
                approval_id,
                decision=body.decision,
                actor_id=body.actor_id,
                comment=body.comment,
                run_after_approval=body.run_after_approval,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except CommandError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except IllegalTransition as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ConcurrencyConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return ApprovalDecisionResponse(**payload)

    # ── events SSE ────────────────────────────────

    @router.get("/tasks/{task_id}/events")
    async def task_events(
        task_id: str,
        request: Request,  # noqa: ARG001  - FastAPI 注入
        since_sequence: int = Query(default=0, ge=0),  # noqa: B008
    ) -> EventSourceResponse:
        from agent_hub.api.pilot_streaming import pilot_event_stream

        workspace_id = await runtime.queries.get_workspace_id_for_task(task_id)
        if workspace_id is None:
            raise HTTPException(status_code=404, detail=f"unknown task: {task_id}")
        return EventSourceResponse(pilot_event_stream(
            workspace_id=workspace_id,
            task_id=task_id,
            since_sequence=since_sequence,
            store=runtime.event_store,
            bus=runtime.bus,
            backfill_limit=runtime.settings.pilot_event_backfill_limit,
        ))

    return router


# ── DTO 转换 ─────────────────────────────────────────


def _to_task_request(req: SubmitTaskRequest) -> TaskRequest:
    return TaskRequest(
        source_channel=req.source_channel,
        raw_text=req.raw_text,
        requester_id=req.requester_id,
        title=req.title,
        source_conversation_id=req.source_conversation_id,
        source_message_id=req.source_message_id,
        workspace_id=req.workspace_id,
        tenant_key=req.tenant_key,
        attachments=list(req.attachments),
        idempotency_key=req.idempotency_key,
        metadata=dict(req.metadata),
        auto_approve=req.auto_approve,
        plan_profile=req.plan_profile,
    )


def _handle_to_response(
    handle: TaskHandle,
    runtime: PilotRuntime | None = None,
) -> SubmitTaskResponse:
    detail_path = f"/api/pilot/tasks/{handle.task_id}"
    events_path = f"/api/pilot/tasks/{handle.task_id}/events"
    if runtime is not None:
        detail_url = runtime.settings.public_url(detail_path)
        events_url = runtime.settings.public_url(events_path)
    else:
        detail_url = detail_path
        events_url = events_path

    return SubmitTaskResponse(
        workspace_id=handle.workspace_id,
        task_id=handle.task_id,
        plan_id=handle.plan_id,
        status=handle.status,
        trace_id=handle.trace_id,
        deduplicated=handle.deduplicated,
        detail_url=detail_url,
        events_url=events_url,
    )


__all__ = ["build_pilot_router"]


# 防止未使用警告
_ = (BaseModel, ConfigDict, Field)
