"""FastAPI HTTP 路由。

提供统一的 REST API 入口，将 HTTP 请求转发到 :class:`AgentPipeline`。

启动方式::

    uvicorn agent_hub.api.routes:app --reload --port 8080
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from agent_hub.config.settings import Settings, get_settings
from agent_hub.core.enums import UserRole
from agent_hub.core.models import TaskInput, UserContext
from agent_hub.core.pipeline import AgentPipeline
from agent_hub.core.tracer import init_tracer

logger = structlog.get_logger(__name__)

# ── 全局状态 ──────────────────────────────────────────

_pipeline: AgentPipeline | None = None
_settings: Settings | None = None


def _get_pipeline() -> AgentPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline 未初始化")
    return _pipeline


# ── Lifespan ─────────────────────────────────────────


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期：启动时初始化 Pipeline，关闭时清理资源。"""
    global _pipeline, _settings  # noqa: PLW0603
    _settings = get_settings()
    init_tracer("agent-hub-api")
    _pipeline = AgentPipeline(_settings)
    logger.info("api_started")
    yield
    _pipeline = None
    logger.info("api_stopped")


# ── FastAPI App ──────────────────────────────────────

app = FastAPI(
    title="Agent-Hub API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — 从配置读取
_cors_settings = get_settings()
_origins = [o.strip() for o in _cors_settings.cors_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── 请求 / 响应模型 ─────────────────────────────────


class ChatRequest(BaseModel):
    """对话请求体。"""

    message: str = Field(..., description="用户消息")
    user_id: str = Field(..., description="用户ID")
    role: str = Field(default="user", description="角色：admin / user")
    channel: str = Field(default="api", description="渠道")
    session_id: str | None = Field(default=None, description="会话ID")
    group_id: str | None = Field(default=None, description="群ID")
    trace_id: str | None = Field(default=None, description="链路追踪ID")


class ChatResponse(BaseModel):
    """对话响应体。"""

    trace_id: str
    response: str
    status: str
    total_duration_ms: int


# ── 路由 ─────────────────────────────────────────────


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """统一对话入口。"""
    pipeline = _get_pipeline()

    user_ctx = UserContext(
        user_id=req.user_id,
        role=UserRole(req.role),
        channel=req.channel,
        session_id=req.session_id or req.user_id,
        group_id=req.group_id,
        is_private=req.group_id is None,
    )

    task_input = TaskInput(
        trace_id=req.trace_id or str(uuid.uuid4()),
        user_context=user_ctx,
        raw_message=req.message,
        attachments=[],
    )

    output = await pipeline.run(task_input)

    return ChatResponse(
        trace_id=output.trace_id,
        response=output.response,
        status=output.status,
        total_duration_ms=output.total_duration_ms,
    )


@app.get("/health")
async def health() -> dict[str, object]:
    """健康检查。"""
    return {"status": "ok", "pipeline": _pipeline is not None}


@app.get("/tools")
async def list_tools(role: str = "user") -> dict[str, list[dict[str, str]]]:
    """列出当前用户可用的工具。"""
    pipeline = _get_pipeline()
    tools = pipeline._registry.list_tools_for_user(role)
    return {"tools": [{"name": t.name, "description": t.description} for t in tools]}


@app.get("/trace/{trace_id}")
async def get_trace(trace_id: str) -> dict[str, str]:
    """查询链路追踪记录（占位实现）。"""
    return {"trace_id": trace_id, "status": "tracked"}


# ── SSE 流式端点 ─────────────────────────────────────


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> EventSourceResponse:
    """SSE 流式对话入口。

    返回 Server-Sent Events 流，逐 token 推送 LLM 生成结果。
    """
    from agent_hub.api.streaming import sse_generator

    pipeline = _get_pipeline()

    user_ctx = UserContext(
        user_id=req.user_id,
        role=UserRole(req.role),
        channel=req.channel,
        session_id=req.session_id or req.user_id,
        group_id=req.group_id,
        is_private=req.group_id is None,
    )

    task_input = TaskInput(
        trace_id=req.trace_id or str(uuid.uuid4()),
        user_context=user_ctx,
        raw_message=req.message,
        attachments=[],
    )

    return EventSourceResponse(sse_generator(pipeline, task_input))
