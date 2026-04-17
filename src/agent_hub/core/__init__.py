"""核心模块：枚举、数据模型、Agent 基类、意图路由器、追踪、Pipeline。"""

from agent_hub.core.base_agent import BaseAgent
from agent_hub.core.enums import IntentType, UserRole
from agent_hub.core.models import (
    AgentResult,
    GuardResult,
    MemoryEntry,
    ReActStep,
    ReActTrace,
    RoutingResult,
    SubTask,
    TaskInput,
    TaskOutput,
    ToolSpec,
    UserContext,
)
from agent_hub.core.router import IntentRouter
from agent_hub.core.tracer import SpanContext, get_current_trace_id, init_tracer


def __getattr__(name: str):  # noqa: ANN001
    """延迟导入 AgentPipeline 以避免循环引用。"""
    if name == "AgentPipeline":
        from agent_hub.core.pipeline import AgentPipeline

        return AgentPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentPipeline",
    "AgentResult",
    "BaseAgent",
    "GuardResult",
    "IntentRouter",
    "IntentType",
    "MemoryEntry",
    "ReActStep",
    "ReActTrace",
    "RoutingResult",
    "SpanContext",
    "SubTask",
    "TaskInput",
    "TaskOutput",
    "ToolSpec",
    "UserContext",
    "UserRole",
    "get_current_trace_id",
    "init_tracer",
]
