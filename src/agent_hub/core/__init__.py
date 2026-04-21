"""核心模块：枚举、数据模型、Agent 基类、路由决策器、追踪、Pipeline。"""

from agent_hub.core.base_agent import BaseAgent
from agent_hub.core.enums import ChannelType, ExecutionMode, NodeType, UserRole
from agent_hub.core.models import (
    AgentResult,
    GuardResult,
    MemoryEntry,
    ReActStep,
    ReActTrace,
    RoutingDecision,
    SubTask,
    TaskInput,
    TaskOutput,
    ToolSpec,
    UserContext,
)
from agent_hub.core.router import DecisionRouter, IntentRouter
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
    "ChannelType",
    "DecisionRouter",
    "ExecutionMode",
    "GuardResult",
    "IntentRouter",  # 过渡期别名
    "MemoryEntry",
    "NodeType",
    "ReActStep",
    "ReActTrace",
    "RoutingDecision",
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
