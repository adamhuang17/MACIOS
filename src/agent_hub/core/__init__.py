"""核心模块：枚举、数据模型、Agent 基类、路由决策器、追踪、Pipeline。"""

from agent_hub.core.base_agent import BaseAgent
from agent_hub.core.binding import BindingResult, BindingRule, SourceBindingResolver
from agent_hub.core.enums import ChannelType, ExecutionMode, NodeType, UserRole
from agent_hub.core.models import (
    AgentResult,
    Artifact,
    GuardResult,
    MemoryEntry,
    ReActStep,
    ReActTrace,
    RoutingDecision,
    SourceChatType,
    SourceContext,
    SubTask,
    TaskInput,
    TaskOutput,
    ToolSpec,
    UserContext,
)
from agent_hub.core.plan_validator import (
    PlanValidationResult,
    SubTaskPlanValidator,
    validate_subtask_plan,
)
from agent_hub.core.risk import RiskDecision, RiskPolicy, ToolProfile
from agent_hub.core.router import DecisionRouter
from agent_hub.core.tracer import SpanContext, get_current_trace_id, init_tracer


def __getattr__(name: str) -> object:  # noqa: ANN001
    """延迟导入 AgentPipeline 以避免循环引用。"""
    if name == "AgentPipeline":
        from agent_hub.core.pipeline import AgentPipeline

        return AgentPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentPipeline",
    "AgentResult",
    "Artifact",
    "BaseAgent",
    "BindingResult",
    "BindingRule",
    "ChannelType",
    "DecisionRouter",
    "ExecutionMode",
    "GuardResult",
    "MemoryEntry",
    "NodeType",
    "PlanValidationResult",
    "ReActStep",
    "ReActTrace",
    "RiskDecision",
    "RiskPolicy",
    "RoutingDecision",
    "SourceChatType",
    "SourceBindingResolver",
    "SourceContext",
    "SpanContext",
    "SubTask",
    "SubTaskPlanValidator",
    "TaskInput",
    "TaskOutput",
    "ToolSpec",
    "ToolProfile",
    "UserContext",
    "UserRole",
    "get_current_trace_id",
    "init_tracer",
    "validate_subtask_plan",
]
