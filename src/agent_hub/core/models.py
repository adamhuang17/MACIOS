"""向后兼容包装层。

正规定义已移至 agent_hub.contracts；本模块仅做 re-export，
供尚未迁移的代码继续使用，不应在新代码中直接 import。
"""

# backward-compat: re-export from contracts
from agent_hub.contracts.artifact import Artifact
from agent_hub.contracts.capability import ToolSpec
from agent_hub.contracts.execution import (
    AgentResult,
    MemoryEntry,
    ReActStep,
    ReActTrace,
    RoutingDecision,
    SubTask,
    TaskInput,
    TaskOutput,
)
from agent_hub.contracts.identity import (
    SourceChatType,
    SourceContext,
    UserContext,
)
from agent_hub.contracts.policy import GuardResult

__all__ = [
    "AgentResult",
    "Artifact",
    "GuardResult",
    "MemoryEntry",
    "ReActStep",
    "ReActTrace",
    "RoutingDecision",
    "SourceChatType",
    "SourceContext",
    "SubTask",
    "TaskInput",
    "TaskOutput",
    "ToolSpec",
    "UserContext",
]
