"""向后兼容包装层。

正规定义已移至 agent_hub.contracts；本模块仅做 re-export，
供尚未迁移的代码继续使用，不应在新代码中直接 import。
"""

# backward-compat: re-export from contracts
from agent_hub.contracts.execution import ExecutionMode, NodeType
from agent_hub.contracts.identity import ChannelType, UserRole

__all__ = [
    "ChannelType",
    "ExecutionMode",
    "NodeType",
    "UserRole",
]
