"""向后兼容包装层。

正规定义已移至 agent_hub.runtime.policy.risk；本模块仅做 re-export，
供尚未迁移的代码继续使用，不应在新代码中直接 import。
"""

# backward-compat: re-export from runtime.policy.risk
from agent_hub.runtime.policy.risk import (
    RiskDecision,
    RiskPolicy,
    ToolProfile,
    parse_tool_profile,
)

__all__ = [
    "RiskDecision",
    "RiskPolicy",
    "ToolProfile",
    "parse_tool_profile",
]
