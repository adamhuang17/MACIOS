"""runtime.policy — 风险策略与源绑定。

从 core/risk.py 和 core/binding.py 迁移的规范定义。
原 core 模块保留为向后兼容的 re-export 包装。

公开 API:
    BindingResult, BindingRule, SourceBindingResolver
    RiskDecision, RiskPolicy, ToolProfile, parse_tool_profile
    build_main_session_key, build_source_session_key
"""

from agent_hub.runtime.policy.risk import (
    RiskDecision,
    RiskPolicy,
    ToolProfile,
    parse_tool_profile,
)
from agent_hub.runtime.policy.source_binding import (
    BindingResult,
    BindingRule,
    SourceBindingResolver,
    build_main_session_key,
    build_source_session_key,
)

__all__ = [
    "BindingResult",
    "BindingRule",
    "RiskDecision",
    "RiskPolicy",
    "SourceBindingResolver",
    "ToolProfile",
    "build_main_session_key",
    "build_source_session_key",
    "parse_tool_profile",
]
