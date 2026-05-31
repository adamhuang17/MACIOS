"""向后兼容包装层。

正规定义已移至 agent_hub.runtime.policy.source_binding；本模块仅做 re-export，
供尚未迁移的代码继续使用，不应在新代码中直接 import。
"""

# backward-compat: re-export from runtime.policy.source_binding
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
    "SourceBindingResolver",
    "build_main_session_key",
    "build_source_session_key",
]
