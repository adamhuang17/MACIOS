"""向后兼容包装层。

正规定义已移至 agent_hub.runtime.observability.tracing；本模块仅做 re-export，
供尚未迁移的代码继续使用，不应在新代码中直接 import。
"""

# backward-compat: re-export from runtime.observability.tracing
from agent_hub.runtime.observability.tracing import (
    SpanContext,
    get_current_trace_id,
    get_tracer,
    init_tracer,
)

__all__ = [
    "SpanContext",
    "get_current_trace_id",
    "get_tracer",
    "init_tracer",
]