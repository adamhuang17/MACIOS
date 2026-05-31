"""跨层能力与工具契约。

定义：工具规范、工具调用请求、工具调用结果、模型可见能力引用。
这些类型是 capabilities 层、runtime 层和 LLM actor 之间传递工具信息的公共格式。
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolSpec(BaseModel):
    """可调用工具的规范描述（供模型和 ToolCatalog 使用）。"""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    params_schema: dict[str, Any] = Field(default_factory=dict)
    requires_admin: bool = False
    risk_level: str = "read"
    side_effect: bool = False
    categories: list[str] = Field(default_factory=list)
    tool_profiles: list[str] = Field(default_factory=list)


class ToolCall(BaseModel):
    """模型发出的单次工具调用请求。"""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class ToolResult(BaseModel):
    """工具调用返回结果。"""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    tool_name: str
    output: Any
    error: str | None = None
    duration_ms: int | None = None


class InstructionSkillRef(BaseModel):
    """给模型看的能力说明引用（不是可执行工具，是提示词能力描述）。"""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    location: str | None = None
    disable_model_invocation: bool = False


__all__ = [
    "InstructionSkillRef",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
]
