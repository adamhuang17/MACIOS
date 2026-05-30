"""核心 Pydantic 数据模型。

定义多 Agent 协同系统中跨模块流转的公共数据结构，涵盖：
用户上下文、来源上下文、任务输入、路由决策、子任务、工具规范、
Agent 执行结果、记忆条目与产物描述。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_hub.core.enums import ExecutionMode, UserRole

# ── 用户 / 来源上下文 ──────────────────────────────────


class UserContext(BaseModel):
    """用户上下文：只描述调用方身份与权限事实。"""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    role: UserRole


class SourceChatType(StrEnum):
    """来源会话类型。"""

    DIRECT = "direct"
    GROUP = "group"
    CHANNEL = "channel"
    UNKNOWN = "unknown"


class SourceContext(BaseModel):
    """消息来源上下文：描述这条请求来自哪个渠道、会话和消息。"""

    model_config = ConfigDict(extra="forbid")

    channel: str
    account_id: str | None = None
    chat_id: str | None = None
    chat_type: SourceChatType = SourceChatType.UNKNOWN
    thread_id: str | None = None
    message_id: str | None = None
    sender_id: str | None = None
    sender_id_type: str | None = None
    team_id: str | None = None
    guild_id: str | None = None
    member_role_ids: list[str] = Field(default_factory=list)
    is_at_bot: bool = False
    is_command_prefix: bool = False
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_user_context(
        cls,
        user_context: UserContext,
        *,
        channel: str = "unknown",
        is_at_bot: bool = False,
        is_command_prefix: bool = False,
    ) -> SourceContext:
        """从身份上下文构造最小来源上下文。"""
        return cls(
            channel=channel,
            chat_type=SourceChatType.DIRECT,
            sender_id=user_context.user_id,
            is_at_bot=is_at_bot,
            is_command_prefix=is_command_prefix,
        )


# ── 任务输入 ──────────────────────────────────────────


class TaskInput(BaseModel):
    """用户输入的原始请求。"""

    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_context: UserContext
    source_context: SourceContext
    raw_message: str
    attachments: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    parent_trace_id: str | None = None


# ── 子任务 & 路由结果 ────────────────────────────────


class SubTask(BaseModel):
    """路由层拆分出的子任务，支持 DAG 依赖编排。"""
    model_config = ConfigDict(extra="forbid")

    subtask_id: str
    description: str
    required_agents: list[str]
    priority: int = Field(default=1, ge=1, le=5, description="优先级（1 最高，5 最低）")
    depends_on: list[str] = Field(default_factory=list)


class RoutingDecision(BaseModel):
    """路由决策层输出。"""

    model_config = ConfigDict(extra="forbid")

    mode: ExecutionMode
    targets: list[str] = Field(default_factory=list)

    # 最终权限事实由 RiskPolicy 重算，Router 不授予权限。
    requires_admin: bool = False

    capabilities: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    plan: list[SubTask] = Field(default_factory=list)
    branch_label: str = ""
    low_confidence: bool = False

    # 新字段：SourceBindingResolver / RiskPolicy 的确定性结果。
    route_source: str = "main"
    agent_id: str | None = None
    session_key: str | None = None
    tool_profile: str = "read_only"
    risk_level: str = "read"
    requires_approval: bool = False


# ── ReAct 推理链路 ───────────────────────────────────


class ReActStep(BaseModel):
    """ReAct 推理循环中的单步记录。"""
    model_config = ConfigDict(extra="forbid")

    thought: str
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None


class ReActTrace(BaseModel):
    """完整的 ReAct 推理链路。"""
    model_config = ConfigDict(extra="forbid")

    steps: list[ReActStep] = Field(default_factory=list)
    final_answer: str | None = None
    total_rounds: int = 0


# ── 安全守卫结果 ─────────────────────────────────────


class GuardResult(BaseModel):
    """Prompt 注入检测结果。"""
    model_config = ConfigDict(extra="forbid")

    is_safe: bool
    risk_level: str = "safe"
    matched_rules: list[str] = Field(default_factory=list)
    llm_analysis: str | None = None
    confidence: float = 1.0


# ── 工具规范 ──────────────────────────────────────────


class ToolSpec(BaseModel):
    """可调用工具的规范描述。"""
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    params_schema: dict[str, Any] = Field(default_factory=dict)
    requires_admin: bool = False
    risk_level: str = "read"
    side_effect: bool = False
    categories: list[str] = Field(default_factory=list)
    tool_profiles: list[str] = Field(default_factory=list)


# ── Agent 执行结果 ───────────────────────────────────


class AgentResult(BaseModel):
    """单个 Agent 的执行结果。"""
    model_config = ConfigDict(extra="forbid")

    agent_name: str
    success: bool
    output: Any | None = None
    error: str | None = None
    duration_ms: int
    trace_id: str
    react_trace: ReActTrace | None = None


# ── 记忆条目 ─────────────────────────────────────────


class MemoryEntry(BaseModel):
    """记忆条目：会话态和持久态共用格式。"""
    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    memory_type: str
    session_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    ttl: int = -1
    is_deleted: bool = False
    embedding: list[float] | None = Field(default=None, exclude=True)


# ── Artifact / 最终输出 ───────────────────────────────


class Artifact(BaseModel):
    """轻量产物描述，供 core Pipeline 和 API 输出使用。

    Pilot 领域内仍保留更完整的 Artifact 实体；本模型用于 core 层统一输出。
    """

    artifact_id: str = Field(default_factory=lambda: f"art_{uuid.uuid4().hex[:16]}")
    type: str
    title: str
    status: str = "generated"
    uri: str | None = None
    storage_key: str | None = None
    mime_type: str | None = None
    checksum: str | None = None
    share_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class TaskOutput(BaseModel):
    """Pipeline 执行完毕的完整结果。"""
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    user_id: str
    response: str
    agent_results: list[AgentResult] = Field(default_factory=list)
    memory_saved: list[MemoryEntry] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    total_duration_ms: int
    status: str = "success"
