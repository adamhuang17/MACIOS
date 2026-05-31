"""跨层执行契约。

定义：执行模式枚举、任务输入、子任务/执行步骤、路由决策、
ReAct 推理链路、Agent 执行结果、记忆条目、任务输出。

这些类型是 core Pipeline / runtime / API 之间传递执行上下文的唯一公共格式。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_hub.contracts.artifact import Artifact
from agent_hub.contracts.identity import SourceContext, UserContext

# ── 执行模式枚举 ──────────────────────────────────────────────────────────────


class ExecutionMode(StrEnum):
    """路由决策执行模式。

    代表"下一步如何处理"，由 router 对用户请求决策后输出。
    """

    IGNORE = "ignore"       # 忽略 / 轻量回复
    QA = "qa"               # 知识问答 / 检索
    PLAN = "plan"           # 内容生成 / 复杂多步骤任务
    ACT = "act"             # 工具执行
    DELEGATE = "delegate"   # 委派给外部 Agent
    REPAIR = "repair"       # 反思 / 修复


class NodeType(StrEnum):
    """工作流节点类型（基础设施节点）。"""

    LLM = "llm"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    QUESTION_CLASSIFIER = "question_classifier"
    IF_ELSE = "if_else"
    TOOL = "tool"
    LOOP = "loop"


# ── 任务输入 ──────────────────────────────────────────────────────────────────


class TaskInput(BaseModel):
    """用户输入的原始请求（跨层传递格式）。"""

    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_context: UserContext
    source_context: SourceContext
    raw_message: str
    attachments: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    parent_trace_id: str | None = None


# ── 子任务 / 执行步骤 ─────────────────────────────────────────────────────────


class SubTask(BaseModel):
    """路由层拆分出的子任务，支持 DAG 依赖编排。"""

    model_config = ConfigDict(extra="forbid")

    subtask_id: str
    description: str
    required_agents: list[str]
    priority: int = Field(default=1, ge=1, le=5, description="优先级（1 最高，5 最低）")
    depends_on: list[str] = Field(default_factory=list)


# ── 路由决策 ──────────────────────────────────────────────────────────────────


class RoutingDecision(BaseModel):
    """路由决策层输出（含策略字段，由 runtime/policy 补充填写）。"""

    model_config = ConfigDict(extra="forbid")

    mode: ExecutionMode
    targets: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    plan: list[SubTask] = Field(default_factory=list)
    branch_label: str = ""
    low_confidence: bool = False

    # 由 runtime/policy 层填入
    route_source: str = "main"
    agent_id: str | None = None
    session_key: str | None = None
    tool_profile: str = "read_only"
    risk_level: str = "read"
    requires_approval: bool = False


# ── ReAct 推理链路 ────────────────────────────────────────────────────────────


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


# ── 记忆条目（跨层输出格式，非 memory 内部存储实体） ──────────────────────────


class MemoryEntry(BaseModel):
    """记忆条目：会话态和持久态共用的跨层输出格式。"""

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


# ── Agent 执行结果 ────────────────────────────────────────────────────────────


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


# ── 任务输出 ──────────────────────────────────────────────────────────────────


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


__all__ = [
    "AgentResult",
    "ExecutionMode",
    "MemoryEntry",
    "NodeType",
    "ReActStep",
    "ReActTrace",
    "RoutingDecision",
    "SubTask",
    "TaskInput",
    "TaskOutput",
]
