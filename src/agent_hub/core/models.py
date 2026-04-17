"""核心 Pydantic 数据模型。

定义了多 Agent 协同系统中流转的全部数据结构，涵盖：
用户上下文、任务输入/输出、路由结果、子任务、工具规范、
Agent 执行结果、记忆条目。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from agent_hub.core.enums import IntentType, UserRole


# ── 用户上下文 ────────────────────────────────────────


class UserContext(BaseModel):
    """用户上下文：识别是谁、从哪个渠道来。

    Attributes:
        user_id: 用户唯一标识（由通讯渠道提供）。
        role: 用户角色，管理员或普通用户。
        channel: 消息来源渠道，如 ``"dingtalk"`` / ``"qq"`` / ``"openclaw"``。
        group_id: 群聊 ID；私聊时为 ``None``。
        is_private: 是否为私聊会话。
    """

    user_id: str
    role: UserRole
    channel: str
    session_id: Optional[str] = None
    group_id: Optional[str] = None
    is_private: bool = False


# ── 任务输入 ──────────────────────────────────────────


class TaskInput(BaseModel):
    """用户输入的原始请求。

    每条请求自动分配 ``trace_id``，用于全链路追踪。
    子任务通过 ``parent_trace_id`` 关联父任务。

    Attributes:
        trace_id: 全链路追踪 ID，自动生成 UUID。
        user_context: 发起请求的用户上下文。
        raw_message: 原始消息文本。
        attachments: 附件路径列表（文件 URL 或本地路径）。
        timestamp: 请求时间戳。
        parent_trace_id: 父任务 trace_id；顶层任务为 ``None``。
    """

    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_context: UserContext
    raw_message: str
    attachments: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    parent_trace_id: Optional[str] = None


# ── 子任务 & 路由结果 ────────────────────────────────


class SubTask(BaseModel):
    """路由层拆分出的子任务。

    支持 DAG 依赖编排：通过 ``depends_on`` 声明前置依赖，
    调度器据此决定并行/串行执行顺序。

    Attributes:
        subtask_id: 子任务唯一标识。
        description: 子任务自然语言描述。
        required_agents: 需要调用的 Agent 类型名称列表。
        priority: 优先级，1 为最高。
        depends_on: 依赖的子任务 ID 列表（用于 DAG 编排）。
    """

    subtask_id: str
    description: str
    required_agents: list[str]
    priority: int = 1
    depends_on: list[str] = Field(default_factory=list)


class RoutingResult(BaseModel):
    """路由层输出：意图分类 + 任务拆解。

    当 ``confidence < 0.7`` 时自动将 ``low_confidence`` 置为 ``True``，
    下游可据此触发人工确认流程。

    Attributes:
        intent: 识别出的用户意图类型。
        confidence: 分类置信度，范围 [0.0, 1.0]。
        reasoning: LLM 对分类决策的解释。
        sub_tasks: 拆解出的子任务列表。
        low_confidence: 置信度是否低于阈值。
    """

    intent: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    sub_tasks: list[SubTask] = Field(default_factory=list)
    low_confidence: bool = False


# ── ReAct 推理链路 ───────────────────────────────────


class ReActStep(BaseModel):
    """ReAct 推理循环中的单步记录。

    Attributes:
        thought: LLM 的推理过程。
        action: 要执行的工具名称；为 ``None`` 表示直接输出。
        action_input: 工具参数字符串。
        observation: 工具执行结果。
    """

    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None


class ReActTrace(BaseModel):
    """完整的 ReAct 推理链路。

    Attributes:
        steps: 推理步骤列表。
        final_answer: 最终输出；达到最大轮次时可能为 ``None``。
        total_rounds: 实际推理轮数。
    """

    steps: list[ReActStep] = Field(default_factory=list)
    final_answer: Optional[str] = None
    total_rounds: int = 0


# ── 安全守卫结果 ─────────────────────────────────────


class GuardResult(BaseModel):
    """Prompt 注入检测结果。

    Attributes:
        is_safe: 是否安全放行。
        risk_level: 风险等级：``"safe"`` / ``"suspicious"`` / ``"blocked"``。
        matched_rules: 命中的规则名称列表。
        llm_analysis: LLM 检测层的分析说明。
        confidence: 检测置信度。
    """

    is_safe: bool
    risk_level: str = "safe"
    matched_rules: list[str] = Field(default_factory=list)
    llm_analysis: Optional[str] = None
    confidence: float = 1.0


# ── 工具规范 ──────────────────────────────────────────


class ToolSpec(BaseModel):
    """可调用工具的规范描述。

    Attributes:
        name: 工具名称，如 ``"web_search"`` / ``"file_reader"``。
        description: 工具功能的自然语言描述。
        params_schema: 参数的 JSON Schema 定义。
        requires_admin: 是否需要管理员权限才能调用。
    """

    name: str
    description: str
    params_schema: dict[str, Any] = Field(default_factory=dict)
    requires_admin: bool = False


# ── Agent 执行结果 ───────────────────────────────────


class AgentResult(BaseModel):
    """单个 Agent 的执行结果。

    Attributes:
        agent_name: 执行该任务的 Agent 名称。
        success: 是否执行成功。
        output: Agent 输出内容（任意类型）。
        error: 失败时的错误信息。
        duration_ms: 执行耗时（毫秒）。
        trace_id: 关联的全链路追踪 ID。
    """

    agent_name: str
    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: int
    trace_id: str
    react_trace: Optional["ReActTrace"] = None


# ── 记忆条目 ─────────────────────────────────────────


class MemoryEntry(BaseModel):
    """记忆条目：会话态和持久态共用格式。

    Attributes:
        memory_id: 记忆唯一标识。
        content: 记忆内容文本。
        memory_type: 记忆类型，``"short_term"`` 或 ``"long_term"``。
        session_id: 所属会话 ID；长期记忆可为 ``None``。
        tags: 标签列表，用于检索和过滤。
        created_at: 创建时间。
        ttl: 存活时间（秒），``-1`` 表示永不过期。
        is_deleted: 软删除标记。
        embedding: 向量表示（可选，避免序列化开销）。
    """

    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    memory_type: str
    session_id: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    ttl: int = -1
    is_deleted: bool = False
    embedding: Optional[list[float]] = Field(default=None, exclude=True)


# ── 最终输出 ─────────────────────────────────────────


class TaskOutput(BaseModel):
    """Pipeline 执行完毕的完整结果。

    聚合所有参与 Agent 的执行记录和本次写入的记忆条目。

    Attributes:
        trace_id: 全链路追踪 ID。
        user_id: 请求发起者的用户 ID。
        response: 最终回复给用户的文本内容。
        agent_results: 各 Agent 的执行记录列表。
        memory_saved: 本次写入的记忆条目列表。
        total_duration_ms: 端到端总耗时（毫秒）。
        status: 执行状态：``"success"`` / ``"partial"`` / ``"failed"``。
    """

    trace_id: str
    user_id: str
    response: str
    agent_results: list[AgentResult] = Field(default_factory=list)
    memory_saved: list[MemoryEntry] = Field(default_factory=list)
    total_duration_ms: int
    status: str = "success"
