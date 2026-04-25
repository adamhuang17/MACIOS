"""路由决策器（DecisionRouter）。

通过 OpenAI 兼容 API（智谱等）对用户请求进行路由决策：
输出执行模式（mode）、能力标签（capabilities）、子任务 DAG（plan），
替代原来的全局 IntentType 语义分类。

核心流程::

    raw_message + UserContext
        → LLM Structured Output（function calling）
        → 置信度规则（低置信度降级为 ignore）
        → RoutingDecision

Example::

    from agent_hub.core.router import DecisionRouter
    from agent_hub.core.models import UserContext
    from agent_hub.core.enums import UserRole

    router = DecisionRouter()
    decision = await router.route(
        raw_message="帮我写一个快速排序",
        user_context=UserContext(
            user_id="u001", role=UserRole.USER, channel="dingtalk",
        ),
    )
    print(decision.mode, decision.confidence, decision.plan)
"""

from __future__ import annotations

from typing import Any, Protocol, cast

import httpx
import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed

from agent_hub.config.settings import Settings, get_settings
from agent_hub.core.enums import ExecutionMode
from agent_hub.core.models import RoutingDecision, SubTask, UserContext

logger = structlog.get_logger(__name__)

# ── 路由决策 System Prompt ──────────────────────────

_SYSTEM_PROMPT = """\
你是一个智能请求路由器。你的任务是分析用户消息和上下文，决定如何处理这条请求。
## 执行模式（6 选 1，mode 字段）
1. **ignore** — 忽略或轻量回复。普通群聊闲聊、不需要处理的消息。
   示例："哈哈哈"、"大家好"、"今天天气真好"、"收到"
2. **qa** — 知识问答 / 检索。用户查询知识、问文档、问答类问题。
   示例："公司年假政策是什么"、"RAG是什么意思"、"搜索关于LLM的论文"
3. **plan** — 内容生成 / 复杂多步骤任务。用户要求写代码、写报告、生成内容。
   示例："帮我写一个快速排序算法"、"生成一份项目周报"、"写一篇关于AI的文章"
4. **act** — 工具执行 / 文件处理。用户要求执行具体操作或处理文件。
   示例："帮我算一下 123*456"、"查一下北京今天天气"、"帮我分析这个Excel文件"
5. **delegate** — 委派给外部 Agent（如 OpenClaw）。需要管理员权限的运维操作。
   示例："重启测试服务器"、"查看系统日志"、"部署最新版本到生产环境"
6. **repair** — 反思修复。用户质疑或要求重新考虑之前的结果。
   示例："你刚才的回答不对"、"重新想想"、"能不能换个方案"、"为什么这样做"
## 能力标签（capabilities，可多选）
从以下标签中选择本次请求所需的能力：
- retrieval：需要检索知识库
- tool：需要调用工具（计算、API调用等）
- file_ingest：需要处理上传的文件
- openclaw：需要调用 OpenClaw 外部系统
- memory_write：需要写入长期记忆
## 约束字段
- requires_admin：true/false，是否必须管理员才能执行（delegate 模式通常为 true）
- private_only：true/false，是否只能在私聊中触发
- allow_in_group：true/false，群聊中是否允许触发
## 子任务拆解规则
当 mode 为 qa/plan/act/delegate/repair 时，将请求拆解为 1-4 个子任务。每个子任务包含：
- description: 子任务自然语言描述
- required_agents: 需要调用的Agent类型列表
    （llm_agent / retrieval_agent / tool_agent / reflection_agent）
- priority: 优先级（1最高）
- depends_on: 依赖的子任务ID列表（用于DAG编排）
**重要**：mode=ignore 时 plan 为空列表；其他 mode 必须至少生成一个子任务。
## 用户上下文
- 用户角色: {role}
- 消息渠道: {channel}
- 是否私聊: {is_private}
- 群聊ID: {group_id}
请根据以上信息输出路由决策，给出置信度（0.0-1.0）并说明推理过程。"""

# ── LLM 输出的内部 Schema（仅用于 function calling） ──────────

class _LLMSubTask(BaseModel):
    """LLM 输出的子任务结构。"""

    description: str = Field(description="子任务自然语言描述")
    required_agents: list[str] = Field(
        description='需要调用的Agent类型名称列表，如 ["llm_agent", "retrieval_agent"]',
    )
    priority: int = Field(default=1, description="优先级，1为最高")
    depends_on: list[str] = Field(
        default_factory=list,
        description='依赖的子任务ID列表，如 ["subtask_1"]',
    )


class _LLMRoutingOutput(BaseModel):
    """LLM 结构化输出的完整路由决策。"""
    mode: str = Field(
        description="执行模式，必须为以下之一: ignore, qa, plan, act, delegate, repair",
    )
    targets: list[str] = Field(
        default_factory=list,
        description="目标 flow 或 agent 名称列表（可为空）",
    )
    requires_admin: bool = Field(
        default=False,
        description="是否必须管理员才能执行，delegate 模式通常为 true",
    )
    private_only: bool = Field(
        default=False,
        description="是否只能在私聊中触发",
    )
    allow_in_group: bool = Field(
        default=True,
        description="群聊中是否允许触发",
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description="所需能力标签：retrieval, tool, file_ingest, openclaw, memory_write",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="路由决策置信度，范围 [0.0, 1.0]",
    )
    reasoning: str = Field(description="对路由决策的解释说明")
    plan: list[_LLMSubTask] = Field(
        default_factory=list,
        description="拆解出的子任务列表，mode=ignore 时为空列表",
    )


# ── OpenAI 响应协议（最小字段） ───────────────────────


class _ToolCallFunction(Protocol):
    arguments: str


class _ToolCall(Protocol):
    function: _ToolCallFunction


class _ToolCallMessage(Protocol):
    tool_calls: list[_ToolCall] | None


class _ToolCallChoice(Protocol):
    message: _ToolCallMessage


class _ToolCallResponse(Protocol):
    choices: list[_ToolCallChoice]


# ── 构造 OpenAI function calling tool 定义 ───────────

def _build_tool_schema() -> dict[str, Any]:
    """从 Pydantic schema 构造 OpenAI function calling tool 定义。"""
    return {
        "type": "function",
        "function": {
            "name": "make_routing_decision",
            "description": "对用户消息进行路由决策，输出执行模式和子任务计划。",
            "parameters": _LLMRoutingOutput.model_json_schema(),
        },
    }


# ── 路由决策器 ───────────────────────────────────────


class DecisionRouter:
    """路由决策器。

    通过 OpenAI 兼容 API（智谱等）的 function calling 模式实现结构化输出，
    将用户请求路由到对应执行模式（mode），并拆解为子任务 DAG（plan）。

    不做全局语义意图分类，只决策"下一步如何执行"。
    权限校验由 Pipeline 的 Policy Gate 负责，不在本层执行。

    Args:
        settings: 配置对象；为 ``None`` 时使用全局单例。
        client: OpenAI 异步客户端；为 ``None`` 时自动创建。
        model: 模型名称；为 ``None`` 时使用 ``settings.llm_model``。

    Example::

        router = DecisionRouter()
        decision = await router.route("帮我写一个快速排序", user_context)
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client or AsyncOpenAI(
            api_key=self._settings.llm_api_key,
            base_url=self._settings.llm_base_url,
        )
        self._model = model or self._settings.llm_model
        self._tool_schema = _build_tool_schema()

    # ── 公共入口 ─────────────────────────────────────

    async def route(
        self,
        raw_message: str,
        user_context: UserContext,
    ) -> RoutingDecision:
        """对用户请求进行路由决策和子任务拆解。

        Args:
            raw_message: 原始消息文本。
            user_context: 用户上下文信息。

        Returns:
            RoutingDecision: 包含执行模式、能力标签、置信度和子任务计划。
        """
        log = logger.bind(
            user_id=user_context.user_id,
            channel=user_context.channel,
            message_preview=raw_message[:50],
        )
        log.info("decision_router.route.start")

        # 1. 构造 prompt
        system_prompt = self._build_system_prompt(user_context)
        messages = [{"role": "user", "content": raw_message}]

        # 2. 调用 LLM（含重试，失败时降级）
        try:
            result = await self._call_llm(system_prompt, messages)
        except Exception:
            log.warning("decision_router.llm_failed", exc_info=True)
            return self._fallback_result("LLM调用失败，降级为忽略处理")

        # 3. 置信度规则
        result = self._apply_confidence_rules(result)

        log.info(
            "decision_router.route.done",
            mode=result.mode.value,
            confidence=result.confidence,
            low_confidence=result.low_confidence,
            plan_count=len(result.plan),
        )
        return result

    # ── Prompt 构造 ──────────────────────────────────

    def _build_system_prompt(self, user_context: UserContext) -> str:
        """将用户上下文填充到 System Prompt 模板中。"""
        return _SYSTEM_PROMPT.format(
            role=user_context.role.value,
            channel=user_context.channel,
            is_private=user_context.is_private,
            group_id=user_context.group_id or "无",
        )

    # ── LLM 调用 ────────────────────────────────────

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(1), reraise=True)
    async def _call_llm(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> RoutingDecision:
        """调用 LLM API 并解析结构化路由决策输出。

        使用 function calling + tool_choice 强制模型输出结构化 JSON，
        然后解析为 RoutingDecision。

        Raises:
            Exception: LLM 调用失败或响应解析失败时抛出，
                       由 tenacity 重试或上层 route() 捕获降级。
        """
        log = logger.bind(model=self._model)
        log.debug("decision_router.llm_call.start")

        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                *messages,  # type: ignore[arg-type]
            ],
            tools=[self._tool_schema],  # type: ignore[list-item]
            tool_choice={"type": "function", "function": {"name": "make_routing_decision"}},  # type: ignore[arg-type]
            timeout=httpx.Timeout(10.0),
        )

        # 从响应中提取 function call
        tool_input = self._extract_tool_input(response)

        # 解析 LLM 原始输出
        llm_output = _LLMRoutingOutput.model_validate(tool_input)

        # 转换为公共模型 RoutingDecision
        result = self._to_routing_decision(llm_output)

        log.debug(
            "decision_router.llm_call.done",
            mode=result.mode.value,
            confidence=result.confidence,
        )
        return result

    @staticmethod
    def _extract_tool_input(response: _ToolCallResponse) -> dict[str, object]:
        """从 OpenAI 兼容响应中提取 function call 的 arguments。"""
        import json

        message = response.choices[0].message
        if not message.tool_calls:
            msg = "LLM 响应中未找到 tool_calls"
            raise ValueError(msg)
        return cast(dict[str, object], json.loads(message.tool_calls[0].function.arguments))

    @staticmethod
    def _to_routing_decision(llm_output: _LLMRoutingOutput) -> RoutingDecision:
        """将 LLM 内部输出转换为公共 RoutingDecision 模型。"""
        mode = ExecutionMode(llm_output.mode)

        # 转换子任务，由路由器统一编号 subtask_id
        plan = [
            SubTask(
                subtask_id=f"subtask_{i + 1}",
                description=st.description,
                required_agents=st.required_agents,
                priority=st.priority,
                depends_on=st.depends_on,
            )
            for i, st in enumerate(llm_output.plan)
        ]

        return RoutingDecision(
            mode=mode,
            targets=llm_output.targets,
            requires_admin=llm_output.requires_admin,
            private_only=llm_output.private_only,
            allow_in_group=llm_output.allow_in_group,
            capabilities=llm_output.capabilities,
            confidence=llm_output.confidence,
            reasoning=llm_output.reasoning,
            plan=plan,
        )

    # ── 置信度规则 ───────────────────────────────────

    @staticmethod
    def _apply_confidence_rules(result: RoutingDecision) -> RoutingDecision:
        """根据置信度阈值应用降级规则。

        - confidence < 0.4 → 强制降级为 ignore 模式
        - confidence < 0.7 → 标记 low_confidence，清空 plan
        """
        updates: dict[str, Any] = {}

        if result.confidence < 0.4:
            # 极低置信度：强制降级为忽略
            logger.warning(
                "decision_router.very_low_confidence",
                confidence=result.confidence,
                original_mode=result.mode.value,
            )
            updates["mode"] = ExecutionMode.IGNORE
            updates["reasoning"] = (
                f"{result.reasoning} [置信度过低 ({result.confidence:.2f})，降级为忽略]"
            )
            updates["low_confidence"] = True
            updates["plan"] = []
        elif result.confidence < 0.7:
            # 低置信度：标记并清空计划
            logger.info(
                "decision_router.low_confidence",
                confidence=result.confidence,
            )
            updates["low_confidence"] = True
            updates["plan"] = []

        if updates:
            return result.model_copy(update=updates)
        return result

    # ── 降级兜底 ────────────────────────────────────

    @staticmethod
    def _fallback_result(reason: str) -> RoutingDecision:
        """LLM 调用失败时的兜底结果。"""
        return RoutingDecision(
            mode=ExecutionMode.IGNORE,
            confidence=0.3,
            reasoning=reason,
            plan=[],
            low_confidence=True,
        )


# ── 向后兼容别名 ─────────────────────────────────────

#: 过渡期别名，后续迭代移除。
IntentRouter = DecisionRouter
