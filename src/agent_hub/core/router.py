"""意图识别路由器。

通过 OpenAI 兼容 API（智谱等）对用户自然语言消息进行
意图分类，并将复杂请求拆解为可执行的子任务 DAG。

核心流程::

    raw_message + UserContext
        → LLM Structured Output（function calling）
        → 权限过滤（ADMIN_COMMAND 降级）
        → 置信度规则（低置信度降级）
        → RoutingResult

Example::

    from agent_hub.core.router import IntentRouter
    from agent_hub.core.models import UserContext
    from agent_hub.core.enums import UserRole

    router = IntentRouter()
    result = await router.route(
        raw_message="帮我写一个快速排序",
        user_context=UserContext(
            user_id="u001", role=UserRole.USER, channel="dingtalk",
        ),
    )
    print(result.intent, result.confidence, result.sub_tasks)
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed

from agent_hub.config.settings import Settings, get_settings
from agent_hub.core.enums import IntentType, UserRole
from agent_hub.core.models import RoutingResult, SubTask, UserContext

logger = structlog.get_logger(__name__)

# ── 意图分类 System Prompt ───────────────────────────

_SYSTEM_PROMPT = """\
你是一个意图分类路由器。你的任务是分析用户消息，判断意图类型并拆解子任务。

## 意图分类规则（7 类）

1. **TASK_GENERATION** — 用户要求写代码、写报告、生成内容、创作文本。
   示例："帮我写一个快速排序算法"、"生成一份项目周报"、"写一篇关于AI的文章"

2. **RETRIEVAL** — 用户查询知识、问文档、搜索已有信息、问答类问题。
   示例："公司年假政策是什么"、"搜索关于LLM的论文"、"RAG是什么意思"

3. **TOOL_EXECUTION** — 用户要求执行特定操作：查天气、算数、调用外部API。
   示例："查一下北京今天天气"、"帮我算一下 123 * 456"、"发送一封邮件给张三"

4. **FILE_PROCESSING** — 用户上传、询问、处理文件相关内容。
   示例："帮我分析这个Excel文件"、"这个PDF里说了什么"、"把这个CSV转成JSON"

5. **ADMIN_COMMAND** — 管理员执行系统命令、控制虚拟机、运维操作。
   示例："重启测试服务器"、"查看系统日志"、"部署最新版本到生产环境"

6. **GROUP_CHAT** — 普通群聊消息、闲聊、不需要Agent处理的内容。
   示例："哈哈哈"、"今天天气真好"、"大家好"、"收到"

7. **REFLECTION** — 用户追问、质疑之前的结果、要求重新考虑。
   示例："你刚才的回答不对"、"重新想想"、"能不能换个方案"、"为什么这样做"

## 子任务拆解规则

当请求较复杂时，将其拆解为 2-4 个子任务。每个子任务包含：
- description: 子任务自然语言描述
- required_agents: 需要调用的Agent类型（如 "llm_agent", "retrieval_agent", "tool_agent", "file_agent"）
- priority: 优先级（1最高）
- depends_on: 依赖的子任务ID列表（用于DAG编排），如第二个子任务依赖第一个，则 depends_on=["subtask_1"]

**重要**：对于 tool_execution、task_generation、retrieval、file_processing、admin_command、reflection 意图，必须至少生成一个子任务。只有 group_chat 意图可以返回空列表。

## 用户上下文

- 用户角色: {role}
- 消息渠道: {channel}
- 是否私聊: {is_private}
- 群聊ID: {group_id}

请根据以上信息准确分类意图，给出置信度（0.0-1.0），并说明推理过程。"""


# ── LLM 输出的内部 Schema（仅用于 tool_use） ──────────


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
    """LLM 结构化输出的完整结果。"""

    intent: str = Field(
        description=(
            "意图类型，必须为以下之一: task_generation, retrieval, "
            "tool_execution, file_processing, admin_command, group_chat, reflection"
        ),
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="分类置信度，范围 [0.0, 1.0]",
    )
    reasoning: str = Field(description="对分类决策的解释说明")
    sub_tasks: list[_LLMSubTask] = Field(
        default_factory=list,
        description="拆解出的子任务列表，简单请求可为空",
    )


# ── 构造 OpenAI function calling tool 定义 ───────────

def _build_tool_schema() -> dict[str, Any]:
    """从 Pydantic schema 构造 OpenAI function calling tool 定义。"""
    return {
        "type": "function",
        "function": {
            "name": "classify_intent",
            "description": "对用户消息进行意图分类，输出分类结果和子任务拆解。",
            "parameters": _LLMRoutingOutput.model_json_schema(),
        },
    }


# ── 意图识别路由器 ───────────────────────────────────


class IntentRouter:
    """意图识别路由器。

    通过 OpenAI 兼容 API（智谱等）的 function calling 模式实现结构化输出，
    将用户自然语言消息分类为 7 种意图之一，并拆解为子任务 DAG。

    Args:
        settings: 配置对象；为 ``None`` 时使用全局单例。
        client: OpenAI 异步客户端；为 ``None`` 时自动创建。
        model: 模型名称；为 ``None`` 时使用 ``settings.llm_model``。

    Example::

        router = IntentRouter()
        result = await router.route("帮我写一个快速排序", user_context)
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
    ) -> RoutingResult:
        """对用户消息进行意图识别和子任务拆解。

        Args:
            raw_message: 原始消息文本。
            user_context: 用户上下文信息。

        Returns:
            RoutingResult: 包含意图分类、置信度、推理说明和子任务列表。
        """
        log = logger.bind(
            user_id=user_context.user_id,
            channel=user_context.channel,
            message_preview=raw_message[:50],
        )
        log.info("intent_router.route.start")

        # 1. 构造 prompt
        system_prompt = self._build_system_prompt(user_context)
        messages = [{"role": "user", "content": raw_message}]

        # 2. 调用 LLM（含重试，失败时降级）
        try:
            result = await self._call_llm(system_prompt, messages)
        except Exception:
            log.warning("intent_router.llm_failed", exc_info=True)
            return self._fallback_result("LLM调用失败，降级为群聊处理")

        # 3. 权限过滤
        result = self._apply_permission_filter(result, user_context)

        # 4. 置信度规则
        result = self._apply_confidence_rules(result)

        log.info(
            "intent_router.route.done",
            intent=result.intent.value,
            confidence=result.confidence,
            low_confidence=result.low_confidence,
            sub_task_count=len(result.sub_tasks),
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
    ) -> RoutingResult:
        """调用 Anthropic Claude API 并解析结构化输出。

        使用 tool_use + tool_choice 强制模型输出结构化 JSON，
        然后解析为 RoutingResult。

        Raises:
            Exception: LLM 调用失败或响应解析失败时抛出，
                       由 tenacity 重试或上层 route() 捕获降级。
        """
        log = logger.bind(model=self._model)
        log.debug("intent_router.llm_call.start")

        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                *messages,  # type: ignore[arg-type]
            ],
            tools=[self._tool_schema],  # type: ignore[list-item]
            tool_choice={"type": "function", "function": {"name": "classify_intent"}},  # type: ignore[arg-type]
            timeout=httpx.Timeout(10.0),
        )

        # 从响应中提取 function call
        tool_input = self._extract_tool_input(response)

        # 解析 LLM 原始输出
        llm_output = _LLMRoutingOutput.model_validate(tool_input)

        # 转换为公共模型 RoutingResult
        result = self._to_routing_result(llm_output)

        log.debug(
            "intent_router.llm_call.done",
            intent=result.intent.value,
            confidence=result.confidence,
        )
        return result

    @staticmethod
    def _extract_tool_input(response: Any) -> dict[str, Any]:
        """从 OpenAI 兼容响应中提取 function call 的 arguments。"""
        import json

        message = response.choices[0].message
        if not message.tool_calls:
            msg = "LLM 响应中未找到 tool_calls"
            raise ValueError(msg)
        return json.loads(message.tool_calls[0].function.arguments)

    @staticmethod
    def _to_routing_result(llm_output: _LLMRoutingOutput) -> RoutingResult:
        """将 LLM 内部输出转换为公共 RoutingResult 模型。"""
        # 解析意图枚举
        intent = IntentType(llm_output.intent)

        # 转换子任务，由路由器统一编号 subtask_id
        sub_tasks = [
            SubTask(
                subtask_id=f"subtask_{i + 1}",
                description=st.description,
                required_agents=st.required_agents,
                priority=st.priority,
                depends_on=st.depends_on,
            )
            for i, st in enumerate(llm_output.sub_tasks)
        ]

        return RoutingResult(
            intent=intent,
            confidence=llm_output.confidence,
            reasoning=llm_output.reasoning,
            sub_tasks=sub_tasks,
        )

    # ── 权限过滤 ────────────────────────────────────

    @staticmethod
    def _apply_permission_filter(
        result: RoutingResult,
        user_context: UserContext,
    ) -> RoutingResult:
        """检查 ADMIN_COMMAND 权限，非管理员降级为 RETRIEVAL。"""
        if (
            result.intent == IntentType.ADMIN_COMMAND
            and user_context.role != UserRole.ADMIN
        ):
            logger.warning(
                "intent_router.permission_denied",
                user_id=user_context.user_id,
                role=user_context.role.value,
                original_intent=result.intent.value,
            )
            return result.model_copy(update={
                "intent": IntentType.RETRIEVAL,
                "reasoning": f"{result.reasoning} [权限不足，降级处理]",
            })
        return result

    # ── 置信度规则 ───────────────────────────────────

    @staticmethod
    def _apply_confidence_rules(result: RoutingResult) -> RoutingResult:
        """根据置信度阈值应用降级规则。

        - confidence < 0.4 → 强制降级为 GROUP_CHAT
        - confidence < 0.7 → 标记 low_confidence，清空 sub_tasks
        """
        updates: dict[str, Any] = {}

        if result.confidence < 0.4:
            # 极低置信度：强制降级为群聊
            logger.warning(
                "intent_router.very_low_confidence",
                confidence=result.confidence,
                original_intent=result.intent.value,
            )
            updates["intent"] = IntentType.GROUP_CHAT
            updates["reasoning"] = f"{result.reasoning} [置信度过低，降级为群聊]"
            updates["low_confidence"] = True
            updates["sub_tasks"] = []
        elif result.confidence < 0.7:
            # 低置信度：标记并清空子任务
            logger.info(
                "intent_router.low_confidence",
                confidence=result.confidence,
            )
            updates["low_confidence"] = True
            updates["sub_tasks"] = []

        if updates:
            return result.model_copy(update=updates)
        return result

    # ── 降级兜底 ────────────────────────────────────

    @staticmethod
    def _fallback_result(reason: str) -> RoutingResult:
        """LLM 调用失败时的兜底结果。"""
        return RoutingResult(
            intent=IntentType.GROUP_CHAT,
            confidence=0.3,
            reasoning=reason,
            sub_tasks=[],
            low_confidence=True,
        )
