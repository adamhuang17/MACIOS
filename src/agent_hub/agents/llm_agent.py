"""LLM 生成 Agent。

调用大模型生成内容，支持双 Provider：OpenAI 兼容（主）+ Anthropic（降级）。
当注入了 ToolRegistry 且子任务涉及工具调用时，自动启用 ReAct 推理循环
（Thought → Action → Observation），支持最多 N 轮迭代与输出校验。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import structlog
from anthropic import Anthropic
from openai import OpenAI

from agent_hub.core.base_agent import BaseAgent
from agent_hub.core.models import AgentResult, ReActStep, ReActTrace, SubTask

if TYPE_CHECKING:
    from agent_hub.agents.registry import ToolRegistry
    from agent_hub.memory import MemoryManager

logger = structlog.get_logger(__name__)

# ── ReAct System Prompt 模板 ─────────────────────────

_REACT_SYSTEM_PROMPT = """\
你是一个具备工具调用能力的推理助手。请严格按照以下格式逐步思考并行动。

## 可用工具

{tools_description}

## 输出格式（必须严格遵守）

每一轮推理必须输出：

Thought: <你的推理过程，分析当前状态和下一步计划>
Action: <要调用的工具名称，必须从可用工具中选择>
Action Input: <工具参数>

当你已获得足够信息可以给出最终答案时，输出：

Thought: <总结推理过程>
Final Answer: <最终回答>

## 规则
1. 每次只能执行一个 Action
2. 必须等待 Observation（工具返回结果）后再进行下一轮思考
3. 最多进行 {max_rounds} 轮推理
4. 如果工具调用失败，在 Thought 中分析原因并尝试替代方案
5. 最终必须给出 Final Answer"""


class LLMAgent(BaseAgent):
    """调用大模型生成内容的 Agent。

    使用 OpenAI 兼容接口（支持智谱等国内 LLM 服务商），
    可选 Anthropic 作为降级 Provider。当注入了 ToolRegistry 且
    子任务涉及工具调用时，自动启用 ReAct 推理循环。

    Args:
        default_model: 默认模型名称。
        api_key: API 密钥（若为空则从环境变量读取）。
        base_url: API 基础 URL（若为空则使用 OpenAI 默认地址）。
        max_react_rounds: ReAct 最大推理轮数。
    """

    # 格式错误最大容忍次数
    _MAX_FORMAT_RETRIES = 2

    def __init__(
        self,
        default_model: str = "glm-4-flash",
        api_key: str | None = None,
        base_url: str | None = None,
        max_react_rounds: int = 5,
    ) -> None:
        super().__init__(name="llm_agent", description="调用大模型生成内容")
        self._default_model = default_model
        self._max_react_rounds = max_react_rounds

        # 主客户端：OpenAI 兼容接口（智谱等）
        client_kwargs: dict = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        self._primary = OpenAI(**client_kwargs)

        # 可选降级客户端：仅当 ANTHROPIC_API_KEY 存在时创建
        import os
        self._fallback: Anthropic | None = None
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            self._fallback = Anthropic(api_key=anthropic_key)

        self._memory_manager: MemoryManager | None = None
        self._registry: ToolRegistry | None = None

    def inject_memory(self, memory_manager: MemoryManager) -> None:
        """注入记忆管理器（Pipeline 在初始化时调用）。"""
        self._memory_manager = memory_manager

    def inject_registry(self, registry: ToolRegistry) -> None:
        """注入工具注册表，启用 ReAct 推理循环。"""
        self._registry = registry

    # ── 主入口 ───────────────────────────────────────

    async def run(
        self, subtask: SubTask, session_id: str, user_id: str,
    ) -> AgentResult:
        """执行 LLM 生成任务。

        当满足 ReAct 条件（有 registry 且子任务涉及 tool_agent）时
        走 ReAct 推理循环，否则走原有单次调用路径。
        """
        if self._should_use_react(subtask):
            return await self._run_react_loop(subtask, session_id, user_id)
        return await self._run_simple(subtask, session_id, user_id)

    def _should_use_react(self, subtask: SubTask) -> bool:
        """判断是否应启用 ReAct 推理循环。"""
        if self._registry is None:
            return False
        return "tool_agent" in subtask.required_agents

    # ── 原有单次调用路径 ─────────────────────────────

    async def _run_simple(
        self, subtask: SubTask, session_id: str, user_id: str,
    ) -> AgentResult:
        """单次 LLM 调用（非 ReAct 路径）。"""
        prompt = self._build_prompt(subtask, session_id, user_id)

        try:
            output = self._call_primary(prompt)
            logger.info(
                "llm_call_success",
                provider="openai_compatible",
                model=self._default_model,
                subtask_id=subtask.subtask_id,
            )
        except Exception as exc:
            logger.warning(
                "llm_primary_failed",
                provider="openai_compatible",
                error=str(exc),
                subtask_id=subtask.subtask_id,
            )
            if self._fallback is not None:
                try:
                    output = self._call_anthropic(prompt)
                    logger.info(
                        "llm_fallback_success",
                        provider="anthropic",
                        subtask_id=subtask.subtask_id,
                    )
                except Exception as fallback_exc:
                    logger.error(
                        "llm_all_providers_failed",
                        subtask_id=subtask.subtask_id,
                        error=str(fallback_exc),
                    )
                    return AgentResult(
                        agent_name=self.name,
                        success=False,
                        error=f"所有 LLM 提供商均不可用: {fallback_exc}",
                        duration_ms=0,
                        trace_id=subtask.subtask_id,
                    )
            else:
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    error=f"主 LLM 调用失败且无降级 Provider: {exc}",
                    duration_ms=0,
                    trace_id=subtask.subtask_id,
                )

        return AgentResult(
            agent_name=self.name,
            success=True,
            output=output,
            duration_ms=0,
            trace_id=subtask.subtask_id,
        )

    # ── ReAct 推理循环 ──────────────────────────────

    async def _run_react_loop(
        self, subtask: SubTask, session_id: str, user_id: str,
    ) -> AgentResult:
        """执行 ReAct 推理循环：Thought → Action → Observation。

        最多执行 ``_max_react_rounds`` 轮，每轮调用 LLM 获取
        Thought+Action 或 Final Answer，然后执行工具获取 Observation。
        """
        assert self._registry is not None  # noqa: S101

        # 构建 ReAct 系统 prompt
        system_prompt = self._build_react_system_prompt()

        # 构建初始用户消息（含记忆上下文）
        user_prompt = self._build_prompt(subtask, session_id, user_id)

        # 对话历史
        messages: list[dict[str, str]] = [
            {"role": "user", "content": user_prompt},
        ]

        steps: list[ReActStep] = []
        format_errors = 0

        for round_num in range(1, self._max_react_rounds + 1):
            # 调用 LLM
            try:
                llm_output = self._call_primary_with_system(system_prompt, messages)
            except Exception as exc:
                logger.warning("react_llm_call_failed", round=round_num, error=str(exc))
                if self._fallback is not None:
                    try:
                        llm_output = self._call_anthropic_with_system(
                            system_prompt, messages,
                        )
                    except Exception:
                        break
                else:
                    break

            # 解析 LLM 输出
            parsed = self._parse_react_output(llm_output)

            if parsed is None:
                format_errors += 1
                logger.warning(
                    "react_parse_failed",
                    round=round_num,
                    format_errors=format_errors,
                    output_preview=llm_output[:200],
                )
                if format_errors > self._MAX_FORMAT_RETRIES:
                    # 格式错误超限，将原始输出作为最终答案
                    return AgentResult(
                        agent_name=self.name,
                        success=True,
                        output=llm_output,
                        duration_ms=0,
                        trace_id=subtask.subtask_id,
                        react_trace=ReActTrace(
                            steps=steps,
                            final_answer=llm_output,
                            total_rounds=round_num,
                        ),
                    )
                # 追加格式修正提示
                messages.append({"role": "assistant", "content": llm_output})
                messages.append({
                    "role": "user",
                    "content": (
                        "输出格式不正确。请严格按照以下格式输出：\n"
                        "Thought: <推理>\nAction: <工具名>\nAction Input: <参数>\n"
                        "或者\nThought: <推理>\nFinal Answer: <最终答案>"
                    ),
                })
                continue

            thought = parsed["thought"]

            # Final Answer 路径
            if parsed.get("final_answer") is not None:
                final_answer = parsed["final_answer"]
                step = ReActStep(thought=thought)
                steps.append(step)

                # 输出校验
                if not self._validate_output(final_answer):
                    logger.warning("react_output_validation_failed", round=round_num)
                    messages.append({"role": "assistant", "content": llm_output})
                    messages.append({
                        "role": "user",
                        "content": "你的 Final Answer 为空或不完整，请重新总结并给出完整答案。",
                    })
                    continue

                logger.info(
                    "react_completed",
                    rounds=round_num,
                    total_steps=len(steps),
                )
                return AgentResult(
                    agent_name=self.name,
                    success=True,
                    output=final_answer,
                    duration_ms=0,
                    trace_id=subtask.subtask_id,
                    react_trace=ReActTrace(
                        steps=steps,
                        final_answer=final_answer,
                        total_rounds=round_num,
                    ),
                )

            # Action 路径：执行工具
            action = parsed["action"]
            action_input = parsed.get("action_input", "")
            observation = await self._execute_tool(action, action_input)

            step = ReActStep(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
            )
            steps.append(step)

            # 将本轮结果追加到对话历史
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}",
            })

        # 达到最大轮次，返回中间结果
        logger.warning(
            "react_max_rounds_reached",
            max_rounds=self._max_react_rounds,
            total_steps=len(steps),
        )
        best_output = self._extract_best_intermediate(steps)
        return AgentResult(
            agent_name=self.name,
            success=True,
            output=f"{best_output}\n\n⚠️ 已达最大推理轮次（{self._max_react_rounds}），以上为中间结果。",
            duration_ms=0,
            trace_id=subtask.subtask_id,
            react_trace=ReActTrace(
                steps=steps,
                final_answer=None,
                total_rounds=self._max_react_rounds,
            ),
        )

    # ── ReAct 输出解析 ──────────────────────────────

    @staticmethod
    def _parse_react_output(text: str) -> dict[str, Any] | None:
        """解析 LLM 的 ReAct 格式输出。

        兼容中英文关键字、全半角冒号等变体格式。

        Returns:
            成功时返回包含 thought / action / action_input 或
            thought / final_answer 的字典；解析失败返回 None。
        """
        if not text or not text.strip():
            return None

        # 兼容中英文、全半角冒号
        _SEP = r"[\s:：]+"

        # 尝试提取 Thought（兼容 "思考" / "Thought"）
        thought_match = re.search(
            rf"(?:Thought|思考){_SEP}(.+?)(?=\n(?:Action|操作|Final Answer|最终答案|FinalAnswer)[\s:：]|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if not thought_match:
            # 宽容模式：把整段非 Action/Final 文本当作 thought
            thought_match = re.search(r"^(.+?)(?=\n(?:Action|Final)[\s:：]|$)", text.strip(), re.DOTALL | re.IGNORECASE)
            if not thought_match:
                return None
        thought = thought_match.group(1).strip()

        # 检查 Final Answer（兼容 "最终答案" / "Final Answer" / "FinalAnswer"）
        final_match = re.search(
            rf"(?:Final\s*Answer|最终答案|FinalAnswer){_SEP}(.+)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if final_match:
            return {
                "thought": thought,
                "final_answer": final_match.group(1).strip(),
            }

        # 检查 Action + Action Input（兼容 "操作" / "Action"）
        action_match = re.search(
            rf"(?:Action|操作){_SEP}(.+?)(?:\n|$)",
            text,
            re.IGNORECASE,
        )
        if not action_match:
            return None
        action = action_match.group(1).strip()

        input_match = re.search(
            rf"(?:Action\s*Input|操作输入|ActionInput|参数){_SEP}(.+?)(?:\n|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        action_input = input_match.group(1).strip() if input_match else ""

        return {
            "thought": thought,
            "action": action,
            "action_input": action_input,
        }

    @staticmethod
    def _validate_output(output: str) -> bool:
        """校验 Final Answer 是否有效。"""
        if not output or not output.strip():
            return False
        # 过短的输出视为不完整
        if len(output.strip()) < 2:
            return False
        return True

    # ── 工具执行 ─────────────────────────────────────

    async def _execute_tool(self, tool_name: str, action_input: str) -> str:
        """通过 ToolRegistry 执行工具调用。"""
        assert self._registry is not None  # noqa: S101

        tool = self._registry.get(tool_name)
        if not tool:
            available = [t.name for t in self._registry.list_tools()]
            return f"错误：工具 '{tool_name}' 不存在。可用工具: {available}"

        # 构造参数：简单值 → {"value": ...}，key=value → 解析
        params = self._parse_action_input(action_input, tool_name)

        try:
            result = await tool.func(params)
            return str(result)
        except Exception as e:
            return f"工具执行错误：{e}"

    @staticmethod
    def _parse_action_input(action_input: str, tool_name: str) -> dict[str, Any]:
        """将 Action Input 字符串解析为工具参数字典。"""
        if not action_input:
            return {}

        # 尝试 key=value 格式
        if "=" in action_input and not action_input.strip().startswith("="):
            params: dict[str, Any] = {}
            for kv in action_input.split(","):
                kv = kv.strip()
                if "=" in kv:
                    key, value = kv.split("=", 1)
                    params[key.strip()] = value.strip()
            if params:
                return params

        # calculator 特殊处理
        if tool_name == "calculator":
            return {"expression": action_input}

        return {"value": action_input}

    @staticmethod
    def _extract_best_intermediate(steps: list[ReActStep]) -> str:
        """从中间步骤中提取最佳结果。"""
        if not steps:
            return "推理过程中未产生有效输出。"

        # 取最后一个有 observation 的步骤
        for step in reversed(steps):
            if step.observation:
                return step.observation

        # 取最后一个 thought
        return steps[-1].thought

    # ── ReAct System Prompt 构建 ─────────────────────

    def _build_react_system_prompt(self) -> str:
        """构建 ReAct 系统 Prompt，包含可用工具列表。"""
        assert self._registry is not None  # noqa: S101

        tools = self._registry.list_tools()
        tool_lines = []
        for t in tools:
            params_desc = ", ".join(
                f"{k}: {v.get('description', v.get('type', 'any'))}"
                for k, v in t.params_schema.items()
            ) if t.params_schema else "无参数"
            tool_lines.append(f"- **{t.name}**: {t.description}（参数: {params_desc}）")

        tools_description = "\n".join(tool_lines) if tool_lines else "（无可用工具）"

        return _REACT_SYSTEM_PROMPT.format(
            tools_description=tools_description,
            max_rounds=self._max_react_rounds,
        )

    # ── Prompt 构建 ──────────────────────────────────

    def _build_prompt(
        self, subtask: SubTask, session_id: str, user_id: str,
    ) -> str:
        """构建发送给 LLM 的完整 prompt。"""
        parts: list[str] = []

        # 如果有记忆上下文，加在最前面
        if self._memory_manager:
            ctx = self._memory_manager.get_context(session_id, user_id)
            if ctx:
                parts.append(f"【上下文参考】\n{ctx}\n")

        # 加任务描述
        parts.append(f"【当前任务】\n{subtask.description}")
        return "\n".join(parts)

    # ── LLM 调用方法 ────────────────────────────────

    def _call_primary(self, prompt: str) -> str:
        """调用主 LLM（OpenAI 兼容接口，如智谱）。"""
        response = self._primary.chat.completions.create(
            model=self._default_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    def _call_primary_with_system(
        self, system_prompt: str, messages: list[dict[str, str]],
    ) -> str:
        """调用主 LLM，附带系统 prompt（用于 ReAct）。"""
        response = self._primary.chat.completions.create(
            model=self._default_model,
            messages=[
                {"role": "system", "content": system_prompt},
                *messages,
            ],
        )
        return response.choices[0].message.content or ""

    def _call_anthropic(self, prompt: str) -> str:
        """降级调用 Anthropic Claude。"""
        if self._fallback is None:
            raise RuntimeError("Anthropic 降级客户端未配置")
        response = self._fallback.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _call_anthropic_with_system(
        self, system_prompt: str, messages: list[dict[str, str]],
    ) -> str:
        """降级调用 Anthropic Claude，附带系统 prompt（用于 ReAct）。"""
        if self._fallback is None:
            raise RuntimeError("Anthropic 降级客户端未配置")
        response = self._fallback.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text

    # ── 流式调用 ─────────────────────────────────────

    def _call_primary_stream(self, prompt: str):
        """流式调用主 LLM，返回 token 生成器。

        Yields:
            逐 token 的文本片段。
        """
        stream = self._primary.chat.completions.create(
            model=self._default_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def run_stream(
        self, subtask: SubTask, session_id: str, user_id: str,
    ):
        """流式执行 LLM 生成，逐 token yield。

        Yields:
            dict: ``{"type": "token", "content": "..."}``
        """
        prompt = self._build_prompt(subtask, session_id, user_id)
        try:
            for token in self._call_primary_stream(prompt):
                yield {"type": "token", "content": token}
        except Exception as exc:
            logger.warning("llm_stream_failed", error=str(exc))
            yield {"type": "error", "content": str(exc)}
