"""Agent 执行层单元测试。

覆盖：
- ToolRegistry 注册 / 查询 / 权限过滤
- 预置工具安全性（calculator ast 安全解析）
- ToolAgent 正则解析 / 权限检查 / 执行
- LLMAgent prompt 构建 / 双 Provider 降级
- RetrievalAgent 检索成功 / RAG 不可用降级
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_hub.agents.llm_agent import LLMAgent
from agent_hub.agents.registry import (
    ToolRegistry,
    _safe_eval_expr,
    calculator_impl,
)
from agent_hub.agents.retrieval_agent import RetrievalAgent
from agent_hub.agents.tool_agent import ToolAgent
from agent_hub.core.models import AgentResult, MemoryEntry, ReActStep, ReActTrace, SubTask
from agent_hub.memory import MemoryManager


# ── Fixtures ─────────────────────────────────────────


@pytest.fixture()
def registry() -> ToolRegistry:
    """带预置工具的注册表。"""
    r = ToolRegistry()
    r.register_defaults()
    return r


@pytest.fixture()
def subtask() -> SubTask:
    return SubTask(
        subtask_id="st-test-001",
        description="calculator(2+3*4)",
        required_agents=["tool_agent"],
    )


# ══════════════════════════════════════════════════════
# ToolRegistry 测试
# ══════════════════════════════════════════════════════


class TestToolRegistry:
    def test_register_and_get(self) -> None:
        r = ToolRegistry()

        async def dummy(params: dict[str, Any]) -> str:
            return "ok"

        r.register("test_tool", "测试工具", {"x": {"type": "string"}}, dummy)
        tool = r.get("test_tool")
        assert tool is not None
        assert tool.spec.name == "test_tool"
        assert tool.spec.description == "测试工具"
        assert tool.func is dummy

    def test_get_nonexistent(self) -> None:
        r = ToolRegistry()
        assert r.get("nonexistent") is None

    def test_list_tools(self, registry: ToolRegistry) -> None:
        tools = registry.list_tools()
        names = {t.name for t in tools}
        assert "calculator" in names
        assert "file_read" in names
        assert "file_write" in names
        assert "system_command" in names
        assert "get_current_time" in names
        assert "search_web" in names
        assert len(tools) == 6

    def test_list_tools_for_user(self, registry: ToolRegistry) -> None:
        user_tools = registry.list_tools_for_user("user")
        admin_tools = registry.list_tools_for_user("admin")

        user_names = {t.name for t in user_tools}
        admin_names = {t.name for t in admin_tools}

        # 普通用户不能看到 admin 专属工具
        assert "file_write" not in user_names
        assert "system_command" not in user_names

        # admin 可以看到所有工具
        assert "file_write" in admin_names
        assert "system_command" in admin_names
        assert "calculator" in admin_names

    def test_list_tools_for_admin_superset(self, registry: ToolRegistry) -> None:
        user_tools = {t.name for t in registry.list_tools_for_user("user")}
        admin_tools = {t.name for t in registry.list_tools_for_user("admin")}
        assert user_tools.issubset(admin_tools)


# ══════════════════════════════════════════════════════
# Calculator 安全性测试
# ══════════════════════════════════════════════════════


class TestCalculatorSafety:
    def test_basic_arithmetic(self) -> None:
        assert _safe_eval_expr("2 + 3") == 5
        assert _safe_eval_expr("10 - 4") == 6
        assert _safe_eval_expr("3 * 7") == 21
        assert _safe_eval_expr("15 / 3") == 5.0

    def test_complex_expression(self) -> None:
        assert _safe_eval_expr("2 + 3 * 4") == 14
        assert _safe_eval_expr("(2 + 3) * 4") == 20

    def test_power(self) -> None:
        assert _safe_eval_expr("2 ** 10") == 1024

    def test_negative(self) -> None:
        assert _safe_eval_expr("-5 + 3") == -2

    def test_reject_function_call(self) -> None:
        with pytest.raises(ValueError, match="不允许的表达式节点"):
            _safe_eval_expr("__import__('os').system('rm -rf /')")

    def test_reject_attribute_access(self) -> None:
        with pytest.raises(ValueError, match="不允许的表达式节点"):
            _safe_eval_expr("().__class__.__bases__")

    def test_reject_name_reference(self) -> None:
        with pytest.raises(ValueError, match="不允许的表达式节点"):
            _safe_eval_expr("open")

    def test_reject_large_power(self) -> None:
        with pytest.raises(ValueError, match="幂指数过大"):
            _safe_eval_expr("2 ** 100000")

    def test_division_by_zero(self) -> None:
        with pytest.raises(ZeroDivisionError):
            _safe_eval_expr("1 / 0")

    @pytest.mark.asyncio
    async def test_calculator_impl_success(self) -> None:
        result = await calculator_impl({"expression": "2 + 3 * 4"})
        assert "14" in result
        assert "计算结果" in result

    @pytest.mark.asyncio
    async def test_calculator_impl_error(self) -> None:
        result = await calculator_impl({"expression": "import os"})
        assert "计算错误" in result


# ══════════════════════════════════════════════════════
# ToolAgent 测试
# ══════════════════════════════════════════════════════


class TestToolAgent:
    def test_parse_tool_call_simple(self, registry: ToolRegistry) -> None:
        agent = ToolAgent(registry)
        result = agent._parse_tool_call("calculator(2+3*4)")
        assert result is not None
        name, params = result
        assert name == "calculator"
        assert params == {"value": "2+3*4"}

    def test_parse_tool_call_kv(self, registry: ToolRegistry) -> None:
        agent = ToolAgent(registry)
        result = agent._parse_tool_call("file_read(path=/tmp/a.txt)")
        assert result is not None
        name, params = result
        assert name == "file_read"
        assert params == {"path": "/tmp/a.txt"}

    def test_parse_tool_call_empty_args(self, registry: ToolRegistry) -> None:
        agent = ToolAgent(registry)
        result = agent._parse_tool_call("get_current_time()")
        assert result is not None
        name, params = result
        assert name == "get_current_time"
        assert params == {}

    def test_parse_tool_call_failure(self, registry: ToolRegistry) -> None:
        agent = ToolAgent(registry)
        assert agent._parse_tool_call("这是一句普通话") is None

    def test_check_permission_user_on_admin_tool(
        self, registry: ToolRegistry,
    ) -> None:
        agent = ToolAgent(registry)
        assert agent._check_permission("file_write", "user") is False
        assert agent._check_permission("system_command", "user") is False

    def test_check_permission_admin_on_admin_tool(
        self, registry: ToolRegistry,
    ) -> None:
        agent = ToolAgent(registry)
        assert agent._check_permission("file_write", "admin") is True
        assert agent._check_permission("system_command", "admin") is True

    def test_check_permission_user_on_normal_tool(
        self, registry: ToolRegistry,
    ) -> None:
        agent = ToolAgent(registry)
        assert agent._check_permission("calculator", "user") is True
        assert agent._check_permission("get_current_time", "user") is True

    def test_check_permission_nonexistent(self, registry: ToolRegistry) -> None:
        agent = ToolAgent(registry)
        assert agent._check_permission("nonexistent", "admin") is False

    @pytest.mark.asyncio
    async def test_run_calculator(self, registry: ToolRegistry) -> None:
        agent = ToolAgent(registry)
        st = SubTask(
            subtask_id="st-1",
            description="calculator(2+3)",
            required_agents=["tool_agent"],
        )
        result = await agent.run(st, "sess-001", "u001")
        assert result.success is True
        assert "5" in str(result.output)

    @pytest.mark.asyncio
    async def test_run_permission_denied(self, registry: ToolRegistry) -> None:
        agent = ToolAgent(registry)
        agent.set_user_role("user")
        st = SubTask(
            subtask_id="st-2",
            description="file_write(path=a.txt,content=hello)",
            required_agents=["tool_agent"],
        )
        result = await agent.run(st, "sess-001", "u001")
        assert result.success is False
        assert "权限不足" in str(result.error)

    @pytest.mark.asyncio
    async def test_run_admin_allowed(self, registry: ToolRegistry) -> None:
        agent = ToolAgent(registry)
        agent.set_user_role("admin")
        st = SubTask(
            subtask_id="st-3",
            description="get_current_time()",
            required_agents=["tool_agent"],
        )
        result = await agent.run(st, "sess-001", "u001")
        assert result.success is True
        assert "当前时间" in str(result.output)

    @pytest.mark.asyncio
    async def test_run_tool_not_found(self, registry: ToolRegistry) -> None:
        agent = ToolAgent(registry)
        st = SubTask(
            subtask_id="st-4",
            description="nonexistent_tool(abc)",
            required_agents=["tool_agent"],
        )
        result = await agent.run(st, "sess-001", "u001")
        assert result.success is False
        assert "不存在" in str(result.error)

    @pytest.mark.asyncio
    async def test_run_parse_failed(self, registry: ToolRegistry) -> None:
        agent = ToolAgent(registry)
        st = SubTask(
            subtask_id="st-5",
            description="这只是一段普通文字",
            required_agents=["tool_agent"],
        )
        result = await agent.run(st, "sess-001", "u001")
        assert result.success is False
        assert "无法从描述中解析" in str(result.error)


# ══════════════════════════════════════════════════════
# LLMAgent 测试
# ══════════════════════════════════════════════════════


class TestLLMAgent:
    @pytest.fixture()
    def mock_llm_agent(self) -> LLMAgent:
        """创建 LLMAgent，mock 掉 SDK 客户端。"""
        with (
            patch("agent_hub.agents.llm_agent.Anthropic") as mock_anth,
            patch("agent_hub.agents.llm_agent.OpenAI") as mock_oai,
        ):
            agent = LLMAgent()
            agent._primary = mock_oai.return_value
            agent._fallback = mock_anth.return_value
            yield agent

    def test_build_prompt_no_memory(self, mock_llm_agent: LLMAgent) -> None:
        st = SubTask(
            subtask_id="st-1",
            description="写一首关于春天的诗",
            required_agents=["llm_agent"],
        )
        prompt = mock_llm_agent._build_prompt(st, "sess-001", "u001")
        assert "【当前任务】" in prompt
        assert "写一首关于春天的诗" in prompt
        assert "【上下文参考】" not in prompt

    def test_build_prompt_with_memory(self, mock_llm_agent: LLMAgent) -> None:
        mm = MagicMock(spec=MemoryManager)
        mm.get_context.return_value = "用户之前提到喜欢李白的诗风"
        mock_llm_agent.inject_memory(mm)

        st = SubTask(
            subtask_id="st-1",
            description="写一首关于春天的诗",
            required_agents=["llm_agent"],
        )
        prompt = mock_llm_agent._build_prompt(st, "sess-001", "u001")
        assert "【上下文参考】" in prompt
        assert "李白" in prompt
        assert "【当前任务】" in prompt

    def test_build_prompt_memory_empty_context(
        self, mock_llm_agent: LLMAgent,
    ) -> None:
        mm = MagicMock(spec=MemoryManager)
        mm.get_context.return_value = ""
        mock_llm_agent.inject_memory(mm)

        st = SubTask(
            subtask_id="st-1",
            description="测试",
            required_agents=["llm_agent"],
        )
        prompt = mock_llm_agent._build_prompt(st, "sess-001", "u001")
        assert "【上下文参考】" not in prompt

    @pytest.mark.asyncio
    async def test_run_primary_success(self, mock_llm_agent: LLMAgent) -> None:
        # Mock OpenAI 兼容接口成功
        mock_msg = MagicMock()
        mock_msg.message.content = "这是主 LLM 的回复"
        mock_response = MagicMock()
        mock_response.choices = [mock_msg]
        mock_llm_agent._primary.chat.completions.create.return_value = mock_response

        st = SubTask(
            subtask_id="st-1",
            description="你好",
            required_agents=["llm_agent"],
        )
        result = await mock_llm_agent.run(st, "sess-001", "u001")
        assert result.success is True
        assert result.output == "这是主 LLM 的回复"
        mock_llm_agent._primary.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_fallback_to_anthropic(self, mock_llm_agent: LLMAgent) -> None:
        # Mock 主 LLM 失败
        mock_llm_agent._primary.chat.completions.create.side_effect = Exception("API 错误")

        # Mock Anthropic 降级成功
        mock_content = MagicMock()
        mock_content.text = "这是 Claude 的回复"
        mock_anth_resp = MagicMock()
        mock_anth_resp.content = [mock_content]
        mock_llm_agent._fallback.messages.create.return_value = mock_anth_resp

        st = SubTask(
            subtask_id="st-1",
            description="你好",
            required_agents=["llm_agent"],
        )
        result = await mock_llm_agent.run(st, "sess-001", "u001")
        assert result.success is True
        assert result.output == "这是 Claude 的回复"
        mock_llm_agent._fallback.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_all_providers_fail(self, mock_llm_agent: LLMAgent) -> None:
        mock_llm_agent._primary.chat.completions.create.side_effect = Exception(
            "主 LLM 挂了",
        )
        mock_llm_agent._fallback.messages.create.side_effect = Exception(
            "Anthropic 也挂了",
        )

        st = SubTask(
            subtask_id="st-1",
            description="你好",
            required_agents=["llm_agent"],
        )
        result = await mock_llm_agent.run(st, "sess-001", "u001")
        assert result.success is False
        assert "所有 LLM 提供商均不可用" in str(result.error)


# ══════════════════════════════════════════════════════
# RetrievalAgent 测试
# ══════════════════════════════════════════════════════


class TestRetrievalAgent:
    @pytest.fixture()
    def mock_retrieval_agent(self) -> RetrievalAgent:
        mock_rag = MagicMock()
        mock_rag.retrieve = AsyncMock()
        agent = RetrievalAgent(rag_pipeline=mock_rag, top_k=5)
        return agent

    @pytest.mark.asyncio
    async def test_run_success(self, mock_retrieval_agent: RetrievalAgent) -> None:
        from agent_hub.rag.vector_store import SearchResult

        mock_retrieval_agent._rag.retrieve.return_value = [
            SearchResult(
                chunk_id="c1", content="Python 是一门编程语言",
                score=0.9, metadata={"source": "wiki/python.md"},
            ),
            SearchResult(
                chunk_id="c2", content="Python 3.12 新特性",
                score=0.8, metadata={"source": "docs/changelog.md"},
            ),
        ]
        mock_retrieval_agent._rag.format_context.return_value = (
            "[1] Python 是一门编程语言\n[2] Python 3.12 新特性"
        )

        st = SubTask(
            subtask_id="st-1",
            description="Python 是什么？",
            required_agents=["retrieval_agent"],
        )
        result = await mock_retrieval_agent.run(st, "sess-001", "u001")
        assert result.success is True
        assert "Python 是一门编程语言" in str(result.output)

    @pytest.mark.asyncio
    async def test_run_no_results(self, mock_retrieval_agent: RetrievalAgent) -> None:
        mock_retrieval_agent._rag.retrieve.return_value = []

        st = SubTask(
            subtask_id="st-1",
            description="量子纠缠的最新论文",
            required_agents=["retrieval_agent"],
        )
        result = await mock_retrieval_agent.run(st, "sess-001", "u001")
        assert result.success is True
        assert "未在知识库中找到" in str(result.output)

    @pytest.mark.asyncio
    async def test_run_rag_unavailable(
        self, mock_retrieval_agent: RetrievalAgent,
    ) -> None:
        mock_retrieval_agent._rag.retrieve.side_effect = ConnectionError(
            "Connection refused",
        )

        st = SubTask(
            subtask_id="st-1",
            description="什么是机器学习？",
            required_agents=["retrieval_agent"],
        )
        result = await mock_retrieval_agent.run(st, "sess-001", "u001")
        assert result.success is True
        assert "知识库暂不可用" in str(result.output)

    @pytest.mark.asyncio
    async def test_enhance_query_no_memory(
        self, mock_retrieval_agent: RetrievalAgent,
    ) -> None:
        enhanced = await mock_retrieval_agent._enhance_query(
            "Python 是什么", "sess-001", "u001",
        )
        assert enhanced == "Python 是什么"

    @pytest.mark.asyncio
    async def test_enhance_query_with_memory(
        self, mock_retrieval_agent: RetrievalAgent,
    ) -> None:
        mm = MagicMock(spec=MemoryManager)
        mm.get_recent.return_value = [
            MemoryEntry(content="之前讨论了 Java", memory_type="short_term"),
            MemoryEntry(content="用户对比了多种语言", memory_type="short_term"),
        ]
        mock_retrieval_agent.inject_memory(mm)

        enhanced = await mock_retrieval_agent._enhance_query(
            "Python 有什么优势", "sess-001", "u001",
        )
        assert "【历史上下文】" in enhanced
        assert "Java" in enhanced
        assert "【当前问题】" in enhanced
        assert "Python 有什么优势" in enhanced

    @pytest.mark.asyncio
    async def test_enhance_query_memory_empty(
        self, mock_retrieval_agent: RetrievalAgent,
    ) -> None:
        mm = MagicMock(spec=MemoryManager)
        mm.get_recent.return_value = []
        mock_retrieval_agent.inject_memory(mm)

        enhanced = await mock_retrieval_agent._enhance_query(
            "原始查询", "sess-001", "u001",
        )
        assert enhanced == "原始查询"


# ══════════════════════════════════════════════════════
# LLMAgent ReAct 推理循环测试
# ══════════════════════════════════════════════════════


class TestReActParsing:
    """ReAct 输出解析单元测试。"""

    def test_parse_final_answer(self) -> None:
        text = "Thought: 已获取计算结果\nFinal Answer: 2+3=5"
        result = LLMAgent._parse_react_output(text)
        assert result is not None
        assert result["thought"] == "已获取计算结果"
        assert result["final_answer"] == "2+3=5"

    def test_parse_action(self) -> None:
        text = "Thought: 需要计算\nAction: calculator\nAction Input: 2+3*4"
        result = LLMAgent._parse_react_output(text)
        assert result is not None
        assert result["thought"] == "需要计算"
        assert result["action"] == "calculator"
        assert result["action_input"] == "2+3*4"

    def test_parse_action_no_input(self) -> None:
        text = "Thought: 查看时间\nAction: get_current_time"
        result = LLMAgent._parse_react_output(text)
        assert result is not None
        assert result["action"] == "get_current_time"
        assert result["action_input"] == ""

    def test_parse_empty(self) -> None:
        assert LLMAgent._parse_react_output("") is None
        assert LLMAgent._parse_react_output("   ") is None

    def test_parse_no_thought(self) -> None:
        text = "Just some random text without format"
        assert LLMAgent._parse_react_output(text) is None

    def test_parse_multiline_thought(self) -> None:
        text = (
            "Thought: 第一步先分析问题\n"
            "然后确定需要调用什么工具\n"
            "Action: calculator\n"
            "Action Input: 100/3"
        )
        result = LLMAgent._parse_react_output(text)
        assert result is not None
        assert "分析问题" in result["thought"]
        assert result["action"] == "calculator"

    def test_parse_multiline_final_answer(self) -> None:
        text = "Thought: 总结\nFinal Answer: 第一行\n第二行\n第三行"
        result = LLMAgent._parse_react_output(text)
        assert result is not None
        assert "第一行" in result["final_answer"]
        assert "第三行" in result["final_answer"]


class TestReActValidation:
    """ReAct 输出校验测试。"""

    def test_validate_valid_output(self) -> None:
        assert LLMAgent._validate_output("这是一个有效输出") is True

    def test_validate_empty_output(self) -> None:
        assert LLMAgent._validate_output("") is False
        assert LLMAgent._validate_output("   ") is False

    def test_validate_too_short(self) -> None:
        assert LLMAgent._validate_output("x") is False

    def test_validate_normal_length(self) -> None:
        assert LLMAgent._validate_output("OK") is True


class TestReActLoop:
    """ReAct 推理循环集成测试。"""

    @pytest.fixture()
    def react_agent(self) -> LLMAgent:
        """创建带 ToolRegistry 的 LLMAgent（ReAct 模式）。"""
        with (
            patch("agent_hub.agents.llm_agent.Anthropic"),
            patch("agent_hub.agents.llm_agent.OpenAI"),
        ):
            agent = LLMAgent(max_react_rounds=3)
            agent._fallback = None  # 简化测试，禁用降级
            registry = ToolRegistry()
            registry.register_defaults()
            agent.inject_registry(registry)
            return agent

    def test_should_use_react_with_registry(self, react_agent: LLMAgent) -> None:
        st = SubTask(
            subtask_id="st-1",
            description="calculator(2+3)",
            required_agents=["tool_agent", "llm_agent"],
        )
        assert react_agent._should_use_react(st) is True

    def test_should_not_use_react_without_tool_agent(
        self, react_agent: LLMAgent,
    ) -> None:
        st = SubTask(
            subtask_id="st-1",
            description="写一首诗",
            required_agents=["llm_agent"],
        )
        assert react_agent._should_use_react(st) is False

    def test_should_not_use_react_without_registry(self) -> None:
        with (
            patch("agent_hub.agents.llm_agent.Anthropic"),
            patch("agent_hub.agents.llm_agent.OpenAI"),
        ):
            agent = LLMAgent()
            st = SubTask(
                subtask_id="st-1",
                description="test",
                required_agents=["tool_agent"],
            )
            assert agent._should_use_react(st) is False

    @pytest.mark.asyncio
    async def test_react_single_round_final_answer(
        self, react_agent: LLMAgent,
    ) -> None:
        """单轮直接给出 Final Answer。"""
        react_agent._call_primary_with_system = MagicMock(
            return_value="Thought: 这个问题很简单\nFinal Answer: 42是宇宙的答案",
        )

        st = SubTask(
            subtask_id="st-1",
            description="宇宙的答案是什么",
            required_agents=["tool_agent", "llm_agent"],
        )
        result = await react_agent.run(st, "sess-001", "u001")
        assert result.success is True
        assert "42" in str(result.output)
        assert result.react_trace is not None
        assert result.react_trace.total_rounds == 1
        assert result.react_trace.final_answer is not None

    @pytest.mark.asyncio
    async def test_react_multi_round_with_tool(
        self, react_agent: LLMAgent,
    ) -> None:
        """多轮 Thought→Action→Observation 后给出 Final Answer。"""
        call_count = 0

        def mock_llm_call(system_prompt, messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Thought: 需要计算 2+3\nAction: calculator\nAction Input: 2+3"
            return "Thought: 计算结果是5\nFinal Answer: 2+3 的结果是 5"

        react_agent._call_primary_with_system = MagicMock(side_effect=mock_llm_call)

        st = SubTask(
            subtask_id="st-1",
            description="请帮我算 2+3",
            required_agents=["tool_agent", "llm_agent"],
        )
        result = await react_agent.run(st, "sess-001", "u001")
        assert result.success is True
        assert "5" in str(result.output)
        assert result.react_trace is not None
        assert result.react_trace.total_rounds == 2
        assert len(result.react_trace.steps) == 2
        # 第一步应该有 action 和 observation
        assert result.react_trace.steps[0].action == "calculator"
        assert result.react_trace.steps[0].observation is not None
        assert "5" in result.react_trace.steps[0].observation

    @pytest.mark.asyncio
    async def test_react_max_rounds_reached(
        self, react_agent: LLMAgent,
    ) -> None:
        """达到最大轮次自动截止。"""
        react_agent._call_primary_with_system = MagicMock(
            return_value="Thought: 需要更多信息\nAction: calculator\nAction Input: 1+1",
        )

        st = SubTask(
            subtask_id="st-1",
            description="不断调用工具",
            required_agents=["tool_agent", "llm_agent"],
        )
        result = await react_agent.run(st, "sess-001", "u001")
        assert result.success is True
        assert "最大推理轮次" in str(result.output)
        assert result.react_trace is not None
        assert result.react_trace.total_rounds == 3  # max_react_rounds=3
        assert result.react_trace.final_answer is None

    @pytest.mark.asyncio
    async def test_react_format_error_retry(
        self, react_agent: LLMAgent,
    ) -> None:
        """格式错误触发重试，最终给出 Final Answer。"""
        call_count = 0

        def mock_llm_call(system_prompt, messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "这是一个格式错误的输出"
            return "Thought: 修正格式\nFinal Answer: 正确答案"

        react_agent._call_primary_with_system = MagicMock(side_effect=mock_llm_call)

        st = SubTask(
            subtask_id="st-1",
            description="测试",
            required_agents=["tool_agent", "llm_agent"],
        )
        result = await react_agent.run(st, "sess-001", "u001")
        assert result.success is True
        assert result.output == "正确答案"

    @pytest.mark.asyncio
    async def test_react_format_error_exceed_limit(
        self, react_agent: LLMAgent,
    ) -> None:
        """格式错误超过容忍次数，将原始输出作为结果。"""
        react_agent._call_primary_with_system = MagicMock(
            return_value="完全不符合格式的输出，但内容有意义",
        )

        st = SubTask(
            subtask_id="st-1",
            description="测试",
            required_agents=["tool_agent", "llm_agent"],
        )
        result = await react_agent.run(st, "sess-001", "u001")
        assert result.success is True
        assert "完全不符合格式的输出" in str(result.output)

    @pytest.mark.asyncio
    async def test_react_tool_not_found(
        self, react_agent: LLMAgent,
    ) -> None:
        """ReAct 循环中调用不存在的工具。"""
        call_count = 0

        def mock_llm_call(system_prompt, messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Thought: 调用不存在的工具\nAction: nonexistent_tool\nAction Input: test"
            return "Thought: 工具不存在，直接回答\nFinal Answer: 无法找到该工具"

        react_agent._call_primary_with_system = MagicMock(side_effect=mock_llm_call)

        st = SubTask(
            subtask_id="st-1",
            description="测试",
            required_agents=["tool_agent", "llm_agent"],
        )
        result = await react_agent.run(st, "sess-001", "u001")
        assert result.success is True
        # 第一步的 observation 应该包含错误信息
        assert result.react_trace is not None
        assert "不存在" in result.react_trace.steps[0].observation

    @pytest.mark.asyncio
    async def test_react_no_registry_falls_back_to_simple(self) -> None:
        """没有 registry 时降级为普通单次调用。"""
        with (
            patch("agent_hub.agents.llm_agent.Anthropic"),
            patch("agent_hub.agents.llm_agent.OpenAI"),
        ):
            agent = LLMAgent()
            agent._fallback = None

            mock_msg = MagicMock()
            mock_msg.message.content = "简单回复"
            mock_resp = MagicMock()
            mock_resp.choices = [mock_msg]
            agent._primary.chat.completions.create.return_value = mock_resp

            st = SubTask(
                subtask_id="st-1",
                description="你好",
                required_agents=["tool_agent", "llm_agent"],
            )
            result = await agent.run(st, "sess-001", "u001")
            assert result.success is True
            assert result.output == "简单回复"
            assert result.react_trace is None


class TestReActActionInputParsing:
    """ReAct Action Input 参数解析测试。"""

    def test_parse_simple_value(self) -> None:
        result = LLMAgent._parse_action_input("2+3*4", "calculator")
        assert result == {"expression": "2+3*4"}

    def test_parse_kv_format(self) -> None:
        result = LLMAgent._parse_action_input("path=/tmp/a.txt", "file_read")
        assert result == {"path": "/tmp/a.txt"}

    def test_parse_empty(self) -> None:
        result = LLMAgent._parse_action_input("", "get_current_time")
        assert result == {}

    def test_parse_non_calculator_value(self) -> None:
        result = LLMAgent._parse_action_input("hello world", "search_web")
        assert result == {"value": "hello world"}
