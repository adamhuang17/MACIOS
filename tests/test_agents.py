"""Agent 执行层单元测试。

覆盖：
- ToolRegistry 注册 / 查询 / 权限过滤
- Tool 默认不内置本地 demo 原语
- ToolAgent 正则解析 / 权限检查 / 执行
- LLMAgent prompt 构建 / 双 Provider 降级
- RetrievalAgent 检索成功 / RAG 不可用降级
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_hub.agents.llm_agent import LLMAgent
from agent_hub.agents.registry import ToolRegistry
from agent_hub.agents.retrieval_agent import RetrievalAgent
from agent_hub.agents.tool_agent import ToolAgent
from agent_hub.core.models import MemoryEntry, SubTask
from agent_hub.memory import MemoryManager

# ── Fixtures ─────────────────────────────────────────


@pytest.fixture()
def registry() -> ToolRegistry:
    """显式注册本地适配器与 MCP 工具 schema 的注册表。"""
    r = ToolRegistry()

    async def local_echo(params: dict[str, Any]) -> str:
        return f"echo:{params.get('value', params.get('text', ''))}"

    r.register(
        "local_echo",
        "测试用本地适配器",
        {"value": {"type": "string"}},
        local_echo,
    )
    r.register(
        "mcp__docs__search",
        "MCP docs server search tool",
        {"query": {"type": "string"}},
        source="mcp",
        server_name="docs",
    )
    r.register(
        "mcp__ops__deploy",
        "MCP ops server deploy tool",
        {"target": {"type": "string"}},
        requires_admin=True,
        risk_level="admin",
        side_effect=True,
        source="mcp",
        server_name="ops",
    )
    return r


@pytest.fixture()
def subtask() -> SubTask:
    return SubTask(
        subtask_id="st-test-001",
        description="mcp__docs__search(query=agent skills)",
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
        assert names == {"local_echo", "mcp__docs__search", "mcp__ops__deploy"}

    def test_register_defaults_is_empty(self) -> None:
        r = ToolRegistry()
        r.register_defaults()
        assert r.list_tools() == []

    def test_list_tools_for_user(self, registry: ToolRegistry) -> None:
        user_tools = registry.list_tools_for_user("user")
        admin_tools = registry.list_tools_for_user("admin")

        user_names = {t.name for t in user_tools}
        admin_names = {t.name for t in admin_tools}

        assert "mcp__ops__deploy" not in user_names

        assert "mcp__ops__deploy" in admin_names
        assert "mcp__docs__search" in admin_names

    def test_list_tools_for_admin_superset(self, registry: ToolRegistry) -> None:
        user_tools = {t.name for t in registry.list_tools_for_user("user")}
        admin_tools = {t.name for t in registry.list_tools_for_user("admin")}
        assert user_tools.issubset(admin_tools)


# ══════════════════════════════════════════════════════
# ToolAgent 测试
# ══════════════════════════════════════════════════════


class TestToolAgent:
    def test_parse_tool_call_simple(self, registry: ToolRegistry) -> None:
        agent = ToolAgent(registry)
        result = agent._parse_tool_call("mcp__docs__search(agent skills)")
        assert result is not None
        name, params = result
        assert name == "mcp__docs__search"
        assert params == {"value": "agent skills"}

    def test_parse_tool_call_kv(self, registry: ToolRegistry) -> None:
        agent = ToolAgent(registry)
        result = agent._parse_tool_call("mcp__docs__search(query=agent skills)")
        assert result is not None
        name, params = result
        assert name == "mcp__docs__search"
        assert params == {"query": "agent skills"}

    def test_parse_tool_call_empty_args(self, registry: ToolRegistry) -> None:
        agent = ToolAgent(registry)
        result = agent._parse_tool_call("local_echo()")
        assert result is not None
        name, params = result
        assert name == "local_echo"
        assert params == {}

    def test_parse_tool_call_failure(self, registry: ToolRegistry) -> None:
        agent = ToolAgent(registry)
        assert agent._parse_tool_call("这是一句普通话") is None

    def test_check_permission_user_on_admin_tool(
        self, registry: ToolRegistry,
    ) -> None:
        agent = ToolAgent(registry)
        assert agent._check_permission("mcp__ops__deploy", "user") is False

    def test_check_permission_admin_on_admin_tool(
        self, registry: ToolRegistry,
    ) -> None:
        agent = ToolAgent(registry)
        assert agent._check_permission("mcp__ops__deploy", "admin") is True

    def test_check_permission_user_on_normal_tool(
        self, registry: ToolRegistry,
    ) -> None:
        agent = ToolAgent(registry)
        assert agent._check_permission("local_echo", "user") is True
        assert agent._check_permission("mcp__docs__search", "user") is True

    def test_check_permission_nonexistent(self, registry: ToolRegistry) -> None:
        agent = ToolAgent(registry)
        assert agent._check_permission("nonexistent", "admin") is False

    @pytest.mark.asyncio
    async def test_run_local_adapter(self, registry: ToolRegistry) -> None:
        agent = ToolAgent(registry)
        st = SubTask(
            subtask_id="st-1",
            description="local_echo(hello)",
            required_agents=["tool_agent"],
        )
        result = await agent.run(st, "sess-001", "u001")
        assert result.success is True
        assert "echo:hello" in str(result.output)

    @pytest.mark.asyncio
    async def test_run_mcp_tool_not_local_executable(
        self, registry: ToolRegistry,
    ) -> None:
        agent = ToolAgent(registry)
        st = SubTask(
            subtask_id="st-mcp",
            description="mcp__docs__search(query=agent skills)",
            required_agents=["tool_agent"],
        )
        result = await agent.run(st, "sess-001", "u001")
        assert result.success is False
        assert "外部 MCP 服务" in str(result.error)

    @pytest.mark.asyncio
    async def test_run_permission_denied(self, registry: ToolRegistry) -> None:
        agent = ToolAgent(registry)
        agent.set_user_role("user")
        st = SubTask(
            subtask_id="st-2",
            description="mcp__ops__deploy(target=prod)",
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
            description="local_echo(admin)",
            required_agents=["tool_agent"],
        )
        result = await agent.run(st, "sess-001", "u001")
        assert result.success is True
        assert "echo:admin" in str(result.output)

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
        text = "Thought: 需要检索\nAction: mcp__docs__search\nAction Input: query=agent skills"
        result = LLMAgent._parse_react_output(text)
        assert result is not None
        assert result["thought"] == "需要检索"
        assert result["action"] == "mcp__docs__search"
        assert result["action_input"] == "query=agent skills"

    def test_parse_action_no_input(self) -> None:
        text = "Thought: 查看文档\nAction: mcp__docs__list"
        result = LLMAgent._parse_react_output(text)
        assert result is not None
        assert result["action"] == "mcp__docs__list"
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
            "Action: mcp__docs__search\n"
            "Action Input: query=agent skills"
        )
        result = LLMAgent._parse_react_output(text)
        assert result is not None
        assert "分析问题" in result["thought"]
        assert result["action"] == "mcp__docs__search"

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

            async def local_echo(params: dict[str, Any]) -> str:
                return f"echo:{params.get('value', '')}"

            registry.register(
                "local_echo",
                "测试用本地适配器",
                {"value": {"type": "string"}},
                local_echo,
            )
            agent.inject_registry(registry)
            return agent

    def test_should_use_react_with_registry(self, react_agent: LLMAgent) -> None:
        st = SubTask(
            subtask_id="st-1",
            description="local_echo(hello)",
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
                return "Thought: 需要调用本地适配器\nAction: local_echo\nAction Input: hello"
            return "Thought: 已拿到工具结果\nFinal Answer: 工具返回 echo:hello"

        react_agent._call_primary_with_system = MagicMock(side_effect=mock_llm_call)

        st = SubTask(
            subtask_id="st-1",
            description="请调用 echo 工具",
            required_agents=["tool_agent", "llm_agent"],
        )
        result = await react_agent.run(st, "sess-001", "u001")
        assert result.success is True
        assert "echo:hello" in str(result.output)
        assert result.react_trace is not None
        assert result.react_trace.total_rounds == 2
        assert len(result.react_trace.steps) == 2
        # 第一步应该有 action 和 observation
        assert result.react_trace.steps[0].action == "local_echo"
        assert result.react_trace.steps[0].observation is not None
        assert "echo:hello" in result.react_trace.steps[0].observation

    @pytest.mark.asyncio
    async def test_react_max_rounds_reached(
        self, react_agent: LLMAgent,
    ) -> None:
        """达到最大轮次自动截止。"""
        react_agent._call_primary_with_system = MagicMock(
            return_value="Thought: 需要更多信息\nAction: local_echo\nAction Input: loop",
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
        result = LLMAgent._parse_action_input("agent skills", "mcp__docs__search")
        assert result == {"value": "agent skills"}

    def test_parse_kv_format(self) -> None:
        result = LLMAgent._parse_action_input("query=agent skills", "mcp__docs__search")
        assert result == {"query": "agent skills"}

    def test_parse_empty(self) -> None:
        result = LLMAgent._parse_action_input("", "mcp__docs__list")
        assert result == {}

    def test_parse_non_kv_value(self) -> None:
        result = LLMAgent._parse_action_input("hello world", "local_echo")
        assert result == {"value": "hello world"}
