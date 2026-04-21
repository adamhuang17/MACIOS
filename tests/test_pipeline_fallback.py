"""Pipeline 兜底逻辑单元测试。

覆盖：
- sub_tasks 为空 + TOOL_EXECUTION → 兜底生成子任务 → ReAct 触发
- sub_tasks 为空 + TASK_GENERATION → 兜底 → llm_agent 执行
- GROUP_CHAT → 不触发兜底
- Router prompt 包含"必须至少生成一个子任务"
"""

from __future__ import annotations

import pytest

from agent_hub.core.enums import ExecutionMode
from agent_hub.core.models import RoutingDecision, SubTask
from agent_hub.core.pipeline import AgentPipeline
from agent_hub.core.router import _SYSTEM_PROMPT


# ── 直接测试 _ensure_default_subtask ──────────────────


class TestPipelineFallback:
    """测试 Pipeline 兜底逻辑。"""

    @pytest.fixture()
    def pipeline(self, tmp_path):
        """构造一个最小化 Pipeline（不依赖真实 LLM）。"""
        from unittest.mock import patch

        from agent_hub.config.settings import Settings

        settings = Settings(
            llm_api_key="test-key",
            llm_base_url="http://localhost:1234",
            obsidian_vault_path=str(tmp_path / "vault"),
            guard_enabled=False,
        )
        with (
            patch("agent_hub.agents.llm_agent.Anthropic"),
            patch("agent_hub.agents.llm_agent.OpenAI"),
            patch("agent_hub.core.pipeline.RAGPipeline"),
        ):
            return AgentPipeline(settings)

    def test_fallback_tool_execution(self, pipeline: AgentPipeline) -> None:
        """ACT + 空 plan → 兜底生成含 tool_agent 的子任务。"""
        routing = RoutingDecision(
            mode=ExecutionMode.ACT,
            confidence=0.9,
            reasoning="用户要求计算",
            plan=[],
        )
        result = pipeline._ensure_default_subtask(routing, "帮我算一下 2+3")

        assert len(result.plan) == 1
        st = result.plan[0]
        assert st.subtask_id == "subtask_fallback_1"
        assert st.description == "帮我算一下 2+3"
        assert "llm_agent" in st.required_agents
        assert "tool_agent" in st.required_agents
        assert st.priority == 1
        assert st.depends_on == []

    def test_fallback_task_generation(self, pipeline: AgentPipeline) -> None:
        """PLAN + 空 plan → 兜底 → llm_agent 子任务。"""
        routing = RoutingDecision(
            mode=ExecutionMode.PLAN,
            confidence=0.85,
            reasoning="用户要写代码",
            plan=[],
        )
        result = pipeline._ensure_default_subtask(routing, "写一个快速排序")

        assert len(result.plan) == 1
        st = result.plan[0]
        assert st.required_agents == ["llm_agent"]
        assert st.description == "写一个快速排序"

    def test_fallback_retrieval(self, pipeline: AgentPipeline) -> None:
        """QA + 空 plan → 兜底生成 retrieval_agent 子任务。"""
        routing = RoutingDecision(
            mode=ExecutionMode.QA,
            confidence=0.8,
            reasoning="用户查询知识",
            plan=[],
        )
        result = pipeline._ensure_default_subtask(routing, "RAG是什么")

        assert len(result.plan) == 1
        assert result.plan[0].required_agents == ["retrieval_agent"]

    def test_no_fallback_group_chat(self, pipeline: AgentPipeline) -> None:
        """IGNORE → _ensure_default_subtask 不生成子任务。"""
        routing = RoutingDecision(
            mode=ExecutionMode.IGNORE,
            confidence=0.95,
            reasoning="普通群聊",
            plan=[],
        )
        result = pipeline._ensure_default_subtask(routing, "哈哈哈")

        assert len(result.plan) == 0

    def test_no_fallback_when_subtasks_exist(self, pipeline: AgentPipeline) -> None:
        """已有 sub_tasks 时，run() 中的条件不会触发兜底。"""
        existing = SubTask(
            subtask_id="subtask_1",
            description="已有任务",
            required_agents=["llm_agent"],
        )
        routing = RoutingDecision(
            mode=ExecutionMode.PLAN,
            confidence=0.9,
            reasoning="测试",
            plan=[existing],
        )
        # 模拟 run() 中的条件判断
        should_fallback = not routing.plan and routing.mode != ExecutionMode.IGNORE
        assert should_fallback is False


class TestRouterPromptEnsuresSubtask:
    """验证 Router prompt 包含强制子任务生成的规则。"""

    def test_router_prompt_ensures_subtask(self) -> None:
        assert "必须至少生成一个子任务" in _SYSTEM_PROMPT

    def test_router_prompt_group_chat_exception(self) -> None:
        assert "ignore" in _SYSTEM_PROMPT
        assert "空列表" in _SYSTEM_PROMPT
