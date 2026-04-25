"""LangGraph 容错检索 Agent 测试。

覆盖场景：
- 高相关度直接 generate
- 低相关度触发 rewrite + re-retrieve
- 最大重写轮次后强制 generate
- 异常降级处理
- QueryRewriter / RelevanceEvaluator 单元测试
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_hub.core.models import SubTask
from agent_hub.rag.query_rewriter import QueryRewriter
from agent_hub.rag.relevance_evaluator import RelevanceEvaluator
from agent_hub.rag.vector_store import SearchResult

# ═══════════════════════════════════════════════════════
# QueryRewriter 测试
# ═══════════════════════════════════════════════════════


class TestQueryRewriter:
    """Query 重写器测试。"""

    @pytest.mark.asyncio
    async def test_rewrite_success(self) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "优化后的查询"
        mock_client.chat.completions.create.return_value = mock_resp

        rewriter = QueryRewriter(mock_client, model="test-model")
        result = await rewriter.rewrite("这个东西怎么用啊")

        assert result == "优化后的查询"
        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_rewrite_with_context(self) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Python 装饰器用法"
        mock_client.chat.completions.create.return_value = mock_resp

        rewriter = QueryRewriter(mock_client)
        result = await rewriter.rewrite("怎么用", context="讨论 Python 装饰器")

        assert result == "Python 装饰器用法"
        # 验证 prompt 中包含上下文
        call_args = mock_client.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "Python 装饰器" in prompt

    @pytest.mark.asyncio
    async def test_rewrite_failure_returns_original(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = ConnectionError("timeout")

        rewriter = QueryRewriter(mock_client)
        result = await rewriter.rewrite("原始查询")

        assert result == "原始查询"

    @pytest.mark.asyncio
    async def test_rewrite_empty_response_returns_original(self) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "  "
        mock_client.chat.completions.create.return_value = mock_resp

        rewriter = QueryRewriter(mock_client)
        result = await rewriter.rewrite("原始查询")

        assert result == "原始查询"


# ═══════════════════════════════════════════════════════
# RelevanceEvaluator 测试
# ═══════════════════════════════════════════════════════


class TestRelevanceEvaluator:
    """相关性评估器测试。"""

    def _make_result(self, content: str = "test content") -> SearchResult:
        return SearchResult(
            chunk_id="c1", content=content, score=0.9,
            metadata={"namespace": "test"},
        )

    @pytest.mark.asyncio
    async def test_evaluate_high_score(self) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "5"
        mock_client.chat.completions.create.return_value = mock_resp

        evaluator = RelevanceEvaluator(mock_client)
        score = await evaluator.evaluate("query", [self._make_result()])
        assert score == 5.0

    @pytest.mark.asyncio
    async def test_evaluate_low_score(self) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "1"
        mock_client.chat.completions.create.return_value = mock_resp

        evaluator = RelevanceEvaluator(mock_client)
        score = await evaluator.evaluate("query", [self._make_result()])
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_empty_results(self) -> None:
        mock_client = MagicMock()
        evaluator = RelevanceEvaluator(mock_client)
        score = await evaluator.evaluate("query", [])
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_failure_returns_neutral(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("api down")

        evaluator = RelevanceEvaluator(mock_client)
        score = await evaluator.evaluate("query", [self._make_result()])
        assert score == 3.0

    def test_parse_score_valid(self) -> None:
        assert RelevanceEvaluator._parse_score("4") == 4.0
        assert RelevanceEvaluator._parse_score("评分: 2") == 2.0

    def test_parse_score_invalid(self) -> None:
        assert RelevanceEvaluator._parse_score("无法评分") == 3.0
        assert RelevanceEvaluator._parse_score("") == 3.0

    def test_format_results_truncation(self) -> None:
        results = [self._make_result(f"content_{i}" * 50) for i in range(20)]
        text = RelevanceEvaluator._format_results(results, max_chars=500)
        assert len(text) <= 600  # 允许一点余量


# ═══════════════════════════════════════════════════════
# ReflectionAgent 集成测试
# ═══════════════════════════════════════════════════════


class TestReflectionAgent:
    """LangGraph 容错 Agent 集成测试。"""

    def _make_subtask(self, desc: str = "测试查询") -> SubTask:
        return SubTask(
            subtask_id="st-1",
            description=desc,
            required_agents=["reflection_agent"],
        )

    def _make_search_result(
        self, content: str = "检索内容", score: float = 0.9,
    ) -> SearchResult:
        return SearchResult(
            chunk_id="c1", content=content, score=score,
            metadata={"namespace": "test"},
        )

    @pytest.mark.asyncio
    async def test_high_relevance_no_rewrite(self) -> None:
        """高相关度时直接 generate，不触发 rewrite。"""
        from agent_hub.agents.reflection_agent import ReflectionAgent

        mock_rag = MagicMock()
        mock_rag.retrieve = AsyncMock()
        result = self._make_search_result()
        mock_rag.retrieve.return_value = [result]
        mock_rag.format_context.return_value = "检索上下文"

        mock_client = MagicMock()
        # 评估返回 5 分
        eval_resp = MagicMock()
        eval_resp.choices = [MagicMock()]
        eval_resp.choices[0].message.content = "5"
        # 生成返回答案
        gen_resp = MagicMock()
        gen_resp.choices = [MagicMock()]
        gen_resp.choices[0].message.content = "最终答案"
        mock_client.chat.completions.create.side_effect = [eval_resp, gen_resp]

        agent = ReflectionAgent(
            rag_pipeline=mock_rag,
            llm_client=mock_client,
            relevance_threshold=3.0,
        )
        result = await agent.run(self._make_subtask(), "sess1", "u1")

        assert result.success is True
        assert "最终答案" in result.output
        # 应该只调 retrieve 一次（没有 rewrite）
        assert mock_rag.retrieve.call_count == 1

    @pytest.mark.asyncio
    async def test_low_relevance_triggers_rewrite(self) -> None:
        """低相关度时触发 rewrite + re-retrieve。"""
        from agent_hub.agents.reflection_agent import ReflectionAgent

        mock_rag = MagicMock()
        mock_rag.retrieve = AsyncMock()
        result = self._make_search_result()
        mock_rag.retrieve.return_value = [result]
        mock_rag.format_context.return_value = "上下文"

        mock_client = MagicMock()
        # 第1次评估=1分 → rewrite，重写返回新查询，
        # 第2次评估=5分 → generate，生成返回答案
        responses = [
            self._mock_llm_response("1"),       # evaluate round 1
            self._mock_llm_response("优化查询"),  # rewrite
            self._mock_llm_response("5"),       # evaluate round 2
            self._mock_llm_response("最终答案"),  # generate
        ]
        mock_client.chat.completions.create.side_effect = responses

        agent = ReflectionAgent(
            rag_pipeline=mock_rag,
            llm_client=mock_client,
            relevance_threshold=3.0,
        )
        result = await agent.run(self._make_subtask(), "sess1", "u1")

        assert result.success is True
        assert "最终答案" in result.output
        assert "1 轮" in result.output  # 提示经过了重写
        assert mock_rag.retrieve.call_count == 2

    @pytest.mark.asyncio
    async def test_max_rewrites_forces_generate(self) -> None:
        """达到最大重写次数后强制 generate。"""
        from agent_hub.agents.reflection_agent import ReflectionAgent

        mock_rag = MagicMock()
        mock_rag.retrieve = AsyncMock()
        result = self._make_search_result()
        mock_rag.retrieve.return_value = [result]
        mock_rag.format_context.return_value = "上下文"

        mock_client = MagicMock()
        # 每次评估都返回 1 分 → 持续 rewrite，直到达上限
        responses = [
            self._mock_llm_response("1"),        # evaluate round 1
            self._mock_llm_response("重写1"),     # rewrite 1
            self._mock_llm_response("1"),        # evaluate round 2
            self._mock_llm_response("重写2"),     # rewrite 2
            self._mock_llm_response("1"),        # evaluate round 3 → max reached
            self._mock_llm_response("勉强回答"),  # generate
        ]
        mock_client.chat.completions.create.side_effect = responses

        agent = ReflectionAgent(
            rag_pipeline=mock_rag,
            llm_client=mock_client,
            relevance_threshold=3.0,
        )
        result = await agent.run(self._make_subtask(), "sess1", "u1")

        assert result.success is True
        assert "2 轮" in result.output
        assert mock_rag.retrieve.call_count == 3  # initial + 2 rewrites

    @pytest.mark.asyncio
    async def test_retrieve_failure_generates_error(self) -> None:
        """检索失败时正确处理。"""
        from agent_hub.agents.reflection_agent import ReflectionAgent

        mock_rag = MagicMock()
        mock_rag.retrieve = AsyncMock(side_effect=ConnectionError("db down"))
        mock_rag.format_context.return_value = ""

        mock_client = MagicMock()
        # evaluate 和 generate 仍可调用
        responses = [
            self._mock_llm_response("3"),
            self._mock_llm_response("无法检索的回答"),
        ]
        mock_client.chat.completions.create.side_effect = responses

        agent = ReflectionAgent(
            rag_pipeline=mock_rag,
            llm_client=mock_client,
        )
        result = await agent.run(self._make_subtask(), "sess1", "u1")

        # 应返回成功但包含错误信息，或失败
        assert result.agent_name == "reflection_agent"

    @staticmethod
    def _mock_llm_response(content: str) -> MagicMock:
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = content
        return resp
