"""意图识别路由器单元测试。

mock 掉 LLM 调用（_call_llm），单独测试路由逻辑：
- 10 条真实 query 分类
-  ADMIN_COMMAND 权限过滤
- 低置信度降级
- LLM 异常降级兜底
- 子任务拆解完整性
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agent_hub.core.enums import IntentType, UserRole
from agent_hub.core.models import RoutingResult, SubTask, UserContext
from agent_hub.core.router import IntentRouter


# ── Fixtures ─────────────────────────────────────────


@pytest.fixture()
def router() -> IntentRouter:
    """创建一个路由器实例（不会真正调用 LLM）。"""
    return IntentRouter(client=AsyncMock())


@pytest.fixture()
def admin_ctx() -> UserContext:
    return UserContext(user_id="admin001", role=UserRole.ADMIN, channel="dingtalk")


@pytest.fixture()
def user_ctx() -> UserContext:
    return UserContext(
        user_id="user001",
        role=UserRole.USER,
        channel="qq",
        group_id="g100",
    )


# ── 辅助函数：构造 mock 的 _call_llm 返回值 ─────────


def _make_routing_result(
    intent: IntentType,
    confidence: float = 0.92,
    reasoning: str = "测试推理",
    sub_tasks: list[SubTask] | None = None,
) -> RoutingResult:
    return RoutingResult(
        intent=intent,
        confidence=confidence,
        reasoning=reasoning,
        sub_tasks=sub_tasks or [],
    )


def _make_result_with_subtasks(
    intent: IntentType,
    confidence: float = 0.92,
) -> RoutingResult:
    return RoutingResult(
        intent=intent,
        confidence=confidence,
        reasoning="复杂请求，拆解子任务",
        sub_tasks=[
            SubTask(
                subtask_id="subtask_1",
                description="第一步操作",
                required_agents=["llm_agent"],
                priority=1,
            ),
            SubTask(
                subtask_id="subtask_2",
                description="第二步操作",
                required_agents=["retrieval_agent"],
                priority=2,
                depends_on=["subtask_1"],
            ),
        ],
    )


# ── 测试 10 条真实 query 分类 ─────────────────────────


class TestRouteClassification:
    """使用 10 条真实 query 测试路由分类结果。"""

    _CASES: list[tuple[str, IntentType, bool]] = [
        # (query, 期望意图, 是否含子任务)
        ("帮我写一个快速排序算法", IntentType.TASK_GENERATION, False),
        ("公司年假政策是什么", IntentType.RETRIEVAL, False),
        ("查一下北京今天天气", IntentType.TOOL_EXECUTION, False),
        ("帮我分析这个Excel文件", IntentType.FILE_PROCESSING, False),
        ("重启测试服务器", IntentType.ADMIN_COMMAND, False),
        ("哈哈哈今天天气真好", IntentType.GROUP_CHAT, False),
        ("你刚才的回答不对，重新想想", IntentType.REFLECTION, False),
        ("帮我写一份项目周报然后发到钉钉群", IntentType.TASK_GENERATION, True),
        ("搜索一下关于LLM的论文然后总结要点", IntentType.RETRIEVAL, True),
        ("这个文件里有什么内容", IntentType.FILE_PROCESSING, False),
    ]

    @pytest.mark.parametrize(
        ("query", "expected_intent", "has_subtasks"),
        _CASES,
        ids=[
            "write_quicksort",
            "vacation_policy",
            "check_weather",
            "analyze_excel",
            "restart_server",
            "casual_chat",
            "reflection",
            "complex_report",
            "complex_search",
            "file_query",
        ],
    )
    async def test_classification(
        self,
        router: IntentRouter,
        admin_ctx: UserContext,
        query: str,
        expected_intent: IntentType,
        has_subtasks: bool,
    ) -> None:
        """验证各 query 被路由到正确的意图类型。"""
        # 构造 mock 返回：模拟 LLM 返回了正确的意图
        if has_subtasks:
            mock_result = _make_result_with_subtasks(expected_intent)
        else:
            mock_result = _make_routing_result(expected_intent)

        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route(query, admin_ctx)

        assert result.intent == expected_intent
        assert 0.0 <= result.confidence <= 1.0
        assert result.reasoning

        if has_subtasks:
            assert len(result.sub_tasks) >= 2


# ── 测试 ADMIN_COMMAND 权限过滤 ──────────────────────


class TestPermissionFilter:
    """测试 ADMIN_COMMAND 对非管理员用户的降级逻辑。"""

    async def test_admin_keeps_admin_command(
        self,
        router: IntentRouter,
        admin_ctx: UserContext,
    ) -> None:
        """管理员用户发出 ADMIN_COMMAND → 保持不变。"""
        mock_result = _make_routing_result(IntentType.ADMIN_COMMAND)
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("重启测试服务器", admin_ctx)

        assert result.intent == IntentType.ADMIN_COMMAND

    async def test_user_downgrade_admin_command(
        self,
        router: IntentRouter,
        user_ctx: UserContext,
    ) -> None:
        """普通用户发出 ADMIN_COMMAND → 降级为 RETRIEVAL。"""
        mock_result = _make_routing_result(IntentType.ADMIN_COMMAND)
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("重启测试服务器", user_ctx)

        assert result.intent == IntentType.RETRIEVAL
        assert "权限不足" in result.reasoning

    async def test_user_non_admin_intent_unaffected(
        self,
        router: IntentRouter,
        user_ctx: UserContext,
    ) -> None:
        """普通用户的非 ADMIN_COMMAND 意图不受权限过滤影响。"""
        mock_result = _make_routing_result(IntentType.RETRIEVAL)
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("公司年假政策是什么", user_ctx)

        assert result.intent == IntentType.RETRIEVAL


# ── 测试低置信度降级 ─────────────────────────────────


class TestConfidenceRules:
    """测试不同置信度区间的降级规则。"""

    async def test_high_confidence_no_downgrade(
        self,
        router: IntentRouter,
        user_ctx: UserContext,
    ) -> None:
        """confidence=0.9 → 不降级，low_confidence=False。"""
        mock_result = _make_result_with_subtasks(IntentType.TASK_GENERATION, confidence=0.9)
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("帮我写代码", user_ctx)

        assert result.intent == IntentType.TASK_GENERATION
        assert result.low_confidence is False
        assert len(result.sub_tasks) == 2

    async def test_medium_confidence_low_flag(
        self,
        router: IntentRouter,
        user_ctx: UserContext,
    ) -> None:
        """confidence=0.6 → low_confidence=True，sub_tasks 被清空。"""
        mock_result = _make_result_with_subtasks(IntentType.RETRIEVAL, confidence=0.6)
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("这个好像是查询", user_ctx)

        assert result.intent == IntentType.RETRIEVAL
        assert result.low_confidence is True
        assert result.sub_tasks == []

    async def test_very_low_confidence_force_group_chat(
        self,
        router: IntentRouter,
        user_ctx: UserContext,
    ) -> None:
        """confidence=0.3 → 强制降级 GROUP_CHAT + low_confidence=True。"""
        mock_result = _make_routing_result(
            IntentType.TASK_GENERATION, confidence=0.3,
        )
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("嗯？", user_ctx)

        assert result.intent == IntentType.GROUP_CHAT
        assert result.low_confidence is True
        assert "置信度过低" in result.reasoning
        assert result.sub_tasks == []

    async def test_boundary_confidence_070(
        self,
        router: IntentRouter,
        user_ctx: UserContext,
    ) -> None:
        """confidence=0.7（刚好不低于阈值）→ 不降级。"""
        mock_result = _make_result_with_subtasks(IntentType.RETRIEVAL, confidence=0.7)
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("查一下文档", user_ctx)

        assert result.low_confidence is False
        assert len(result.sub_tasks) == 2

    async def test_boundary_confidence_040(
        self,
        router: IntentRouter,
        user_ctx: UserContext,
    ) -> None:
        """confidence=0.4（刚好不低于极低阈值）→ low_confidence=True 但不强制 GROUP_CHAT。"""
        mock_result = _make_routing_result(IntentType.RETRIEVAL, confidence=0.4)
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("可能是查询吧", user_ctx)

        assert result.intent == IntentType.RETRIEVAL
        assert result.low_confidence is True


# ── 测试 LLM 异常降级兜底 ────────────────────────────


class TestFallback:
    """测试 LLM 调用失败时的兜底降级。"""

    async def test_llm_exception_fallback(
        self,
        router: IntentRouter,
        user_ctx: UserContext,
    ) -> None:
        """_call_llm 抛异常 → 返回 GROUP_CHAT + confidence=0.3。"""
        with patch.object(
            router,
            "_call_llm",
            new_callable=AsyncMock,
            side_effect=Exception("API timeout"),
        ):
            result = await router.route("任意消息", user_ctx)

        assert result.intent == IntentType.GROUP_CHAT
        assert result.confidence == pytest.approx(0.3)
        assert result.low_confidence is True
        assert "LLM调用失败" in result.reasoning

    async def test_llm_timeout_fallback(
        self,
        router: IntentRouter,
        user_ctx: UserContext,
    ) -> None:
        """模拟超时异常 → 同样触发兜底降级。"""
        import httpx

        with patch.object(
            router,
            "_call_llm",
            new_callable=AsyncMock,
            side_effect=httpx.TimeoutException("Connection timed out"),
        ):
            result = await router.route("超时消息", user_ctx)

        assert result.intent == IntentType.GROUP_CHAT
        assert result.low_confidence is True


# ── 测试子任务拆解 ───────────────────────────────────


class TestSubTaskDecomposition:
    """测试子任务拆解的完整性和 DAG 依赖。"""

    async def test_subtasks_preserved_high_confidence(
        self,
        router: IntentRouter,
        admin_ctx: UserContext,
    ) -> None:
        """高置信度时 sub_tasks 完整保留。"""
        mock_result = _make_result_with_subtasks(IntentType.TASK_GENERATION)
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("帮我写一份项目周报然后发到钉钉群", admin_ctx)

        assert len(result.sub_tasks) == 2
        assert result.sub_tasks[0].subtask_id == "subtask_1"
        assert result.sub_tasks[1].subtask_id == "subtask_2"

    async def test_subtask_dag_dependency(
        self,
        router: IntentRouter,
        admin_ctx: UserContext,
    ) -> None:
        """验证子任务间的 DAG 依赖关系。"""
        mock_result = _make_result_with_subtasks(IntentType.RETRIEVAL)
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("搜索论文然后总结", admin_ctx)

        # subtask_2 依赖 subtask_1
        assert result.sub_tasks[1].depends_on == ["subtask_1"]
        # subtask_1 无依赖
        assert result.sub_tasks[0].depends_on == []

    async def test_subtask_required_agents(
        self,
        router: IntentRouter,
        admin_ctx: UserContext,
    ) -> None:
        """验证子任务包含所需 Agent 类型。"""
        mock_result = _make_result_with_subtasks(IntentType.TASK_GENERATION)
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("写报告", admin_ctx)

        for st in result.sub_tasks:
            assert len(st.required_agents) > 0
            assert st.description

    async def test_subtasks_cleared_low_confidence(
        self,
        router: IntentRouter,
        user_ctx: UserContext,
    ) -> None:
        """低置信度时 sub_tasks 被清空。"""
        mock_result = _make_result_with_subtasks(IntentType.RETRIEVAL, confidence=0.5)
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("不确定的请求", user_ctx)

        assert result.sub_tasks == []
        assert result.low_confidence is True

    async def test_simple_request_no_subtasks(
        self,
        router: IntentRouter,
        admin_ctx: UserContext,
    ) -> None:
        """简单请求无子任务拆解。"""
        mock_result = _make_routing_result(IntentType.GROUP_CHAT, confidence=0.95)
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("哈哈", admin_ctx)

        assert result.sub_tasks == []


# ── 测试权限过滤 + 置信度规则组合 ─────────────────────


class TestCombinedRules:
    """测试权限过滤和置信度规则的组合场景。"""

    async def test_admin_command_low_confidence_user(
        self,
        router: IntentRouter,
        user_ctx: UserContext,
    ) -> None:
        """普通用户 + ADMIN_COMMAND + 低置信度 → 先权限降级再置信度降级。"""
        mock_result = _make_routing_result(
            IntentType.ADMIN_COMMAND, confidence=0.3,
        )
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("关闭服务", user_ctx)

        # 权限降级为 RETRIEVAL，再因置信度过低降级为 GROUP_CHAT
        assert result.intent == IntentType.GROUP_CHAT
        assert result.low_confidence is True
        assert "权限不足" in result.reasoning
        assert "置信度过低" in result.reasoning
