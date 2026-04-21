"""路由决策器单元测试。

mock 掉 LLM 调用（_call_llm），单独测试路由逻辑：
- 6 种执行模式分类
- requires_admin 约束字段
- 低置信度降级
- LLM 异常降级兜底
- 子任务计划（plan）完整性
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agent_hub.core.enums import ExecutionMode, UserRole
from agent_hub.core.models import RoutingDecision, SubTask, UserContext
from agent_hub.core.router import DecisionRouter


# ── Fixtures ─────────────────────────────────────────


@pytest.fixture()
def router() -> DecisionRouter:
    """创建一个路由器实例（不会真正调用 LLM）。"""
    return DecisionRouter(client=AsyncMock())


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


def _make_routing_decision(
    mode: ExecutionMode,
    confidence: float = 0.92,
    reasoning: str = "测试推理",
    plan: list[SubTask] | None = None,
    requires_admin: bool = False,
    capabilities: list[str] | None = None,
) -> RoutingDecision:
    return RoutingDecision(
        mode=mode,
        confidence=confidence,
        reasoning=reasoning,
        plan=plan or [],
        requires_admin=requires_admin,
        capabilities=capabilities or [],
    )


def _make_decision_with_plan(
    mode: ExecutionMode,
    confidence: float = 0.92,
) -> RoutingDecision:
    return RoutingDecision(
        mode=mode,
        confidence=confidence,
        reasoning="复杂请求，拆解子任务",
        plan=[
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


# ── 测试执行模式分类 ──────────────────────────────────


class TestRouteClassification:
    """使用典型 query 测试路由决策结果。"""

    _CASES: list[tuple[str, ExecutionMode, bool]] = [
        # (query, 期望模式, 是否含计划)
        ("帮我写一个快速排序算法", ExecutionMode.PLAN, False),
        ("公司年假政策是什么", ExecutionMode.QA, False),
        ("帮我算一下北京今天天气", ExecutionMode.ACT, False),
        ("帮我分析这个Excel文件", ExecutionMode.ACT, False),
        ("重启测试服务器", ExecutionMode.DELEGATE, False),
        ("哈哈哈今天天气真好", ExecutionMode.IGNORE, False),
        ("你刚才的回答不对，重新想想", ExecutionMode.REPAIR, False),
        ("帮我写一份项目周报然后发到钉钉群", ExecutionMode.PLAN, True),
        ("搜索一下关于LLM的论文然后总结要点", ExecutionMode.QA, True),
        ("这个文件里有什么内容", ExecutionMode.ACT, False),
    ]

    @pytest.mark.parametrize(
        ("query", "expected_mode", "has_plan"),
        _CASES,
        ids=[
            "write_quicksort",
            "vacation_policy",
            "check_weather",
            "analyze_excel",
            "restart_server",
            "casual_chat",
            "repair",
            "complex_report",
            "complex_search",
            "file_query",
        ],
    )
    async def test_classification(
        self,
        router: DecisionRouter,
        admin_ctx: UserContext,
        query: str,
        expected_mode: ExecutionMode,
        has_plan: bool,
    ) -> None:
        """验证各 query 被路由到正确的执行模式。"""
        if has_plan:
            mock_result = _make_decision_with_plan(expected_mode)
        else:
            mock_result = _make_routing_decision(expected_mode)

        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route(query, admin_ctx)

        assert result.mode == expected_mode
        assert 0.0 <= result.confidence <= 1.0
        assert result.reasoning

        if has_plan:
            assert len(result.plan) >= 2


# ── 测试 requires_admin 约束字段 ─────────────────────


class TestRequiresAdmin:
    """测试 requires_admin 字段的路由行为（权限校验已移至 Pipeline）。"""

    async def test_admin_decision_carries_requires_admin(
        self,
        router: DecisionRouter,
        admin_ctx: UserContext,
    ) -> None:
        """delegate 模式 + requires_admin=True → 原样返回，由 Pipeline 做权限校验。"""
        mock_result = _make_routing_decision(
            ExecutionMode.DELEGATE, requires_admin=True,
        )
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("重启测试服务器", admin_ctx)

        assert result.mode == ExecutionMode.DELEGATE
        assert result.requires_admin is True

    async def test_user_delegate_decision_not_downgraded_by_router(
        self,
        router: DecisionRouter,
        user_ctx: UserContext,
    ) -> None:
        """普通用户发出 delegate 请求 → Router 不降级，由 Pipeline 处理权限。"""
        mock_result = _make_routing_decision(
            ExecutionMode.DELEGATE, requires_admin=True,
        )
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("重启测试服务器", user_ctx)

        # Router 不再做权限降级，原样返回，Pipeline 负责
        assert result.mode == ExecutionMode.DELEGATE
        assert result.requires_admin is True

    async def test_non_admin_mode_unaffected(
        self,
        router: DecisionRouter,
        user_ctx: UserContext,
    ) -> None:
        """普通 qa 模式不携带 requires_admin → 正常返回。"""
        mock_result = _make_routing_decision(ExecutionMode.QA)
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("公司年假政策是什么", user_ctx)

        assert result.mode == ExecutionMode.QA
        assert result.requires_admin is False


# ── 测试低置信度降级 ─────────────────────────────────


class TestConfidenceRules:
    """测试不同置信度区间的降级规则。"""

    async def test_high_confidence_no_downgrade(
        self,
        router: DecisionRouter,
        user_ctx: UserContext,
    ) -> None:
        """confidence=0.9 → 不降级，low_confidence=False。"""
        mock_result = _make_decision_with_plan(ExecutionMode.PLAN, confidence=0.9)
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("帮我写代码", user_ctx)

        assert result.mode == ExecutionMode.PLAN
        assert result.low_confidence is False
        assert len(result.plan) == 2

    async def test_medium_confidence_low_flag(
        self,
        router: DecisionRouter,
        user_ctx: UserContext,
    ) -> None:
        """confidence=0.6 → low_confidence=True，plan 被清空。"""
        mock_result = _make_decision_with_plan(ExecutionMode.QA, confidence=0.6)
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("这个好像是查询", user_ctx)

        assert result.mode == ExecutionMode.QA
        assert result.low_confidence is True
        assert result.plan == []

    async def test_very_low_confidence_force_ignore(
        self,
        router: DecisionRouter,
        user_ctx: UserContext,
    ) -> None:
        """confidence=0.3 → 强制降级 IGNORE + low_confidence=True。"""
        mock_result = _make_routing_decision(
            ExecutionMode.PLAN, confidence=0.3,
        )
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("嗯？", user_ctx)

        assert result.mode == ExecutionMode.IGNORE
        assert result.low_confidence is True
        assert "置信度过低" in result.reasoning
        assert result.plan == []

    async def test_boundary_confidence_070(
        self,
        router: DecisionRouter,
        user_ctx: UserContext,
    ) -> None:
        """confidence=0.7（刚好不低于阈值）→ 不降级。"""
        mock_result = _make_decision_with_plan(ExecutionMode.QA, confidence=0.7)
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("查一下文档", user_ctx)

        assert result.low_confidence is False
        assert len(result.plan) == 2

    async def test_boundary_confidence_040(
        self,
        router: DecisionRouter,
        user_ctx: UserContext,
    ) -> None:
        """confidence=0.4（刚好不低于极低阈值）→ low_confidence=True 但不强制 IGNORE。"""
        mock_result = _make_routing_decision(ExecutionMode.QA, confidence=0.4)
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("可能是查询吧", user_ctx)

        assert result.mode == ExecutionMode.QA
        assert result.low_confidence is True


# ── 测试 LLM 异常降级兜底 ────────────────────────────


class TestFallback:
    """测试 LLM 调用失败时的兜底降级。"""

    async def test_llm_exception_fallback(
        self,
        router: DecisionRouter,
        user_ctx: UserContext,
    ) -> None:
        """_call_llm 抛异常 → 返回 IGNORE + confidence=0.3。"""
        with patch.object(
            router,
            "_call_llm",
            new_callable=AsyncMock,
            side_effect=Exception("API timeout"),
        ):
            result = await router.route("任意消息", user_ctx)

        assert result.mode == ExecutionMode.IGNORE
        assert result.confidence == pytest.approx(0.3)
        assert result.low_confidence is True
        assert "LLM调用失败" in result.reasoning

    async def test_llm_timeout_fallback(
        self,
        router: DecisionRouter,
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

        assert result.mode == ExecutionMode.IGNORE
        assert result.low_confidence is True


# ── 测试子任务计划 ────────────────────────────────────


class TestPlanDecomposition:
    """测试 plan 字段的完整性和 DAG 依赖。"""

    async def test_plan_preserved_high_confidence(
        self,
        router: DecisionRouter,
        admin_ctx: UserContext,
    ) -> None:
        """高置信度时 plan 完整保留。"""
        mock_result = _make_decision_with_plan(ExecutionMode.PLAN)
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("帮我写一份项目周报然后发到钉钉群", admin_ctx)

        assert len(result.plan) == 2
        assert result.plan[0].subtask_id == "subtask_1"
        assert result.plan[1].subtask_id == "subtask_2"

    async def test_plan_dag_dependency(
        self,
        router: DecisionRouter,
        admin_ctx: UserContext,
    ) -> None:
        """验证子任务间的 DAG 依赖关系。"""
        mock_result = _make_decision_with_plan(ExecutionMode.QA)
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("搜索论文然后总结", admin_ctx)

        # subtask_2 依赖 subtask_1
        assert result.plan[1].depends_on == ["subtask_1"]
        # subtask_1 无依赖
        assert result.plan[0].depends_on == []

    async def test_plan_required_agents(
        self,
        router: DecisionRouter,
        admin_ctx: UserContext,
    ) -> None:
        """验证子任务包含所需 Agent 类型。"""
        mock_result = _make_decision_with_plan(ExecutionMode.PLAN)
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("写报告", admin_ctx)

        for st in result.plan:
            assert len(st.required_agents) > 0
            assert st.description

    async def test_plan_cleared_low_confidence(
        self,
        router: DecisionRouter,
        user_ctx: UserContext,
    ) -> None:
        """低置信度时 plan 被清空。"""
        mock_result = _make_decision_with_plan(ExecutionMode.QA, confidence=0.5)
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("不确定的请求", user_ctx)

        assert result.plan == []
        assert result.low_confidence is True

    async def test_ignore_mode_no_plan(
        self,
        router: DecisionRouter,
        admin_ctx: UserContext,
    ) -> None:
        """IGNORE 模式不含子任务计划。"""
        mock_result = _make_routing_decision(ExecutionMode.IGNORE, confidence=0.95)
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("哈哈", admin_ctx)

        assert result.plan == []


# ── 测试 capabilities 字段 ────────────────────────────


class TestCapabilities:
    """测试 capabilities 能力标签字段。"""

    async def test_tool_capability(
        self,
        router: DecisionRouter,
        user_ctx: UserContext,
    ) -> None:
        """act 模式应携带 tool 能力标签。"""
        mock_result = _make_routing_decision(
            ExecutionMode.ACT, capabilities=["tool"],
        )
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("帮我算 2+3", user_ctx)

        assert "tool" in result.capabilities

    async def test_retrieval_capability(
        self,
        router: DecisionRouter,
        user_ctx: UserContext,
    ) -> None:
        """qa 模式应携带 retrieval 能力标签。"""
        mock_result = _make_routing_decision(
            ExecutionMode.QA, capabilities=["retrieval"],
        )
        with patch.object(
            router, "_call_llm", new_callable=AsyncMock, return_value=mock_result,
        ):
            result = await router.route("RAG 是什么", user_ctx)

        assert "retrieval" in result.capabilities

