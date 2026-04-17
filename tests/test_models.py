"""核心数据模型和 BaseAgent 的单元测试。"""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from agent_hub.core import (
    AgentResult,
    BaseAgent,
    IntentType,
    MemoryEntry,
    RoutingResult,
    SubTask,
    TaskInput,
    TaskOutput,
    ToolSpec,
    UserContext,
    UserRole,
)


# ── 枚举测试 ─────────────────────────────────────────


class TestEnums:
    def test_intent_type_values(self) -> None:
        assert IntentType.TASK_GENERATION == "task_generation"
        assert IntentType.RETRIEVAL == "retrieval"
        assert IntentType.TOOL_EXECUTION == "tool_execution"
        assert IntentType.FILE_PROCESSING == "file_processing"
        assert IntentType.ADMIN_COMMAND == "admin_command"
        assert IntentType.GROUP_CHAT == "group_chat"
        assert IntentType.REFLECTION == "reflection"

    def test_intent_type_count(self) -> None:
        assert len(IntentType) == 7

    def test_user_role_values(self) -> None:
        assert UserRole.ADMIN == "admin"
        assert UserRole.USER == "user"


# ── UserContext 测试 ──────────────────────────────────


class TestUserContext:
    def test_minimal(self) -> None:
        ctx = UserContext(user_id="u001", role=UserRole.ADMIN, channel="dingtalk")
        assert ctx.user_id == "u001"
        assert ctx.role == UserRole.ADMIN
        assert ctx.channel == "dingtalk"
        assert ctx.group_id is None
        assert ctx.is_private is False

    def test_full(self) -> None:
        ctx = UserContext(
            user_id="u002",
            role=UserRole.USER,
            channel="qq",
            group_id="g100",
            is_private=True,
        )
        assert ctx.group_id == "g100"
        assert ctx.is_private is True


# ── TaskInput 测试 ────────────────────────────────────


class TestTaskInput:
    def test_auto_trace_id(self) -> None:
        task = TaskInput(
            user_context=UserContext(
                user_id="u001", role=UserRole.USER, channel="dingtalk"
            ),
            raw_message="你好",
        )
        assert len(task.trace_id) == 36  # UUID 格式
        assert isinstance(task.timestamp, datetime)
        assert task.attachments == []
        assert task.parent_trace_id is None

    def test_with_parent_trace(self) -> None:
        task = TaskInput(
            user_context=UserContext(
                user_id="u001", role=UserRole.USER, channel="qq"
            ),
            raw_message="子任务",
            parent_trace_id="parent-trace-123",
        )
        assert task.parent_trace_id == "parent-trace-123"

    def test_serialization_roundtrip(self) -> None:
        task = TaskInput(
            user_context=UserContext(
                user_id="u001", role=UserRole.USER, channel="dingtalk"
            ),
            raw_message="测试序列化",
        )
        data = task.model_dump()
        restored = TaskInput.model_validate(data)
        assert restored.trace_id == task.trace_id
        assert restored.raw_message == task.raw_message


# ── SubTask & RoutingResult 测试 ─────────────────────


class TestSubTask:
    def test_defaults(self) -> None:
        st = SubTask(
            subtask_id="st-1",
            description="检索相关文档",
            required_agents=["rag_agent"],
        )
        assert st.priority == 1
        assert st.depends_on == []


class TestRoutingResult:
    def test_valid(self) -> None:
        rr = RoutingResult(
            intent=IntentType.RETRIEVAL,
            confidence=0.92,
            reasoning="用户请求查找资料",
        )
        assert rr.low_confidence is False
        assert rr.sub_tasks == []

    def test_low_confidence_flag(self) -> None:
        rr = RoutingResult(
            intent=IntentType.TASK_GENERATION,
            confidence=0.55,
            reasoning="不确定意图",
            low_confidence=True,
        )
        assert rr.low_confidence is True

    def test_confidence_range_validation(self) -> None:
        with pytest.raises(ValueError):
            RoutingResult(
                intent=IntentType.RETRIEVAL,
                confidence=1.5,
                reasoning="超范围",
            )


# ── ToolSpec 测试 ─────────────────────────────────────


class TestToolSpec:
    def test_basic(self) -> None:
        tool = ToolSpec(
            name="web_search",
            description="搜索网页",
            params_schema={"query": {"type": "string"}},
        )
        assert tool.requires_admin is False

    def test_admin_tool(self) -> None:
        tool = ToolSpec(
            name="system_restart",
            description="重启系统",
            requires_admin=True,
        )
        assert tool.requires_admin is True


# ── AgentResult 测试 ──────────────────────────────────


class TestAgentResult:
    def test_success(self) -> None:
        ar = AgentResult(
            agent_name="echo",
            success=True,
            output="hello",
            duration_ms=42,
            trace_id="t-001",
        )
        assert ar.error is None

    def test_failure(self) -> None:
        ar = AgentResult(
            agent_name="echo",
            success=False,
            error="timeout",
            duration_ms=5000,
            trace_id="t-002",
        )
        assert ar.output is None


# ── MemoryEntry 测试 ──────────────────────────────────


class TestMemoryEntry:
    def test_short_term(self) -> None:
        me = MemoryEntry(
            content="用户偏好中文",
            memory_type="short_term",
            session_id="s-001",
            tags=["preference"],
        )
        assert isinstance(me.created_at, datetime)

    def test_long_term(self) -> None:
        me = MemoryEntry(content="重要知识点", memory_type="long_term")
        assert me.session_id is None
        assert me.tags == []


# ── TaskOutput 测试 ───────────────────────────────────


class TestTaskOutput:
    def test_success_output(self) -> None:
        to = TaskOutput(
            trace_id="t-001",
            user_id="u001",
            response="处理完成",
            total_duration_ms=150,
        )
        assert to.status == "success"
        assert to.agent_results == []
        assert to.memory_saved == []

    def test_failed_output(self) -> None:
        to = TaskOutput(
            trace_id="t-002",
            user_id="u001",
            response="处理失败",
            total_duration_ms=300,
            status="failed",
        )
        assert to.status == "failed"


# ── BaseAgent 测试 ───────────────────────────────────


class _EchoAgent(BaseAgent):
    """用于测试的 Echo Agent。"""

    def __init__(self) -> None:
        super().__init__(name="echo", description="原样返回用户输入")

    async def run(
        self, subtask: SubTask, session_id: str, user_id: str,
    ) -> AgentResult:
        return AgentResult(
            agent_name=self.name,
            success=True,
            output=subtask.description,
            duration_ms=0,
            trace_id=subtask.subtask_id,
        )


class _FailingAgent(BaseAgent):
    """用于测试异常捕获的 Agent。"""

    def __init__(self) -> None:
        super().__init__(name="failing", description="总是抛出异常")

    async def run(
        self, subtask: SubTask, session_id: str, user_id: str,
    ) -> AgentResult:
        raise RuntimeError("模拟故障")


class TestBaseAgent:
    @pytest.fixture()
    def subtask(self) -> SubTask:
        return SubTask(
            subtask_id="st-test-001",
            description="测试消息",
            required_agents=["echo"],
        )

    def test_repr(self) -> None:
        agent = _EchoAgent()
        assert "echo" in repr(agent)

    @pytest.mark.asyncio
    async def test_execute_success(self, subtask: SubTask) -> None:
        agent = _EchoAgent()
        result = await agent.execute(subtask, "sess-001", "u001")
        assert result.success is True
        assert result.output == "测试消息"
        assert result.agent_name == "echo"
        assert result.duration_ms >= 0
        assert result.trace_id == subtask.subtask_id

    @pytest.mark.asyncio
    async def test_execute_failure(self, subtask: SubTask) -> None:
        agent = _FailingAgent()
        result = await agent.execute(subtask, "sess-001", "u001")
        assert result.success is False
        assert result.error is not None
        assert result.agent_name == "failing"
        assert result.duration_ms >= 0
