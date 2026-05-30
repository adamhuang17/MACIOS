"""Binding / RiskPolicy / plan validator contract tests."""

from __future__ import annotations

from agent_hub.agents.registry import ToolRegistry
from agent_hub.core.binding import BindingRule, SourceBindingResolver
from agent_hub.core.enums import ExecutionMode, UserRole
from agent_hub.core.models import (
    GuardResult,
    RoutingDecision,
    SourceChatType,
    SourceContext,
    SubTask,
    UserContext,
)
from agent_hub.core.plan_validator import SubTaskPlanValidator
from agent_hub.core.risk import RiskPolicy, ToolProfile


def test_binding_priority_message_over_chat() -> None:
    resolver = SourceBindingResolver([
        BindingRule(agent_id="chat_agent", channel="feishu", chat_id="oc_1"),
        BindingRule(agent_id="message_agent", channel="feishu", message_id="om_1"),
    ])
    result = resolver.resolve(
        SourceContext(
            channel="feishu",
            chat_id="oc_1",
            chat_type=SourceChatType.GROUP,
            message_id="om_1",
            sender_id="ou_1",
        ),
        UserContext(user_id="ou_1", role=UserRole.USER),
    )

    assert result.agent_id == "message_agent"
    assert result.route_source == "message"
    assert result.session_key == "agent:message_agent:feishu:group:oc_1"


def test_risk_policy_does_not_let_user_escalate_to_admin() -> None:
    binding = SourceBindingResolver().resolve(
        SourceContext(channel="api", sender_id="u1", chat_type=SourceChatType.DIRECT),
        UserContext(user_id="u1", role=UserRole.USER),
    )
    routing = RoutingDecision(
        mode=ExecutionMode.DELEGATE,
        confidence=0.9,
        reasoning="ops request",
        capabilities=["openclaw"],
    )
    decision = RiskPolicy().assess(
        user_context=UserContext(user_id="u1", role=UserRole.USER),
        source_context=SourceContext(
            channel="api",
            sender_id="u1",
            chat_type=SourceChatType.DIRECT,
        ),
        binding=binding,
        routing=routing,
        guard_result=GuardResult(is_safe=True),
    )

    assert decision.requires_admin is True
    assert decision.tool_profile is ToolProfile.READ_ONLY
    assert decision.risk_level == "admin"


def test_binding_profile_caps_requested_tool_profile() -> None:
    resolver = SourceBindingResolver([
        BindingRule(channel="api", agent_id="main", tool_profile="read_only"),
    ])
    user = UserContext(user_id="u1", role=UserRole.ADMIN)
    source = SourceContext(channel="api", sender_id="u1", chat_type=SourceChatType.DIRECT)
    binding = resolver.resolve(source, user)
    routing = RoutingDecision(
        mode=ExecutionMode.ACT,
        confidence=0.9,
        reasoning="tool request",
        capabilities=["tool"],
    )

    decision = RiskPolicy().assess(
        user_context=user,
        source_context=source,
        binding=binding,
        routing=routing,
        guard_result=GuardResult(is_safe=True),
    )

    assert decision.tool_profile is ToolProfile.READ_ONLY


def test_risk_policy_ignores_routing_approval_flag() -> None:
    resolver = SourceBindingResolver()
    user = UserContext(user_id="u1", role=UserRole.ADMIN)
    source = SourceContext(channel="api", sender_id="u1", chat_type=SourceChatType.DIRECT)
    binding = resolver.resolve(source, user)
    routing = RoutingDecision(
        mode=ExecutionMode.QA,
        confidence=0.9,
        reasoning="routing flags should not drive policy",
        requires_approval=True,
    )

    decision = RiskPolicy().assess(
        user_context=user,
        source_context=source,
        binding=binding,
        routing=routing,
        guard_result=GuardResult(is_safe=True),
    )

    assert decision.requires_admin is False
    assert decision.requires_approval is False
    assert decision.tool_profile is ToolProfile.READ_ONLY


def test_tool_registry_profile_filtering() -> None:
    registry = ToolRegistry()
    registry.register_defaults()
    registry.register(
        "mcp__docs__search",
        "MCP docs server search",
        {"query": {"type": "string"}},
        source="mcp",
        server_name="docs",
    )
    registry.register(
        "mcp__ops__deploy",
        "MCP ops server deploy",
        {"target": {"type": "string"}},
        requires_admin=True,
        risk_level="admin",
        side_effect=True,
        source="mcp",
        server_name="ops",
    )

    read_only_names = {tool.name for tool in registry.list_tools_for_profile("read_only")}
    admin_names = {tool.name for tool in registry.list_tools_for_profile("admin_ops")}

    assert "mcp__docs__search" in read_only_names
    assert "mcp__ops__deploy" not in read_only_names
    assert "mcp__ops__deploy" in admin_names


def test_subtask_plan_validator_detects_cycle() -> None:
    result = SubTaskPlanValidator().validate([
        SubTask(
            subtask_id="a",
            description="A",
            required_agents=["llm_agent"],
            depends_on=["b"],
        ),
        SubTask(
            subtask_id="b",
            description="B",
            required_agents=["llm_agent"],
            depends_on=["a"],
        ),
    ])

    assert result.valid is False
    assert any("环" in error for error in result.errors)
