"""Tool registry for executable primitives.

This registry intentionally has no built-in local tools. In the OpenClaw /
Claude Code model, executable primitives should normally arrive through MCP
servers; skills and plugins are loaded by the capability layer, not registered
as tools.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import structlog

from agent_hub.core.models import ToolSpec
from agent_hub.core.risk import RiskPolicy, ToolProfile

logger = structlog.get_logger(__name__)

ToolFunction = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass
class RegisteredTool:
    """Registered executable primitive.

    ``func`` is optional because MCP-discovered tools are executable only through
    an MCP client connection. The local registry can still carry their schema for
    prompt/reporting purposes without pretending it can execute them directly.
    """

    spec: ToolSpec
    func: ToolFunction | None = None
    source: str = "manual"
    server_name: str | None = None


class ToolRegistry:
    """Registry of tool schemas and optional local adapters."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        name: str,
        description: str,
        params_schema: dict[str, Any],
        func: ToolFunction | None = None,
        requires_admin: bool = False,
        risk_level: str = "read",
        side_effect: bool = False,
        categories: list[str] | None = None,
        tool_profiles: list[str] | None = None,
        *,
        source: str = "manual",
        server_name: str | None = None,
    ) -> None:
        """Register a tool schema.

        Use ``func`` only for explicit in-process adapters. Built-in demo
        utilities are deliberately not registered here.
        """
        spec = ToolSpec(
            name=name,
            description=description,
            params_schema=params_schema,
            requires_admin=requires_admin,
            risk_level=risk_level,
            side_effect=side_effect,
            categories=categories or [],
            tool_profiles=tool_profiles or [],
        )
        self.register_spec(
            spec,
            func=func,
            source=source,
            server_name=server_name,
        )

    def register_spec(
        self,
        spec: ToolSpec,
        *,
        func: ToolFunction | None = None,
        source: str = "manual",
        server_name: str | None = None,
    ) -> None:
        """Register an already-normalized ``ToolSpec``."""
        self._tools[spec.name] = RegisteredTool(
            spec=spec,
            func=func,
            source=source,
            server_name=server_name,
        )
        logger.debug(
            "tool_registered",
            tool=spec.name,
            source=source,
            server_name=server_name,
            executable=func is not None,
        )

    def get(self, name: str) -> RegisteredTool | None:
        """Return a registered tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSpec]:
        """List all registered tool schemas."""
        return [rt.spec for rt in self._tools.values()]

    def list_tools_for_user(self, role: str) -> list[ToolSpec]:
        """List tools visible to a user role."""
        return [
            rt.spec
            for rt in self._tools.values()
            if not rt.spec.requires_admin or role == "admin"
        ]

    def list_tools_for_profile(self, profile: ToolProfile | str) -> list[ToolSpec]:
        """List tools allowed by a deterministic tool profile."""
        return [
            rt.spec
            for rt in self._tools.values()
            if RiskPolicy.tool_allowed(rt.spec, profile)
        ]

    def register_defaults(self) -> None:
        """Register built-in tools.

        No-op by design. Tool availability should come from MCP server discovery
        or from explicit adapters registered by application code.
        """
        logger.debug("tool_defaults_skipped", reason="mcp_only_tool_layer")
