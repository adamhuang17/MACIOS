"""Tool profile and risk policy resolution（规范定义位置）。

从 agent_hub.core.risk 迁移至此；core.risk 保留为 re-export 向后兼容层。

The router may suggest capabilities, but it cannot grant permissions. This
module computes the final tool profile, admin requirement, approval requirement,
and risk level from deterministic facts.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel

from agent_hub.contracts.capability import ToolSpec
from agent_hub.contracts.execution import ExecutionMode, RoutingDecision
from agent_hub.contracts.identity import SourceChatType, SourceContext, UserContext, UserRole
from agent_hub.contracts.policy import GuardResult

if TYPE_CHECKING:
    from agent_hub.runtime.policy.source_binding import BindingResult


class ToolProfile(StrEnum):
    """Fixed tool profiles, ordered from least to most privileged."""

    READ_ONLY = "read_only"
    BASELINE_ARTIFACT = "baseline_artifact"
    TOOL_ASSISTED_SAFE = "tool_assisted_safe"
    ADMIN_OPS = "admin_ops"


class RiskDecision(BaseModel):
    """RiskPolicy output."""

    tool_profile: ToolProfile
    risk_level: str
    requires_admin: bool = False
    requires_approval: bool = False
    reason: str = ""


class RiskPolicy:
    """Compute effective permissions without trusting LLM output."""

    _PROFILE_RANK: dict[ToolProfile, int] = {
        ToolProfile.READ_ONLY: 0,
        ToolProfile.BASELINE_ARTIFACT: 1,
        ToolProfile.TOOL_ASSISTED_SAFE: 2,
        ToolProfile.ADMIN_OPS: 3,
    }

    _WRITE_CAPABILITIES = {"file_ingest", "memory_write", "artifact", "write", "share"}
    _ADMIN_CAPABILITIES = {"admin", "system", "ops"}

    def assess(
        self,
        *,
        user_context: UserContext,
        source_context: SourceContext,
        binding: BindingResult,
        routing: RoutingDecision,
        guard_result: GuardResult | None = None,
    ) -> RiskDecision:
        """Resolve the final risk decision for a routed request."""
        guard_risk = (guard_result.risk_level if guard_result else "safe").lower()
        capabilities = {cap.lower() for cap in routing.capabilities}
        needs_admin = self._requires_admin(routing, capabilities)
        requested_profile = self._requested_profile(routing, capabilities, needs_admin)

        binding_cap = parse_tool_profile(binding.tool_profile)
        if binding_cap is not None:
            requested_profile = self._min_profile(requested_profile, binding_cap)

        if needs_admin and user_context.role != UserRole.ADMIN:
            profile = ToolProfile.READ_ONLY
        else:
            profile = requested_profile

        risk_level = self._risk_level(capabilities, needs_admin, guard_risk)
        requires_approval = self._requires_approval(
            risk_level=risk_level,
            guard_risk=guard_risk,
            source_context=source_context,
        )

        reason_parts = [
            f"profile={profile.value}",
            f"risk={risk_level}",
            f"binding={binding.matched_by}",
        ]
        if needs_admin:
            reason_parts.append("admin_capability")
        if guard_risk != "safe":
            reason_parts.append(f"guard={guard_risk}")

        return RiskDecision(
            tool_profile=profile,
            risk_level=risk_level,
            requires_admin=needs_admin,
            requires_approval=requires_approval,
            reason="; ".join(reason_parts),
        )

    def apply(
        self,
        *,
        routing: RoutingDecision,
        binding: BindingResult,
        decision: RiskDecision,
    ) -> RoutingDecision:
        """Return a RoutingDecision patched with binding and risk facts."""
        return routing.model_copy(update={
            "route_source": binding.route_source,
            "agent_id": binding.agent_id,
            "session_key": binding.session_key,
            "tool_profile": decision.tool_profile.value,
            "risk_level": decision.risk_level,
            "requires_approval": decision.requires_approval,
        })

    @classmethod
    def tool_allowed(cls, spec: ToolSpec, profile: ToolProfile | str) -> bool:
        """Return whether a tool is available under a profile."""
        resolved = parse_tool_profile(profile) or ToolProfile.READ_ONLY
        if resolved is ToolProfile.ADMIN_OPS:
            return True
        if spec.tool_profiles:
            return resolved.value in spec.tool_profiles
        if spec.requires_admin or spec.risk_level == "admin":
            return False
        if spec.risk_level == "read":
            return True
        if spec.risk_level in {"artifact", "share"}:
            return resolved in {
                ToolProfile.BASELINE_ARTIFACT,
                ToolProfile.TOOL_ASSISTED_SAFE,
            }
        if spec.risk_level == "write":
            return resolved is ToolProfile.TOOL_ASSISTED_SAFE and not spec.side_effect
        return not spec.side_effect

    def _requested_profile(
        self,
        routing: RoutingDecision,
        capabilities: set[str],
        needs_admin: bool,
    ) -> ToolProfile:
        if needs_admin:
            return ToolProfile.ADMIN_OPS
        if routing.mode == ExecutionMode.ACT or "tool" in capabilities:
            return ToolProfile.TOOL_ASSISTED_SAFE
        if capabilities & self._WRITE_CAPABILITIES:
            return ToolProfile.BASELINE_ARTIFACT
        return ToolProfile.READ_ONLY

    def _requires_admin(self, routing: RoutingDecision, capabilities: set[str]) -> bool:
        return (
            routing.mode == ExecutionMode.DELEGATE
            or bool(capabilities & self._ADMIN_CAPABILITIES)
        )

    def _requires_approval(
        self,
        *,
        risk_level: str,
        guard_risk: str,
        source_context: SourceContext,
    ) -> bool:
        if guard_risk == "suspicious":
            return True
        if risk_level in {"admin", "share"}:
            return True
        return (
            risk_level == "write"
            and source_context.chat_type is not SourceChatType.DIRECT
        )

    def _risk_level(
        self,
        capabilities: set[str],
        needs_admin: bool,
        guard_risk: str,
    ) -> str:
        if guard_risk == "blocked":
            return "blocked"
        if needs_admin:
            return "admin"
        if "share" in capabilities:
            return "share"
        if capabilities & self._WRITE_CAPABILITIES:
            return "write"
        if guard_risk == "suspicious":
            return "suspicious"
        return "read"

    def _min_profile(self, left: ToolProfile, right: ToolProfile) -> ToolProfile:
        return left if self._PROFILE_RANK[left] <= self._PROFILE_RANK[right] else right


def parse_tool_profile(value: ToolProfile | str | None) -> ToolProfile | None:
    """Parse a profile id, returning None for unknown values."""
    if isinstance(value, ToolProfile):
        return value
    if not value:
        return None
    try:
        return ToolProfile(str(value))
    except ValueError:
        return None


__all__ = [
    "RiskDecision",
    "RiskPolicy",
    "ToolProfile",
    "parse_tool_profile",
]
