"""跨层策略与权限契约。

定义：权限档案枚举、风险等级、策略决策输出、守卫检测结果。
这些类型是 runtime/policy 与其他层之间传递权限事实的公共格式。
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class RiskLevel(StrEnum):
    """风险等级枚举。"""

    SAFE = "safe"
    READ = "read"
    WRITE = "write"
    SHARE = "share"
    ADMIN = "admin"
    SUSPICIOUS = "suspicious"
    BLOCKED = "blocked"


class PermissionProfile(StrEnum):
    """工具权限档案（最小权限原则，从低到高）。"""

    READ_ONLY = "read_only"
    BASELINE_ARTIFACT = "baseline_artifact"
    TOOL_ASSISTED_SAFE = "tool_assisted_safe"
    ADMIN_OPS = "admin_ops"


class PolicyDecision(BaseModel):
    """策略引擎输出的权限决策。"""

    model_config = ConfigDict(extra="forbid")

    permission_profile: PermissionProfile
    risk_level: RiskLevel
    requires_admin: bool = False
    requires_approval: bool = False
    reason: str = ""


class GuardResult(BaseModel):
    """Prompt 注入检测结果。"""

    model_config = ConfigDict(extra="forbid")

    is_safe: bool
    risk_level: str = "safe"
    matched_rules: list[str] = []
    llm_analysis: str | None = None
    confidence: float = 1.0


__all__ = [
    "GuardResult",
    "PermissionProfile",
    "PolicyDecision",
    "RiskLevel",
]
