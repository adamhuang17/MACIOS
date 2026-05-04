"""SkillRegistry：Pilot 业务技能的注册、查询、dry-run 与 invoke 入口。

设计要点：
- Skill 不持有状态，不直接修改实体快照，不直接发事件；
- registry 只做存在性、scope、dry-run 支持、timeout 等策略检查；
- 写/分享类 skill 默认 ``requires_approval=True``；
- 未知 skill 抛 :class:`SkillNotFound`；
  不支持 dry-run 时调用 ``dry_run`` 抛 :class:`DryRunUnsupported`。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger(__name__)


SideEffect = Literal["read", "write", "share"]


class SkillSpec(BaseModel):
    """单个技能的规范描述。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    skill_name: str
    description: str = ""
    side_effect: SideEffect = "read"
    requires_approval: bool = False
    supports_dry_run: bool = True
    scopes: list[str] = Field(default_factory=list)
    idempotency_key_fields: list[str] = Field(default_factory=list)
    timeout_ms: int = 30_000
    params_schema: dict[str, Any] = Field(default_factory=dict)
    result_schema: dict[str, Any] = Field(default_factory=dict)


class SkillInvocation(BaseModel):
    """对单次 skill 调用的入参描述。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    skill_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = False
    idempotency_key: str | None = None
    actor_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillResult(BaseModel):
    """skill 调用的统一返回结构。"""

    model_config = ConfigDict(extra="forbid")

    skill_name: str
    success: bool
    output: dict[str, Any] = Field(default_factory=dict)
    artifact_payload: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: int = 0
    dry_run: bool = False


SkillFunc = Callable[[SkillInvocation], Awaitable[SkillResult]]


@dataclass
class RegisteredSkill:
    spec: SkillSpec
    func: SkillFunc


class SkillNotFound(KeyError):  # noqa: N818
    """Skill 未在 registry 中注册。"""


class DryRunUnsupported(RuntimeError):  # noqa: N818
    """Skill 不支持 dry-run 但调用了 ``dry_run``。"""


class SkillRegistry:
    """Pilot 技能注册表。

    与 :class:`agent_hub.agents.registry.ToolRegistry` 完全独立，
    专门服务 Pilot 业务编排：携带 side_effect / approval / dry-run 策略。
    """

    def __init__(self) -> None:
        self._skills: dict[str, RegisteredSkill] = {}

    def register(
        self,
        spec: SkillSpec,
        func: SkillFunc,
        *,
        allow_overwrite: bool = False,
    ) -> None:
        if not allow_overwrite and spec.skill_name in self._skills:
            msg = f"skill 已注册: {spec.skill_name}"
            raise ValueError(msg)
        self._skills[spec.skill_name] = RegisteredSkill(spec=spec, func=func)
        logger.debug(
            "pilot.skill_registered",
            skill=spec.skill_name,
            side_effect=spec.side_effect,
            requires_approval=spec.requires_approval,
        )

    def get(self, skill_name: str) -> RegisteredSkill | None:
        return self._skills.get(skill_name)

    def must_get(self, skill_name: str) -> RegisteredSkill:
        rs = self._skills.get(skill_name)
        if rs is None:
            raise SkillNotFound(skill_name)
        return rs

    def list_skills(self) -> list[SkillSpec]:
        return [rs.spec for rs in self._skills.values()]

    def list_for_scopes(self, scopes: list[str]) -> list[SkillSpec]:
        """返回所有 scopes 都被授予的技能；无 scopes 限制的技能恒可见。"""
        granted = set(scopes)
        return [
            rs.spec
            for rs in self._skills.values()
            if not rs.spec.scopes or set(rs.spec.scopes).issubset(granted)
        ]

    async def dry_run(self, invocation: SkillInvocation) -> SkillResult:
        rs = self.must_get(invocation.skill_name)
        if not rs.spec.supports_dry_run:
            raise DryRunUnsupported(invocation.skill_name)
        forced = invocation.model_copy(update={"dry_run": True})
        return await self._invoke_with_policy(rs, forced)

    async def invoke(self, invocation: SkillInvocation) -> SkillResult:
        rs = self.must_get(invocation.skill_name)
        return await self._invoke_with_policy(rs, invocation)

    async def _invoke_with_policy(
        self,
        rs: RegisteredSkill,
        invocation: SkillInvocation,
    ) -> SkillResult:
        timeout_s = rs.spec.timeout_ms / 1000.0
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(rs.func(invocation), timeout=timeout_s)
        except TimeoutError:
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.warning(
                "pilot.skill_timeout",
                skill=rs.spec.skill_name,
                timeout_ms=rs.spec.timeout_ms,
            )
            return SkillResult(
                skill_name=rs.spec.skill_name,
                success=False,
                error=f"timeout after {rs.spec.timeout_ms}ms",
                duration_ms=duration_ms,
                dry_run=invocation.dry_run,
            )
        except Exception as exc:  # noqa: BLE001
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.warning(
                "pilot.skill_error",
                skill=rs.spec.skill_name,
                error=str(exc),
            )
            return SkillResult(
                skill_name=rs.spec.skill_name,
                success=False,
                error=str(exc),
                duration_ms=duration_ms,
                dry_run=invocation.dry_run,
            )

        # 强制覆盖 duration_ms / dry_run / skill_name，防止 skill 实现写错
        result = result.model_copy(
            update={
                "skill_name": rs.spec.skill_name,
                "duration_ms": int((time.monotonic() - start) * 1000),
                "dry_run": invocation.dry_run,
            },
        )
        return result


__all__ = [
    "DryRunUnsupported",
    "RegisteredSkill",
    "SideEffect",
    "SkillFunc",
    "SkillInvocation",
    "SkillNotFound",
    "SkillRegistry",
    "SkillResult",
    "SkillSpec",
]
