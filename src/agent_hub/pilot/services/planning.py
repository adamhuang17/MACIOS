"""PlanningService：把 ModelGateway 的 PlanBlueprint 物化为 Plan + PlanSteps。

根据 :class:`PlanProfile` 不同动态选择必选 step kind 集合：

- ``DECK_FULL``    ： 6 步完整链（READ_CONTEXT/GENERATE_DOC/GENERATE_SLIDE_SPEC/
  RENDER_SLIDES/UPLOAD/SUMMARIZE）；
- ``DOC_ONLY``     ： READ_CONTEXT + GENERATE_DOC；
- ``SUMMARY_ONLY`` ： READ_CONTEXT + SUMMARIZE；
- ``REVISION``     ： READ_CONTEXT + GENERATE_DOC（revise）。

所有 write/share 风险等级的 step 自动设为 ``requires_approval=True``；
所有 step 在写入前都会经过 :func:`validate_plan_graph` DAG 校验。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, ConfigDict, Field

from agent_hub.pilot.domain.enums import (
    EventLevel,
    EventType,
    PlanStatus,
    PlanStepKind,
    PlanStepStatus,
    RiskLevel,
)
from agent_hub.pilot.domain.errors import PilotDomainError
from agent_hub.pilot.domain.events import ExecutionEvent
from agent_hub.pilot.domain.models import Plan, PlanStep, Task
from agent_hub.pilot.domain.state import validate_plan_graph
from agent_hub.pilot.services.dto import PlanBlueprint, PlanProfile, PlanStepDraft

if TYPE_CHECKING:
    from agent_hub.pilot.services.event_publisher import PilotEventPublisher
    from agent_hub.pilot.services.model_gateway import ModelGateway, PlanContext
    from agent_hub.pilot.services.repository import PilotRepository

logger = structlog.get_logger(__name__)


REQUIRED_KINDS_BY_PROFILE: dict[PlanProfile, frozenset[PlanStepKind]] = {
    PlanProfile.DECK_FULL: frozenset({
        PlanStepKind.READ_CONTEXT,
        PlanStepKind.GENERATE_DOC,
        PlanStepKind.GENERATE_SLIDE_SPEC,
        PlanStepKind.RENDER_SLIDES,
        PlanStepKind.UPLOAD,
        PlanStepKind.SUMMARIZE,
    }),
    PlanProfile.DOC_ONLY: frozenset({
        PlanStepKind.READ_CONTEXT,
        PlanStepKind.GENERATE_DOC,
    }),
    PlanProfile.SUMMARY_ONLY: frozenset({
        PlanStepKind.READ_CONTEXT,
        PlanStepKind.SUMMARIZE,
    }),
    PlanProfile.REVISION: frozenset({
        PlanStepKind.READ_CONTEXT,
        PlanStepKind.GENERATE_DOC,
    }),
}

# 兼容别名：M1~M4 代码引用的 ``REQUIRED_KINDS`` 依然指向 deck_full 版本。
REQUIRED_KINDS: frozenset[PlanStepKind] = REQUIRED_KINDS_BY_PROFILE[
    PlanProfile.DECK_FULL
]


class PlanTemplateError(PilotDomainError):
    """plan 模板硬约束未满足。"""


class BriefRevisionContext(BaseModel):
    """Brief 修订 mini-plan 的输入上下文。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_brief_artifact_id: str
    revision_instruction: str
    title: str | None = None
    language: str = "zh-CN"
    attachments: list[str] = Field(default_factory=list)
    markdown_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)


class PlanningService:
    def __init__(
        self,
        gateway: ModelGateway,
        repository: PilotRepository,
        publisher: PilotEventPublisher,
    ) -> None:
        self._gateway = gateway
        self._repo = repository
        self._publisher = publisher

    async def generate(
        self,
        task: Task,
        ctx: PlanContext,
    ) -> tuple[Plan, list[PlanStep]]:
        blueprint = await self._gateway.complex_plan(ctx)
        self._enforce_template(blueprint)
        return await self._persist_blueprint(
            task,
            blueprint,
            event_message=f"plan generated with {len(blueprint.steps)} steps",
            extra_payload={"goal": blueprint.goal, "step_count": len(blueprint.steps)},
        )

    @staticmethod
    def _enforce_template(blueprint: PlanBlueprint) -> None:
        if not blueprint.steps:
            raise PlanTemplateError("blueprint 不能为空")
        required = REQUIRED_KINDS_BY_PROFILE.get(blueprint.profile)
        if required is None:
            raise PlanTemplateError(f"未知 plan profile: {blueprint.profile}")
        kinds = {s.kind for s in blueprint.steps}
        missing = required - kinds
        if missing:
            names = sorted(k.value for k in missing)
            raise PlanTemplateError(
                f"plan ({blueprint.profile.value}) 缺失必选 step kind: {names}",
            )

        refs = [s.ref for s in blueprint.steps]
        if len(set(refs)) != len(refs):
            raise PlanTemplateError("blueprint step.ref 必须唯一")

        ref_set = set(refs)
        for s in blueprint.steps:
            for dep in s.depends_on:
                if dep not in ref_set:
                    raise PlanTemplateError(f"step {s.ref} 依赖未知 ref {dep}")

    @staticmethod
    def _materialize_steps(plan: Plan, blueprint: PlanBlueprint) -> list[PlanStep]:
        ref_to_id: dict[str, str] = {}
        drafts_with_steps: list[tuple[PlanStep, list[str]]] = []
        for index, draft in enumerate(blueprint.steps):
            requires_approval = draft.requires_approval or draft.risk_level in (
                RiskLevel.WRITE,
                RiskLevel.SHARE,
                RiskLevel.ADMIN,
            )
            step = PlanStep(
                workspace_id=plan.workspace_id,
                task_id=plan.task_id,
                plan_id=plan.plan_id,
                index=index,
                title=draft.title,
                kind=draft.kind,
                skill_name=draft.skill_name,
                input_params=dict(draft.input_params),
                inputs_from=list(draft.inputs_from),
                output_artifact_ref=draft.output_artifact_ref,
                depends_on=[],
                risk_level=draft.risk_level,
                requires_approval=requires_approval,
                timeout_ms=draft.timeout_ms,
                max_retries=draft.max_retries,
                status=PlanStepStatus.PENDING,
            )
            ref_to_id[draft.ref] = step.step_id
            drafts_with_steps.append((step, list(draft.depends_on)))

        result: list[PlanStep] = []
        for step, draft_deps in drafts_with_steps:
            real_deps = [ref_to_id[d] for d in draft_deps]
            result.append(step.model_copy(update={"depends_on": real_deps}))
        return result

    # ── M3 mini-plan：Brief 修订 ───────────────────

    async def generate_revision(
        self,
        task: Task,
        ctx: BriefRevisionContext,
    ) -> tuple[Plan, list[PlanStep]]:
        """生成 Brief 修订 mini-plan：READ_CONTEXT → GENERATE_DOC（revise）。

        与 :meth:`generate` 区别：
        - 不强制完整六步模板；只保证 ``READ_CONTEXT`` 与 ``GENERATE_DOC``；
        - revise step 通过 ``base_brief_artifact_id`` 指向旧 Brief，由
          ``ArtifactStore`` 在保存新 Brief 时把旧 Brief 标 SUPERSEDED。
        """
        title = (ctx.title or task.origin_text[:32] or "Project Brief").strip()
        drafts: list[PlanStepDraft] = [
            PlanStepDraft(
                ref="ctx",
                title="重新收集上下文",
                kind=PlanStepKind.READ_CONTEXT,
                skill_name="internal.context.assemble",
                input_params={
                    "origin_text": task.origin_text,
                    "attachments": list(ctx.attachments),
                    "markdown_refs": list(ctx.markdown_refs),
                },
                output_artifact_ref="ctx",
            ),
            PlanStepDraft(
                ref="brief",
                title="重写 Brief",
                kind=PlanStepKind.GENERATE_DOC,
                skill_name="internal.brief.revise",
                input_params={
                    "title": f"{title} (revised)",
                    "base_brief_artifact_id": ctx.base_brief_artifact_id,
                    "revision_instruction": ctx.revision_instruction,
                    "language": ctx.language,
                },
                inputs_from=["ctx"],
                output_artifact_ref="brief",
                depends_on=["ctx"],
            ),
        ]
        blueprint = PlanBlueprint(
            goal=f"按指令修订 Brief：{ctx.revision_instruction[:60]}",
            assumptions=[f"base_brief_artifact_id={ctx.base_brief_artifact_id}"],
            steps=drafts,
            profile=PlanProfile.REVISION,
            metadata={"gateway": "planning.brief_revision"},
        )
        self._enforce_template(blueprint)

        return await self._persist_blueprint(
            task,
            blueprint,
            event_message="brief revision plan generated",
            extra_payload={
                "goal": blueprint.goal,
                "step_count": len(blueprint.steps),
                "kind": "brief_revision",
                "base_brief_artifact_id": ctx.base_brief_artifact_id,
            },
        )

    # ── 内部 ── ───────────────────────────────────

    async def _persist_blueprint(
        self,
        task: Task,
        blueprint: PlanBlueprint,
        *,
        event_message: str,
        extra_payload: dict[str, object] | None = None,
    ) -> tuple[Plan, list[PlanStep]]:
        plan_metadata: dict[str, object] = {
            **dict(blueprint.metadata),
            "plan_profile": blueprint.profile.value,
        }
        plan = Plan(
            workspace_id=task.workspace_id,
            task_id=task.task_id,
            goal=blueprint.goal,
            assumptions=list(blueprint.assumptions),
            status=PlanStatus.DRAFT,
            metadata=plan_metadata,
        )
        steps = self._materialize_steps(plan, blueprint)
        validate_plan_graph(steps)

        plan = plan.model_copy(update={
            "step_ids": [s.step_id for s in steps],
        })
        await self._repo.save(plan, expected_version=None)
        for step in steps:
            await self._repo.save(step, expected_version=None)

        payload: dict[str, object] = {
            "plan_profile": blueprint.profile.value,
            "gateway": blueprint.metadata.get("gateway"),
            "skill_mode": blueprint.metadata.get("skill_mode"),
        }
        if extra_payload:
            payload.update(extra_payload)

        await self._publisher.record(ExecutionEvent(
            workspace_id=plan.workspace_id,
            trace_id=task.trace_id,
            task_id=task.task_id,
            plan_id=plan.plan_id,
            type=EventType.PLAN_GENERATED,
            level=EventLevel.INFO,
            message=event_message,
            payload=payload,
        ))

        logger.info(
            "pilot.plan_generated",
            plan_id=plan.plan_id,
            task_id=task.task_id,
            step_count=len(steps),
            plan_profile=blueprint.profile.value,
        )
        return plan, steps


__all__ = [
    "REQUIRED_KINDS",
    "REQUIRED_KINDS_BY_PROFILE",
    "BriefRevisionContext",
    "PlanTemplateError",
    "PlanningService",
]
