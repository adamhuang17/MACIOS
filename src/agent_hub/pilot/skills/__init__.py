"""Pilot 技能层：SkillRegistry + 默认 Fake 技能集合。

Skill 与现有 ``agent_hub.agents.registry.ToolRegistry`` 完全独立，
仅负责能力执行；状态迁移、审批、事件、Artifact 生命周期由
``agent_hub.pilot.services`` 统一管理。
"""

from agent_hub.pilot.skills.fake import register_fake_skills
from agent_hub.pilot.skills.feishu import FEISHU_SCOPES, register_feishu_skills
from agent_hub.pilot.skills.internal import (
    BRIEF_SECTIONS,
    ArtifactContentReader,
    BriefGenerator,
    TemplateBriefGenerator,
    register_internal_document_skills,
)
from agent_hub.pilot.skills.registry import (
    DryRunUnsupported,
    RegisteredSkill,
    SkillFunc,
    SkillInvocation,
    SkillNotFound,
    SkillRegistry,
    SkillResult,
    SkillSpec,
)

__all__ = [
    "BRIEF_SECTIONS",
    "FEISHU_SCOPES",
    "ArtifactContentReader",
    "BriefGenerator",
    "DryRunUnsupported",
    "RegisteredSkill",
    "SkillFunc",
    "SkillInvocation",
    "SkillNotFound",
    "SkillRegistry",
    "SkillResult",
    "SkillSpec",
    "TemplateBriefGenerator",
    "register_fake_skills",
    "register_feishu_skills",
    "register_internal_document_skills",
]
