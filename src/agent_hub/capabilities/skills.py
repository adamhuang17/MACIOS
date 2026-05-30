"""Claude/OpenClaw-style skill package loader.

Skill means "instructions that teach the model how and when to use a
capability". A skill is not executable by itself and is not registered as a
tool. The loader follows the Agent Skills convention used by Claude Code and
OpenClaw:

- a directory containing ``SKILL.md`` is a skill root;
- direct root ``*.md`` files can also be loaded as skills;
- ``name`` and ``description`` are read from YAML frontmatter;
- skills with ``disable-model-invocation: true`` are hidden from prompt listing.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024


@dataclass(frozen=True, slots=True)
class SkillDiagnostic:
    """Warning produced while scanning skill packages."""

    path: Path
    message: str
    type: str = "warning"


@dataclass(frozen=True, slots=True)
class SkillPackage:
    """Loaded model-instruction package."""

    name: str
    description: str
    content: str
    file_path: Path
    base_dir: Path
    source: str = "path"
    disable_model_invocation: bool = False


def load_skills(
    paths: list[str | Path],
    *,
    source: str = "path",
) -> tuple[list[SkillPackage], list[SkillDiagnostic]]:
    """Load skills from files or directories, de-duplicating by name."""
    loaded: dict[str, SkillPackage] = {}
    diagnostics: list[SkillDiagnostic] = []
    real_paths: set[Path] = set()

    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            diagnostics.append(SkillDiagnostic(path=path, message="skill path does not exist"))
            continue

        if path.is_dir():
            skills, warnings = load_skills_from_dir(path, source=source)
        elif path.is_file() and path.suffix.lower() == ".md":
            skill, warnings = _load_skill_file(path, source=source)
            skills = [skill] if skill is not None else []
        else:
            skills = []
            warnings = [SkillDiagnostic(path=path, message="skill path is not a markdown file")]

        diagnostics.extend(warnings)
        for skill in skills:
            real_path = skill.file_path.resolve()
            if real_path in real_paths:
                continue
            existing = loaded.get(skill.name)
            if existing is not None:
                diagnostics.append(
                    SkillDiagnostic(
                        path=skill.file_path,
                        message=(
                            f'skill name "{skill.name}" collision; '
                            f"winner={existing.file_path}"
                        ),
                        type="collision",
                    )
                )
                continue
            loaded[skill.name] = skill
            real_paths.add(real_path)

    return list(loaded.values()), diagnostics


def load_skills_from_dir(
    directory: str | Path,
    *,
    source: str = "path",
    include_root_files: bool = True,
) -> tuple[list[SkillPackage], list[SkillDiagnostic]]:
    """Recursively load skills from a directory."""
    root = Path(directory).expanduser().resolve()
    return _load_skills_from_dir(root, source=source, include_root_files=include_root_files)


def format_skills_for_prompt(skills: list[SkillPackage]) -> str:
    """Return an ``<available_skills>`` prompt block."""
    visible = [skill for skill in skills if not skill.disable_model_invocation]
    if not visible:
        return ""

    lines = [
        "The following skills provide specialized instructions for specific tasks.",
        "Load a skill when the task matches its description.",
        "Resolve relative references from the skill directory.",
        "",
        "<available_skills>",
    ]
    for skill in visible:
        lines.extend([
            "  <skill>",
            f"    <name>{html.escape(skill.name)}</name>",
            f"    <description>{html.escape(skill.description)}</description>",
            f"    <location>{html.escape(str(skill.file_path))}</location>",
            "  </skill>",
        ])
    lines.append("</available_skills>")
    return "\n".join(lines)


def format_skill_invocation(
    skill: SkillPackage,
    additional_instructions: str | None = None,
) -> str:
    """Format the exact skill content for injection once selected."""
    block = (
        f'<skill name="{html.escape(skill.name)}" '
        f'location="{html.escape(str(skill.file_path))}">\n'
        f"References are relative to {skill.base_dir}.\n\n"
        f"{skill.content}\n"
        "</skill>"
    )
    if additional_instructions:
        return f"{block}\n\n{additional_instructions}"
    return block


def _load_skills_from_dir(
    directory: Path,
    *,
    source: str,
    include_root_files: bool,
) -> tuple[list[SkillPackage], list[SkillDiagnostic]]:
    skills: list[SkillPackage] = []
    diagnostics: list[SkillDiagnostic] = []
    if not directory.exists() or not directory.is_dir():
        return skills, diagnostics

    skill_file = directory / "SKILL.md"
    if skill_file.is_file():
        skill, warnings = _load_skill_file(skill_file, source=source)
        if skill is not None:
            skills.append(skill)
        diagnostics.extend(warnings)
        return skills, diagnostics

    entries = sorted(directory.iterdir(), key=lambda p: p.name.lower())
    for entry in entries:
        if entry.name.startswith(".") or entry.name == "node_modules":
            continue
        if entry.is_dir():
            nested, warnings = _load_skills_from_dir(
                entry,
                source=source,
                include_root_files=False,
            )
            skills.extend(nested)
            diagnostics.extend(warnings)
            continue
        if include_root_files and entry.is_file() and entry.suffix.lower() == ".md":
            skill, warnings = _load_skill_file(entry, source=source)
            if skill is not None:
                skills.append(skill)
            diagnostics.extend(warnings)
    return skills, diagnostics


def _load_skill_file(
    path: Path,
    *,
    source: str,
) -> tuple[SkillPackage | None, list[SkillDiagnostic]]:
    diagnostics: list[SkillDiagnostic] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, [SkillDiagnostic(path=path, message=str(exc))]

    try:
        frontmatter, body = _parse_frontmatter(raw)
    except ValueError as exc:
        return None, [SkillDiagnostic(path=path, message=str(exc))]

    base_dir = path.parent
    name = str(frontmatter.get("name") or base_dir.name).strip()
    description = str(frontmatter.get("description") or "").strip()

    for error in _validate_name(name):
        diagnostics.append(SkillDiagnostic(path=path, message=error))
    for error in _validate_description(description):
        diagnostics.append(SkillDiagnostic(path=path, message=error))
    if not description:
        return None, diagnostics

    disable_model_invocation = frontmatter.get("disable-model-invocation") is True
    return (
        SkillPackage(
            name=name,
            description=description,
            content=body,
            file_path=path,
            base_dir=base_dir,
            source=source,
            disable_model_invocation=disable_model_invocation,
        ),
        diagnostics,
    )


def _parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---\n"):
        return {}, normalized.strip()
    end_index = normalized.find("\n---", 4)
    if end_index == -1:
        return {}, normalized.strip()
    frontmatter_raw = normalized[4:end_index]
    body = normalized[end_index + 4 :].strip()
    parsed = yaml.safe_load(frontmatter_raw) or {}
    if not isinstance(parsed, dict):
        raise ValueError("skill frontmatter must be an object")
    return parsed, body


def _validate_name(name: str) -> list[str]:
    errors: list[str] = []
    if len(name) > MAX_NAME_LENGTH:
        errors.append(f"name exceeds {MAX_NAME_LENGTH} characters")
    if not re.fullmatch(r"[a-z0-9-]+", name):
        errors.append("name must contain only lowercase a-z, 0-9, and hyphens")
    if name.startswith("-") or name.endswith("-"):
        errors.append("name must not start or end with a hyphen")
    if "--" in name:
        errors.append("name must not contain consecutive hyphens")
    return errors


def _validate_description(description: str) -> list[str]:
    if not description:
        return ["description is required"]
    if len(description) > MAX_DESCRIPTION_LENGTH:
        return [f"description exceeds {MAX_DESCRIPTION_LENGTH} characters"]
    return []
