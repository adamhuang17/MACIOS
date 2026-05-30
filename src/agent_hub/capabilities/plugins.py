"""Claude Code plugin bundle loader.

Plugin means a distribution unit that packages commands, agents, skills and
MCP server configuration. This loader targets the Claude Code bundle shape:

```
plugin/
|-- .claude-plugin/plugin.json
|-- commands/
|-- agents/
|-- skills/
`-- .mcp.json
```
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_hub.capabilities.mcp import (
    MCPConfig,
    MCPDiagnostic,
    load_inline_mcp_config,
    load_mcp_config_file,
    merge_mcp_configs,
)
from agent_hub.capabilities.skills import (
    SkillDiagnostic,
    SkillPackage,
    load_skills_from_dir,
)

CLAUDE_MANIFEST = Path(".claude-plugin") / "plugin.json"


@dataclass(frozen=True, slots=True)
class PluginDiagnostic:
    """Warning produced while loading plugin bundles."""

    path: Path | None
    message: str
    plugin_id: str | None = None


@dataclass(frozen=True, slots=True)
class PluginBundle:
    """Loaded Claude-style plugin bundle."""

    plugin_id: str
    root_dir: Path
    manifest_path: Path
    name: str | None = None
    description: str | None = None
    version: str | None = None
    command_paths: list[Path] = field(default_factory=list)
    agent_paths: list[Path] = field(default_factory=list)
    skill_paths: list[Path] = field(default_factory=list)
    mcp_paths: list[Path] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    skills: list[SkillPackage] = field(default_factory=list)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    diagnostics: list[PluginDiagnostic | SkillDiagnostic | MCPDiagnostic] = field(
        default_factory=list
    )


def load_claude_plugin(root_dir: str | Path) -> PluginBundle:
    """Load a Claude Code plugin bundle from a root directory."""
    root = Path(root_dir).expanduser().resolve()
    manifest_path = root / CLAUDE_MANIFEST
    raw, manifest_warnings = _read_manifest(manifest_path)
    plugin_id = _plugin_id(raw, root)

    command_paths = _resolve_component_paths(root, raw, "commands", ["commands"])
    agent_paths = _resolve_component_paths(root, raw, "agents", ["agents"])
    skill_paths = _resolve_component_paths(root, raw, "skills", ["skills"])
    mcp_paths = _resolve_mcp_paths(root, raw)

    skills: list[SkillPackage] = []
    diagnostics: list[PluginDiagnostic | SkillDiagnostic | MCPDiagnostic] = [
        *manifest_warnings
    ]
    for skill_path in skill_paths:
        loaded, warnings = load_skills_from_dir(skill_path, source=f"plugin:{plugin_id}")
        skills.extend(loaded)
        diagnostics.extend(warnings)

    mcp_configs: list[MCPConfig] = []
    for mcp_path in mcp_paths:
        mcp_configs.append(
            load_mcp_config_file(
                mcp_path,
                plugin_root=root,
                plugin_id=plugin_id,
                source="plugin",
            )
        )
    if isinstance(raw.get("mcpServers"), dict):
        mcp_configs.append(
            load_inline_mcp_config(
                raw["mcpServers"],
                plugin_root=root,
                plugin_id=plugin_id,
                source="plugin",
            )
        )
    mcp = merge_mcp_configs(*mcp_configs) if mcp_configs else MCPConfig()
    diagnostics.extend(mcp.diagnostics)

    return PluginBundle(
        plugin_id=plugin_id,
        root_dir=root,
        manifest_path=manifest_path,
        name=_optional_str(raw.get("name")),
        description=_optional_str(raw.get("description")),
        version=_optional_str(raw.get("version")),
        command_paths=command_paths,
        agent_paths=agent_paths,
        skill_paths=skill_paths,
        mcp_paths=mcp_paths,
        capabilities=_capabilities(
            command_paths=command_paths,
            agent_paths=agent_paths,
            skill_paths=skill_paths,
            mcp=mcp,
        ),
        skills=skills,
        mcp=mcp,
        diagnostics=diagnostics,
    )


def discover_claude_plugins(parent_dir: str | Path) -> list[PluginBundle]:
    """Load every direct child containing `.claude-plugin/plugin.json`."""
    parent = Path(parent_dir).expanduser().resolve()
    if not parent.is_dir():
        return []
    plugins: list[PluginBundle] = []
    for child in sorted(parent.iterdir(), key=lambda path: path.name.lower()):
        if child.is_dir() and (child / CLAUDE_MANIFEST).is_file():
            plugins.append(load_claude_plugin(child))
    return plugins


def _read_manifest(path: Path) -> tuple[dict[str, Any], list[PluginDiagnostic]]:
    if not path.is_file():
        return {}, [PluginDiagnostic(path=path, message="plugin manifest not found")]
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [PluginDiagnostic(path=path, message=str(exc))]
    if not isinstance(raw, dict):
        return {}, [PluginDiagnostic(path=path, message="plugin manifest must be an object")]
    return raw, []


def _resolve_component_paths(
    root: Path,
    raw: dict[str, Any],
    key: str,
    defaults: list[str],
) -> list[Path]:
    declared = _path_list(raw.get(key))
    existing_defaults = [default for default in defaults if (root / default).exists()]
    return _normalize_paths(root, [*existing_defaults, *declared])


def _resolve_mcp_paths(root: Path, raw: dict[str, Any]) -> list[Path]:
    declared = [] if isinstance(raw.get("mcpServers"), dict) else _path_list(raw.get("mcpServers"))
    defaults = [".mcp.json"] if (root / ".mcp.json").is_file() else []
    return _normalize_paths(root, [*defaults, *declared])


def _normalize_paths(root: Path, values: list[str]) -> list[Path]:
    normalized: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        if not value:
            continue
        candidate = (root / value).resolve()
        if not _is_under(candidate, root):
            continue
        if candidate not in seen:
            normalized.append(candidate)
            seen.add(candidate)
    return normalized


def _path_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def _plugin_id(raw: dict[str, Any], root: Path) -> str:
    source = _optional_str(raw.get("name")) or root.name
    slug = "".join(ch if ch.isalnum() else "-" for ch in source.lower())
    slug = "-".join(part for part in slug.split("-") if part)
    return slug or "bundle-plugin"


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _capabilities(
    *,
    command_paths: list[Path],
    agent_paths: list[Path],
    skill_paths: list[Path],
    mcp: MCPConfig,
) -> list[str]:
    capabilities: list[str] = []
    if command_paths:
        capabilities.append("commands")
    if agent_paths:
        capabilities.append("agents")
    if skill_paths:
        capabilities.append("skills")
    if mcp.servers:
        capabilities.append("mcpServers")
    return capabilities
