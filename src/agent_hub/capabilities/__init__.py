"""Capability layer: skills, MCP server configs, and plugin bundles."""

from agent_hub.capabilities.mcp import (
    MCPConfig,
    MCPDiagnostic,
    MCPServerSpec,
    extract_mcp_server_map,
    load_inline_mcp_config,
    load_mcp_config_file,
    merge_mcp_configs,
)
from agent_hub.capabilities.plugins import (
    PluginBundle,
    PluginDiagnostic,
    discover_claude_plugins,
    load_claude_plugin,
)
from agent_hub.capabilities.skills import (
    SkillDiagnostic,
    SkillPackage,
    format_skill_invocation,
    format_skills_for_prompt,
    load_skills,
    load_skills_from_dir,
)

__all__ = [
    "MCPConfig",
    "MCPDiagnostic",
    "MCPServerSpec",
    "PluginBundle",
    "PluginDiagnostic",
    "SkillDiagnostic",
    "SkillPackage",
    "discover_claude_plugins",
    "extract_mcp_server_map",
    "format_skill_invocation",
    "format_skills_for_prompt",
    "load_inline_mcp_config",
    "load_claude_plugin",
    "load_mcp_config_file",
    "load_skills",
    "load_skills_from_dir",
    "merge_mcp_configs",
]
