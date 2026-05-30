"""Capability layer contract tests for skills, MCP and plugin bundles."""

from __future__ import annotations

import json
from pathlib import Path

from agent_hub.capabilities.mcp import extract_mcp_server_map, load_mcp_config_file
from agent_hub.capabilities.plugins import load_claude_plugin
from agent_hub.capabilities.skills import (
    format_skill_invocation,
    format_skills_for_prompt,
    load_skills_from_dir,
)


def test_load_skill_package_from_skill_md(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "doc-review"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        """---
name: doc-review
description: Review documents for structure and missing evidence.
---
Use this skill when the user asks for a document review.
""",
        encoding="utf-8",
    )

    skills, diagnostics = load_skills_from_dir(tmp_path / "skills")

    assert diagnostics == []
    assert len(skills) == 1
    assert skills[0].name == "doc-review"
    assert skills[0].file_path == skill_file.resolve()
    assert "doc-review" in format_skills_for_prompt(skills)
    invocation = format_skill_invocation(skills[0])
    assert "<skill" in invocation
    assert "References are relative to" in invocation


def test_mcp_config_file_normalizes_plugin_paths(tmp_path: Path) -> None:
    mcp_file = tmp_path / ".mcp.json"
    mcp_file.write_text(
        json.dumps({
            "mcpServers": {
                "docs": {
                    "command": "node",
                    "args": ["${CLAUDE_PLUGIN_ROOT}/server.js", "./config.json"],
                    "env": {"PLUGIN_ROOT": "${CLAUDE_PLUGIN_ROOT}"},
                }
            }
        }),
        encoding="utf-8",
    )

    loaded = load_mcp_config_file(mcp_file, plugin_root=tmp_path, plugin_id="docs-plugin")

    assert loaded.diagnostics == []
    server = loaded.servers["docs"]
    assert server.plugin_id == "docs-plugin"
    assert server.config["command"] == "node"
    assert server.config["args"][0] == str((tmp_path / "server.js").resolve())
    assert server.config["args"][1] == str((tmp_path / "config.json").resolve())
    assert server.config["env"]["PLUGIN_ROOT"] == str(tmp_path.resolve())


def test_extract_mcp_server_map_accepts_common_shapes() -> None:
    assert set(extract_mcp_server_map({"docs": {"command": "node"}})) == {"docs"}
    assert set(
        extract_mcp_server_map({"mcpServers": {"docs": {"command": "node"}}})
    ) == {"docs"}
    assert set(extract_mcp_server_map({"servers": {"docs": {"command": "node"}}})) == {
        "docs"
    }


def test_load_claude_plugin_bundle(tmp_path: Path) -> None:
    plugin_root = tmp_path / "docs-plugin"
    (plugin_root / ".claude-plugin").mkdir(parents=True)
    (plugin_root / "commands").mkdir()
    (plugin_root / "agents").mkdir()
    skill_dir = plugin_root / "skills" / "doc-review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: doc-review
description: Review documents for structure and missing evidence.
---
Review the provided document.
""",
        encoding="utf-8",
    )
    (plugin_root / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"docs": {"command": "node"}}}),
        encoding="utf-8",
    )
    (plugin_root / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({
            "name": "docs-plugin",
            "version": "0.1.0",
            "description": "Document workflow plugin",
        }),
        encoding="utf-8",
    )

    bundle = load_claude_plugin(plugin_root)

    assert bundle.plugin_id == "docs-plugin"
    assert bundle.capabilities == ["commands", "agents", "skills", "mcpServers"]
    assert [skill.name for skill in bundle.skills] == ["doc-review"]
    assert set(bundle.mcp.servers) == {"docs"}
