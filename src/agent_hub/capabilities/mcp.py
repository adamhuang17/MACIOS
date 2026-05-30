"""MCP server configuration loader.

MCP is treated as the external tool-service protocol. This module does not
launch servers or execute tools; it normalizes server configs from `.mcp.json`
or plugin manifests so a runtime MCP client can connect and discover tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CLAUDE_PLUGIN_ROOT = "${CLAUDE_PLUGIN_ROOT}"


@dataclass(frozen=True, slots=True)
class MCPDiagnostic:
    """Warning produced while loading MCP config."""

    path: Path | None
    message: str
    plugin_id: str | None = None


@dataclass(frozen=True, slots=True)
class MCPServerSpec:
    """Normalized MCP server entry."""

    name: str
    config: dict[str, Any]
    source: str = "path"
    plugin_id: str | None = None


@dataclass(frozen=True, slots=True)
class MCPConfig:
    """Loaded MCP server catalog."""

    servers: dict[str, MCPServerSpec] = field(default_factory=dict)
    diagnostics: list[MCPDiagnostic] = field(default_factory=list)


def extract_mcp_server_map(raw: Any) -> dict[str, dict[str, Any]]:
    """Extract a server map from common MCP shapes.

    Supports plain server maps, `{ "mcpServers": ... }`, and `{ "servers": ... }`.
    """
    if not isinstance(raw, dict):
        return {}
    nested = raw.get("mcpServers")
    if not isinstance(nested, dict):
        nested = raw.get("servers")
    if not isinstance(nested, dict):
        nested = raw

    servers: dict[str, dict[str, Any]] = {}
    for name, value in nested.items():
        if isinstance(name, str) and isinstance(value, dict):
            servers[name] = dict(value)
    return servers


def load_mcp_config_file(
    path: str | Path,
    *,
    plugin_root: str | Path | None = None,
    plugin_id: str | None = None,
    source: str = "path",
) -> MCPConfig:
    """Load MCP servers from a JSON file."""
    config_path = Path(path).expanduser().resolve()
    diagnostics: list[MCPDiagnostic] = []
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return MCPConfig(
            diagnostics=[MCPDiagnostic(path=config_path, message="MCP config not found")]
        )
    except (OSError, json.JSONDecodeError) as exc:
        return MCPConfig(diagnostics=[MCPDiagnostic(path=config_path, message=str(exc))])

    root = Path(plugin_root).expanduser().resolve() if plugin_root else config_path.parent
    base_dir = config_path.parent
    servers = _normalize_servers(
        extract_mcp_server_map(raw),
        base_dir=base_dir,
        plugin_root=root,
        plugin_id=plugin_id,
        source=source,
    )
    return MCPConfig(servers=servers, diagnostics=diagnostics)


def load_inline_mcp_config(
    raw_mcp_servers: Any,
    *,
    plugin_root: str | Path,
    plugin_id: str | None = None,
    source: str = "plugin",
) -> MCPConfig:
    """Load MCP servers from a manifest inline `mcpServers` object."""
    root = Path(plugin_root).expanduser().resolve()
    servers = _normalize_servers(
        extract_mcp_server_map({"mcpServers": raw_mcp_servers}),
        base_dir=root,
        plugin_root=root,
        plugin_id=plugin_id,
        source=source,
    )
    return MCPConfig(servers=servers)


def merge_mcp_configs(*configs: MCPConfig) -> MCPConfig:
    """Merge MCP configs in order; later entries override by server name."""
    servers: dict[str, MCPServerSpec] = {}
    diagnostics: list[MCPDiagnostic] = []
    for config in configs:
        diagnostics.extend(config.diagnostics)
        servers.update(config.servers)
    return MCPConfig(servers=servers, diagnostics=diagnostics)


def _normalize_servers(
    raw_servers: dict[str, dict[str, Any]],
    *,
    base_dir: Path,
    plugin_root: Path,
    plugin_id: str | None,
    source: str,
) -> dict[str, MCPServerSpec]:
    result: dict[str, MCPServerSpec] = {}
    for name, server in raw_servers.items():
        result[name] = MCPServerSpec(
            name=name,
            config=_absolutize_server_config(server, base_dir=base_dir, plugin_root=plugin_root),
            plugin_id=plugin_id,
            source=source,
        )
    return result


def _absolutize_server_config(
    server: dict[str, Any],
    *,
    base_dir: Path,
    plugin_root: Path,
) -> dict[str, Any]:
    next_config = dict(server)

    if "cwd" not in next_config and "workingDirectory" not in next_config:
        next_config["cwd"] = str(base_dir)

    for key in ("command", "cwd", "workingDirectory"):
        value = next_config.get(key)
        if isinstance(value, str):
            next_config[key] = _resolve_mcp_path(value, base_dir=base_dir, plugin_root=plugin_root)

    args = next_config.get("args")
    if isinstance(args, list):
        next_config["args"] = [
            _resolve_mcp_path(value, base_dir=base_dir, plugin_root=plugin_root)
            if isinstance(value, str)
            else value
            for value in args
        ]

    env = next_config.get("env")
    if isinstance(env, dict):
        next_config["env"] = {
            key: (
                _expand_plugin_root(value, plugin_root)
                if isinstance(value, str)
                else value
            )
            for key, value in env.items()
        }

    return next_config


def _resolve_mcp_path(value: str, *, base_dir: Path, plugin_root: Path) -> str:
    expanded = _expand_plugin_root(value, plugin_root)
    if expanded.startswith("./") or expanded.startswith("../"):
        return str((base_dir / expanded).resolve())
    candidate = Path(expanded)
    return str(candidate.resolve()) if candidate.is_absolute() else expanded


def _expand_plugin_root(value: str, plugin_root: Path) -> str:
    return value.replace(CLAUDE_PLUGIN_ROOT, str(plugin_root))
