"""架构边界守卫测试。

在重构期间防止模块间依赖关系倒置，确保分层约束不被破坏。
这些测试通过静态 AST 分析完成，不需要 import 任何被测模块。

守卫的核心规则：
1. pilot/domain/ 不得依赖 FastAPI、飞书、LLM 客户端、AgentPipeline
2. capabilities/ loader 不得直接调用 subprocess 或 os.system
3. core/ 不得直接 import connectors.feishu 具体实现
4. pilot/domain/ 不得 import pilot/services/（保持领域层独立）
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# 项目 src 根
_SRC = Path(__file__).resolve().parents[1] / "src" / "agent_hub"

# ── AST 工具函数 ──────────────────────────────────────────────────────────────


def _collect_imports(path: Path) -> list[str]:
    """返回文件中所有 import 语句所引用的模块名（顶层 + 全名）。

    支持 ``import a.b.c`` 和 ``from a.b import c`` 两种形式。
    对于语法错误文件跳过（不让守卫本身崩溃）。
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return []

    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module)
    return modules


def _py_files(directory: Path) -> list[Path]:
    """递归收集目录下所有 .py 文件。"""
    return list(directory.rglob("*.py"))


def _violations(
    directory: Path,
    forbidden_patterns: list[str],
    *,
    exclude_dirs: list[str] | None = None,
) -> list[tuple[Path, str, str]]:
    """返回 (文件路径, 被禁止的 import 模块名, 匹配的 forbidden 模式) 三元组列表。"""
    result: list[tuple[Path, str, str]] = []
    exclude = set(exclude_dirs or [])
    for py_file in _py_files(directory):
        # 排除指定子目录
        if any(part in exclude for part in py_file.parts):
            continue
        for mod in _collect_imports(py_file):
            for pattern in forbidden_patterns:
                if mod == pattern or mod.startswith(pattern + ".") or mod.startswith(pattern):
                    result.append((py_file, mod, pattern))
                    break
    return result


# ── 规则 1: pilot/domain 层边界 ──────────────────────────────────────────────


class TestPilotDomainBoundary:
    """pilot/domain/ 是领域内核，必须对外部框架完全独立。"""

    _DOMAIN = _SRC / "pilot" / "domain"

    def test_no_fastapi(self) -> None:
        """pilot/domain 不得依赖 FastAPI。"""
        violations = _violations(self._DOMAIN, ["fastapi", "starlette"])
        assert not violations, _fmt(violations, "pilot/domain 不能 import FastAPI/Starlette")

    def test_no_feishu_connector(self) -> None:
        """pilot/domain 不得依赖飞书连接器。"""
        violations = _violations(self._DOMAIN, ["agent_hub.connectors.feishu", "connectors.feishu"])
        assert not violations, _fmt(violations, "pilot/domain 不能 import 飞书连接器")

    def test_no_llm_clients(self) -> None:
        """pilot/domain 不得依赖 LLM 客户端库（anthropic/openai）。"""
        violations = _violations(self._DOMAIN, ["anthropic", "openai"])
        assert not violations, _fmt(violations, "pilot/domain 不能 import LLM 客户端")

    def test_no_agent_pipeline(self) -> None:
        """pilot/domain 不得依赖 AgentPipeline（core 主线）。"""
        violations = _violations(
            self._DOMAIN,
            ["agent_hub.core.pipeline", "agent_hub.agents"],
        )
        assert not violations, _fmt(violations, "pilot/domain 不能 import core pipeline 或 agents")

    def test_no_pilot_services(self) -> None:
        """pilot/domain 不得向上依赖 pilot/services（领域层不能反向依赖应用层）。"""
        violations = _violations(
            self._DOMAIN,
            ["agent_hub.pilot.services", "agent_hub.pilot.skills"],
        )
        assert not violations, _fmt(violations, "pilot/domain 不能 import pilot/services 或 pilot/skills")


# ── 规则 2: capabilities loader 边界 ─────────────────────────────────────────


class TestCapabilitiesBoundary:
    """capabilities/ 是能力加载层，只能读配置文件，不能直接执行外部命令。"""

    _CAPS = _SRC / "capabilities"

    def test_no_subprocess(self) -> None:
        """capabilities loader 不得直接使用 subprocess 执行外部命令。"""
        violations = _violations(self._CAPS, ["subprocess"])
        assert not violations, _fmt(violations, "capabilities 不能 import subprocess")

    def test_no_os_system(self) -> None:
        """capabilities loader 不得使用 os（对 os.system/popen 的防护）。

        注意：只拦截直接 import os，os.path 通过 pathlib 代替。
        """
        # capabilities 模块理论上不需要 import os（应该用 pathlib）。
        # 如果有合理用途（如 os.environ），则移到豁免列表。
        violations = _violations(self._CAPS, ["os"])
        # 过滤掉只用 os.path 或 os.fspath 的合法场景（通过 pathlib 替代更好，
        # 但不强制；这里只是记录而非必须通过）
        # 考虑到部分系统兼容代码可能 import os，仅作为警告性检查。
        # 若有合理 os 用途，可在此 assert 下注释豁免原因。
        assert not violations, _fmt(violations, "capabilities 不应 import os（使用 pathlib 替代）")

    def test_no_pilot_import(self) -> None:
        """capabilities loader 不得依赖 pilot 业务逻辑。"""
        violations = _violations(self._CAPS, ["agent_hub.pilot"])
        assert not violations, _fmt(violations, "capabilities 不能 import pilot")

    def test_no_api_import(self) -> None:
        """capabilities loader 不得依赖 API 层。"""
        violations = _violations(self._CAPS, ["agent_hub.api", "fastapi"])
        assert not violations, _fmt(violations, "capabilities 不能 import api 或 fastapi")


# ── 规则 3: core 层边界 ───────────────────────────────────────────────────────


class TestCoreBoundary:
    """core/ 是 Agent 执行内核，不得与特定连接器实现耦合。"""

    _CORE = _SRC / "core"

    def test_no_feishu_connector(self) -> None:
        """core 不得直接 import 飞书连接器的具体实现。"""
        violations = _violations(
            self._CORE,
            ["agent_hub.connectors.feishu", "connectors.feishu"],
        )
        assert not violations, _fmt(violations, "core 不能直接 import 飞书连接器实现")

    def test_no_fastapi(self) -> None:
        """core 不得依赖 FastAPI（避免执行层与 HTTP 框架耦合）。"""
        violations = _violations(self._CORE, ["fastapi", "starlette"])
        assert not violations, _fmt(violations, "core 不能 import FastAPI/Starlette")

    def test_no_pilot(self) -> None:
        """core 不得依赖 pilot 领域（避免循环依赖）。"""
        violations = _violations(self._CORE, ["agent_hub.pilot"])
        assert not violations, _fmt(violations, "core 不能 import pilot")


# ── 规则 4: connectors 层边界 ─────────────────────────────────────────────────


class TestConnectorsBoundary:
    """connectors/ 是出入站边界，不得侵入 core runtime 内部。"""

    _CONNECTORS = _SRC / "connectors"

    def test_no_core_pipeline(self) -> None:
        """connectors 不得直接 import AgentPipeline（core 主线）。

        connector 只能通过注入的协议接口与业务交互，不能直接持有 pipeline。
        """
        violations = _violations(
            self._CONNECTORS,
            ["agent_hub.core.pipeline"],
        )
        assert not violations, _fmt(violations, "connectors 不能 import core.pipeline")

    def test_no_pilot_domain(self) -> None:
        """connectors 不得直接依赖 pilot.domain 业务实体。

        connector 只能使用 contracts 中的 DTO 与 pilot.application facade 交互。

        已知存量违规（Phase 8 修复，届时从列表移除）：
        - feishu/approval_notifier.py  — 直接使用 Approval, Task 实体
        - feishu/progress_notifier.py  — 直接使用 ExecutionEvent, EventType
        - feishu/service.py            — 直接使用 Workspace, WorkspaceStatus
        """
        _KNOWN_VIOLATIONS: set[str] = {
            "approval_notifier.py",
            "progress_notifier.py",
            "service.py",
        }
        all_violations = _violations(
            self._CONNECTORS,
            ["agent_hub.pilot.domain"],
        )
        new_violations = [v for v in all_violations if v[0].name not in _KNOWN_VIOLATIONS]
        assert not new_violations, _fmt(
            new_violations,
            "connectors 不能 import pilot.domain（新增违规，非存量文件）",
        )


# ── 规则 5: pilot/events 基础设施层 ──────────────────────────────────────────


class TestPilotEventsBoundary:
    """pilot/events/ 是事件基础设施，应对业务服务保持独立。"""

    _EVENTS = _SRC / "pilot" / "events"

    def test_no_fastapi(self) -> None:
        """pilot/events 不得依赖 FastAPI。"""
        violations = _violations(self._EVENTS, ["fastapi"])
        assert not violations, _fmt(violations, "pilot/events 不能 import FastAPI")

    def test_no_feishu(self) -> None:
        """pilot/events 不得依赖飞书连接器。"""
        violations = _violations(self._EVENTS, ["agent_hub.connectors.feishu"])
        assert not violations, _fmt(violations, "pilot/events 不能 import 飞书连接器")

    def test_no_pilot_services(self) -> None:
        """pilot/events 不得依赖 pilot/services（基础设施不能依赖应用服务）。"""
        violations = _violations(self._EVENTS, ["agent_hub.pilot.services"])
        assert not violations, _fmt(violations, "pilot/events 不能 import pilot/services")


# ── 规则 6: contracts 层边界（Phase 2 新增） ─────────────────────────────────


class TestContractsBoundary:
    """contracts/ 是跨层公共契约，不得依赖任何具体实现层。"""

    _CONTRACTS = _SRC / "contracts"

    def test_contracts_no_fastapi(self) -> None:
        """contracts 不得依赖 FastAPI。"""
        violations = _violations(self._CONTRACTS, ["fastapi"])
        assert not violations, _fmt(violations, "contracts 不能 import FastAPI")

    def test_contracts_no_connectors(self) -> None:
        """contracts 不得依赖连接器（feishu 等）。"""
        violations = _violations(self._CONTRACTS, ["agent_hub.connectors"])
        assert not violations, _fmt(violations, "contracts 不能 import connectors")

    def test_contracts_no_pilot_services(self) -> None:
        """contracts 不得依赖 pilot 服务层（防止循环）。"""
        violations = _violations(self._CONTRACTS, ["agent_hub.pilot.services"])
        assert not violations, _fmt(violations, "contracts 不能 import pilot.services")

    def test_contracts_no_agents(self) -> None:
        """contracts 不得依赖 agents 实现层。"""
        violations = _violations(self._CONTRACTS, ["agent_hub.agents"])
        assert not violations, _fmt(violations, "contracts 不能 import agents")

    def test_contracts_no_core(self) -> None:
        """contracts 不得依赖 core 具体实现（应自持模型定义）。

        contracts 是跨层公共契约，必须只依赖 pydantic + stdlib。
        当前 __init__.py 有存量 re-export，创建子模块后会移除。
        """
        violations = _violations(self._CONTRACTS, ["agent_hub.core"])
        assert not violations, _fmt(violations, "contracts 不能 import agent_hub.core")


# ── 规则 7: runtime 层边界（占位，当 runtime/ 目录创建后自动生效）──────────────


class TestRuntimeBoundary:
    """runtime/ 是通用执行引擎，不得依赖 pilot 业务逻辑。"""

    _RUNTIME = _SRC / "runtime"

    def test_no_pilot_import(self) -> None:
        """runtime 不得依赖 pilot（通用引擎不能知道业务领域细节）。"""
        if not self._RUNTIME.exists():
            import pytest
            pytest.skip("runtime/ 尚未创建，测试在 Phase 2 后激活")
        violations = _violations(self._RUNTIME, ["agent_hub.pilot"])
        assert not violations, _fmt(violations, "runtime 不能 import pilot")

    def test_no_connectors_import(self) -> None:
        """runtime 不得依赖 connectors（协议适配器不属于通用引擎）。"""
        if not self._RUNTIME.exists():
            import pytest
            pytest.skip("runtime/ 尚未创建，测试在 Phase 2 后激活")
        violations = _violations(self._RUNTIME, ["agent_hub.connectors"])
        assert not violations, _fmt(violations, "runtime 不能 import connectors")


# ── 辅助 ──────────────────────────────────────────────────────────────────────


def _fmt(violations: list[tuple[Path, str, str]], context: str) -> str:
    lines = [f"架构边界违规 — {context}:"]
    src_root = _SRC.parent.parent  # repo root
    for file_path, module, pattern in violations:
        try:
            rel = file_path.relative_to(src_root)
        except ValueError:
            rel = file_path
        lines.append(f"  {rel}  import '{module}'  (禁止模式: '{pattern}')")
    return "\n".join(lines)
