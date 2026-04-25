"""工具注册表。

提供全局工具注册、查询和权限过滤功能，并预置常用工具。
"""

from __future__ import annotations

import ast
import asyncio
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from agent_hub.core.models import ToolSpec

logger = structlog.get_logger(__name__)

ToolFunction = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass
class RegisteredTool:
    """已注册工具：规范描述 + 实际执行函数。"""

    spec: ToolSpec
    func: ToolFunction


class ToolRegistry:
    """全局工具注册表。"""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        name: str,
        description: str,
        params_schema: dict[str, Any],
        func: ToolFunction,
        requires_admin: bool = False,
    ) -> None:
        """注册一个工具。"""
        spec = ToolSpec(
            name=name,
            description=description,
            params_schema=params_schema,
            requires_admin=requires_admin,
        )
        self._tools[name] = RegisteredTool(spec=spec, func=func)
        logger.debug("tool_registered", tool=name, requires_admin=requires_admin)

    def get(self, name: str) -> RegisteredTool | None:
        """获取工具。"""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSpec]:
        """列出所有已注册工具。"""
        return [rt.spec for rt in self._tools.values()]

    def list_tools_for_user(self, role: str) -> list[ToolSpec]:
        """列出当前用户可用的工具（过滤 admin 专属工具）。"""
        return [
            rt.spec
            for rt in self._tools.values()
            if not rt.spec.requires_admin or role == "admin"
        ]

    def register_defaults(self) -> None:
        """注册所有预置工具。"""
        self.register(
            name="calculator",
            description="计算器，支持加减乘除幂运算",
            params_schema={"expression": {"type": "string", "description": "数学表达式"}},
            func=calculator_impl,
            requires_admin=False,
        )
        self.register(
            name="file_read",
            description="读取文件内容",
            params_schema={"path": {"type": "string", "description": "文件路径"}},
            func=file_read_impl,
            requires_admin=False,
        )
        self.register(
            name="file_write",
            description="写入文件内容",
            params_schema={
                "path": {"type": "string", "description": "文件路径"},
                "content": {"type": "string", "description": "写入内容"},
            },
            func=file_write_impl,
            requires_admin=True,
        )
        self.register(
            name="system_command",
            description="执行系统命令",
            params_schema={"command": {"type": "string", "description": "系统命令"}},
            func=system_command_impl,
            requires_admin=True,
        )
        self.register(
            name="get_current_time",
            description="获取当前时间",
            params_schema={},
            func=get_current_time_impl,
            requires_admin=False,
        )
        self.register(
            name="search_web",
            description="搜索网络（模拟）",
            params_schema={"query": {"type": "string", "description": "搜索关键词"}},
            func=search_web_impl,
            requires_admin=False,
        )


# ── 安全数学表达式求值 ───────────────────────────────

# ast 节点白名单：只允许数字、一元运算、二元运算
_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.Constant,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
)


def _safe_eval_expr(expr: str) -> float | int:
    """安全求值数学表达式。

    使用 ``ast.parse`` 解析并校验 AST 节点白名单，
    仅允许数字常量和基本算术运算，拒绝函数调用、属性访问等。

    Raises:
        ValueError: 表达式包含不允许的语法。
    """
    tree = ast.parse(expr.strip(), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"不允许的表达式节点: {type(node).__name__}")
        # 额外校验 Pow 的右操作数不能过大，防止 DoS
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Pow)
            and isinstance(node.right, ast.Constant)
            and isinstance(node.right.value, int | float)
            and abs(node.right.value) > 1000
        ):
            raise ValueError("幂指数过大，最大允许 1000")
    code = compile(tree, "<expr>", "eval")
    return eval(code, {"__builtins__": {}})  # noqa: S307


# ── 预置工具实现 ─────────────────────────────────────

# 文件操作的安全根目录
_SAFE_ROOT = Path("./data").resolve()


def _validate_path(path_str: str) -> Path:
    """校验文件路径在安全根目录之内，防止路径穿越。

    Raises:
        PermissionError: 路径不在安全根目录内。
    """
    resolved = Path(path_str).resolve()
    if not str(resolved).startswith(str(_SAFE_ROOT)):
        raise PermissionError(f"路径不在允许范围内: {path_str}")
    return resolved


async def calculator_impl(params: dict[str, Any]) -> str:
    """计算器工具实现（安全 AST 求值）。"""
    expr = params.get("expression", "") or params.get("value", "")
    try:
        result = _safe_eval_expr(str(expr))
        return f"计算结果：{expr} = {result}"
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as e:
        return f"计算错误：{e}"


async def file_read_impl(params: dict[str, Any]) -> str:
    """读取文件内容。"""
    path_str = params.get("path", "") or params.get("value", "")
    try:
        file_path = _validate_path(str(path_str))
        if not file_path.exists():
            return f"错误：文件不存在 {path_str}"
        content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
        return content
    except PermissionError as e:
        return f"权限错误：{e}"
    except Exception as e:
        return f"读取文件错误：{e}"


async def file_write_impl(params: dict[str, Any]) -> str:
    """写入文件内容。"""
    path_str = params.get("path", "") or params.get("value", "")
    content = params.get("content", "")
    try:
        file_path = _validate_path(str(path_str))
        file_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(file_path.write_text, str(content), encoding="utf-8")
        return f"文件写入成功：{path_str}"
    except PermissionError as e:
        return f"权限错误：{e}"
    except Exception as e:
        return f"写入文件错误：{e}"


async def system_command_impl(params: dict[str, Any]) -> str:
    """执行系统命令（shell=False，带 timeout）。"""
    command = params.get("command", "") or params.get("value", "")
    try:
        cmd_parts = str(command).split()
        if not cmd_parts:
            return "错误：命令为空"
        result = await asyncio.to_thread(
            subprocess.run,
            cmd_parts,
            capture_output=True,
            text=True,
            timeout=30,
            shell=False,
        )
        output = result.stdout or result.stderr or "(无输出)"
        return f"命令执行完成（退出码 {result.returncode}）：\n{output}"
    except subprocess.TimeoutExpired:
        return "错误：命令执行超时（30s）"
    except Exception as e:
        return f"命令执行错误：{e}"


async def get_current_time_impl(params: dict[str, Any]) -> str:
    """获取当前时间。"""
    now = datetime.now(tz=UTC).astimezone()
    return f"当前时间：{now:%Y-%m-%d %H:%M:%S %Z}"


async def search_web_impl(params: dict[str, Any]) -> str:
    """模拟网络搜索（占位实现）。"""
    query = params.get("query", "") or params.get("value", "")
    return f"搜索结果（模拟）：关于「{query}」的相关信息暂不可用，请接入真实搜索 API。"
