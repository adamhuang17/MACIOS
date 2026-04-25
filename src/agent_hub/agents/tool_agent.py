"""工具执行 Agent。

从 SubTask 描述中解析工具名和参数，
校验权限后调用注册表中的工具函数。
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from agent_hub.agents.registry import ToolRegistry
from agent_hub.core.base_agent import BaseAgent
from agent_hub.core.models import AgentResult, SubTask

logger = structlog.get_logger(__name__)


class ToolAgent(BaseAgent):
    """执行工具调用的 Agent。

    Args:
        registry: 工具注册表实例。
    """

    def __init__(self, registry: ToolRegistry) -> None:
        super().__init__(name="tool_agent", description="执行预注册的工具")
        self._registry = registry
        self._user_role: str = "user"

    def set_user_role(self, role: str) -> None:
        """设置当前用户角色（Pipeline 在调用前设置）。"""
        self._user_role = role

    async def run(
        self, subtask: SubTask, session_id: str, user_id: str,
    ) -> AgentResult:
        """执行工具调用任务。"""
        # 1. 解析工具名和参数
        parsed = self._parse_tool_call(subtask.description)
        if not parsed:
            logger.warning(
                "tool_parse_failed",
                description=subtask.description,
                subtask_id=subtask.subtask_id,
            )
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=f"无法从描述中解析工具调用: {subtask.description}",
                duration_ms=0,
                trace_id=subtask.subtask_id,
            )

        tool_name, params = parsed

        # 2. 检查工具是否存在
        tool = self._registry.get(tool_name)
        if not tool:
            available = [t.name for t in self._registry.list_tools()]
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=f"工具 '{tool_name}' 不存在。可用工具: {available}",
                duration_ms=0,
                trace_id=subtask.subtask_id,
            )

        # 3. 检查权限
        if not self._check_permission(tool_name, self._user_role):
            logger.warning(
                "tool_permission_denied",
                tool=tool_name,
                user_id=user_id,
                role=self._user_role,
            )
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=f"权限不足：工具 '{tool_name}' 需要管理员权限",
                duration_ms=0,
                trace_id=subtask.subtask_id,
            )

        # 4. 执行工具
        try:
            result = await tool.func(params)
            logger.info(
                "tool_executed",
                tool=tool_name,
                subtask_id=subtask.subtask_id,
            )
            return AgentResult(
                agent_name=self.name,
                success=True,
                output=result,
                duration_ms=0,
                trace_id=subtask.subtask_id,
            )
        except Exception as e:
            logger.error(
                "tool_execution_error",
                tool=tool_name,
                error=str(e),
                subtask_id=subtask.subtask_id,
            )
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=f"工具 '{tool_name}' 执行失败: {e}",
                duration_ms=0,
                trace_id=subtask.subtask_id,
            )

    def _parse_tool_call(
        self, description: str,
    ) -> tuple[str, dict[str, Any]] | None:
        """从描述中解析工具名和参数。

        支持格式: ``工具名(参数)``，正确处理嵌套括号。

        Examples:
            - ``calculator((2+3)*4)`` → ``("calculator", {"expression": "(2+3)*4"})``
            - ``file_read(path=/tmp/a.txt)`` → ``("file_read", {"path": "/tmp/a.txt"})``
        """
        # 先匹配工具名 + 左括号
        name_match = re.search(r"(\w+)\(", description)
        if not name_match:
            return None

        tool_name = name_match.group(1)
        start = name_match.end()  # '(' 后一位

        # 平衡括号匹配：找到对应的右括号
        depth = 1
        pos = start
        while pos < len(description) and depth > 0:
            if description[pos] == "(":
                depth += 1
            elif description[pos] == ")":
                depth -= 1
            pos += 1

        if depth != 0:
            return None

        raw_args = description[start : pos - 1].strip()

        if not raw_args:
            return tool_name, {}

        # 支持 key=value 和纯值
        if "=" in raw_args:
            params: dict[str, Any] = {}
            for kv in raw_args.split(","):
                kv = kv.strip()
                if "=" in kv:
                    key, value = kv.split("=", 1)
                    params[key.strip()] = value.strip()
            return tool_name, params

        return tool_name, {"value": raw_args}

    def _check_permission(self, tool_name: str, user_role: str) -> bool:
        """检查用户是否有权限调用该工具。"""
        tool = self._registry.get(tool_name)
        return bool(tool) and (
            not tool.spec.requires_admin or user_role == "admin"
        )
