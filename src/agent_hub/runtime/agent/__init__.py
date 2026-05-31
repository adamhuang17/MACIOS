"""runtime.agent — Agent Actor 基类与协议。

定义 BaseAgent 协议，约束所有 Agent Actor 必须实现的接口。
具体的 LLMAgent、ToolAgent、RetrievalAgent、ReflectionAgent 在 Phase 3 迁移至此。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agent_hub.contracts.execution import AgentResult, TaskInput


class BaseAgent(ABC):
    """所有 Agent Actor 的抽象基类。

    子类必须实现 :meth:`run`。
    ``agent_name`` 用于日志和 AgentResult 标识。
    """

    agent_name: str = "base"

    @abstractmethod
    async def run(self, task: TaskInput, context: dict[str, Any]) -> AgentResult:
        """执行 Agent 逻辑，返回 :class:`AgentResult`。

        Args:
            task: 当前任务输入（含路由决策等上下文已在 context 中）。
            context: 运行时上下文，包含 routing_decision、memory_entries、
                     tool_results 等中间结果。

        Returns:
            AgentResult，供 Pipeline 汇总。
        """


__all__ = [
    "BaseAgent",
]
