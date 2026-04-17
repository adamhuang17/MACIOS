"""Agent 抽象基类。

所有具体 Agent（路由、检索、工具调用等）均继承 ``BaseAgent``，
只需实现 ``run()`` 方法；通用的计时、异常捕获、结构化返回
由 ``execute()`` 模板方法统一处理。
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from agent_hub.core.models import AgentResult, SubTask, ToolSpec

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Agent 抽象基类。

    子类必须实现 ``run()`` 方法来定义具体的业务逻辑。
    外部调用统一通过 ``execute()`` 入口，该方法会自动完成：

    1. 前置日志记录
    2. 执行计时
    3. 调用子类 ``run()``
    4. 异常捕获 → 构造失败 ``AgentResult``

    Attributes:
        name: Agent 名称，用于日志和结果标识。
        description: Agent 功能的自然语言描述。
        tools: 该 Agent 可使用的工具列表。

    Example::

        class EchoAgent(BaseAgent):
            def __init__(self) -> None:
                super().__init__(
                    name="echo",
                    description="原样返回用户输入",
                )

            async def run(
                self, subtask: SubTask, session_id: str, user_id: str,
            ) -> AgentResult:
                return AgentResult(
                    agent_name=self.name,
                    success=True,
                    output=subtask.description,
                    duration_ms=0,
                    trace_id=subtask.subtask_id,
                )
    """

    def __init__(
        self,
        name: str,
        description: str,
        tools: list[ToolSpec] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.tools: list[ToolSpec] = tools or []

    @abstractmethod
    async def run(
        self, subtask: SubTask, session_id: str, user_id: str,
    ) -> AgentResult:
        """执行核心业务逻辑（子类实现）。

        Args:
            subtask: 路由层拆分出的子任务。
            session_id: 会话唯一标识。
            user_id: 用户唯一标识。

        Returns:
            AgentResult: Agent 执行结果。
        """
        ...

    async def execute(
        self, subtask: SubTask, session_id: str, user_id: str,
    ) -> AgentResult:
        """模板方法：统一的执行入口。

        封装计时、日志、异常捕获，子类无需关注这些横切关注点。

        Args:
            subtask: 路由层拆分出的子任务。
            session_id: 会话唯一标识。
            user_id: 用户唯一标识。

        Returns:
            AgentResult: 成功时返回 ``run()`` 的结果；
            失败时返回带有错误信息的 ``AgentResult``。
        """
        logger.info(
            "[%s] 开始执行 | subtask=%s | user=%s",
            self.name,
            subtask.subtask_id,
            user_id,
        )

        start_ms = time.monotonic_ns() // 1_000_000

        try:
            result = await self.run(subtask, session_id, user_id)
        except Exception:
            elapsed_ms = time.monotonic_ns() // 1_000_000 - start_ms
            logger.exception(
                "[%s] 执行异常 | subtask=%s", self.name, subtask.subtask_id,
            )
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=f"{self.name} 执行失败，请查看日志",
                duration_ms=elapsed_ms,
                trace_id=subtask.subtask_id,
            )

        elapsed_ms = time.monotonic_ns() // 1_000_000 - start_ms

        # 回填实际耗时（子类可能填了占位值）
        result.duration_ms = elapsed_ms

        logger.info(
            "[%s] 执行完成 | subtask=%s | success=%s | %dms",
            self.name,
            subtask.subtask_id,
            result.success,
            elapsed_ms,
        )

        return result

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} tools={len(self.tools)}>"
