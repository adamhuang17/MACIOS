"""RAG 检索 Agent。

对接 RAGPipeline 实现 Dense + Sparse 混合知识检索，
支持基于记忆的 query 增强。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from agent_hub.core.base_agent import BaseAgent
from agent_hub.core.models import AgentResult, SubTask

if TYPE_CHECKING:
    from agent_hub.memory import MemoryManager
    from agent_hub.rag.pipeline import RAGPipeline

logger = structlog.get_logger(__name__)


class RetrievalAgent(BaseAgent):
    """从 RAG 知识库检索相关上下文的 Agent。

    Args:
        rag_pipeline: RAG Pipeline 实例。
        top_k: 返回的最大结果数。
        default_namespace: 默认检索命名空间（None 搜全部）。
    """

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        top_k: int = 5,
        default_namespace: str | None = None,
    ) -> None:
        super().__init__(name="retrieval_agent", description="从 RAG 知识库检索上下文")
        self._rag = rag_pipeline
        self._top_k = top_k
        self._default_namespace = default_namespace
        self._memory_manager: MemoryManager | None = None

    def inject_memory(self, memory_manager: MemoryManager) -> None:
        """注入记忆管理器（Pipeline 在初始化时调用）。"""
        self._memory_manager = memory_manager

    async def run(
        self, subtask: SubTask, session_id: str, user_id: str,
    ) -> AgentResult:
        """执行检索任务。"""
        # 1. 构造增强 query
        query = await self._enhance_query(
            subtask.description, session_id, user_id,
        )

        # 2. 调用 RAG Pipeline 混合检索
        try:
            results = await self._rag.retrieve(
                query=query,
                user_id=user_id,
                namespace=self._default_namespace,
                top_k=self._top_k,
            )
        except Exception as e:
            logger.warning(
                "rag_retrieval_failed",
                error=str(e),
                subtask_id=subtask.subtask_id,
            )
            return AgentResult(
                agent_name=self.name,
                success=True,
                output="知识库暂不可用，请稍后再试。",
                duration_ms=0,
                trace_id=subtask.subtask_id,
            )

        # 3. 格式化结果
        if not results:
            output = f"未在知识库中找到与「{subtask.description}」相关的内容。"
        else:
            context_text = self._rag.format_context(results)
            sources = [
                r.metadata.get("source", r.metadata.get("namespace", ""))
                for r in results if r.metadata
            ]
            source_text = ", ".join(s for s in sources if s) or "无来源"
            output = f"【检索结果】\n{context_text}\n\n【来源】{source_text}"

        logger.info(
            "retrieval_success",
            subtask_id=subtask.subtask_id,
            result_count=len(results),
        )

        return AgentResult(
            agent_name=self.name,
            success=True,
            output=output,
            duration_ms=0,
            trace_id=subtask.subtask_id,
        )

    async def _enhance_query(
        self, query: str, session_id: str, user_id: str,
    ) -> str:
        """用记忆上下文增强检索 query。"""
        if not self._memory_manager:
            return query

        recent = self._memory_manager.get_recent(session_id, n=3)
        if not recent:
            return query

        history = "\n".join([f"- {e.content}" for e in recent])
        return f"【历史上下文】\n{history}\n\n【当前问题】\n{query}"
