"""LangGraph 容错检索 Agent。

实现 检索 → 评估 → (重写 → 重新检索)? → 生成 的工作流，
最多 2 轮 query rewrite，超限后使用当前最优结果生成回答。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import structlog
from langgraph.graph import END, StateGraph

from agent_hub.core.base_agent import BaseAgent
from agent_hub.core.models import AgentResult, SubTask
from agent_hub.rag.query_rewriter import QueryRewriter
from agent_hub.rag.relevance_evaluator import RelevanceEvaluator

if TYPE_CHECKING:
    from openai import OpenAI

    from agent_hub.memory import MemoryManager
    from agent_hub.rag.pipeline import RAGPipeline

logger = structlog.get_logger(__name__)

_MAX_REWRITES = 2

_GENERATE_PROMPT = """\
基于以下检索到的上下文回答用户的问题。如果上下文信息不足，请说明并尽力提供有用的回答。

用户问题：{query}

检索上下文：
{context}

请回答："""


# ── LangGraph 状态定义 ──────────────────────────────


class ReflectionState(TypedDict):
    """LangGraph 工作流状态。"""

    query: str
    original_query: str
    user_id: str
    session_id: str
    results: list[Any]  # list[SearchResult]
    score: float
    rewrite_count: int
    context_text: str
    final_answer: str
    error: str


# ── ReflectionAgent ─────────────────────────────────


class ReflectionAgent(BaseAgent):
    """LangGraph 容错检索 Agent。

    工作流：retrieve → evaluate → (rewrite → retrieve)* → generate
    最多 ``_MAX_REWRITES`` 轮重写。

    Args:
        rag_pipeline: RAG Pipeline 实例。
        llm_client: OpenAI 兼容客户端。
        model: LLM 模型名称。
        relevance_threshold: 相关性评分阈值（≥ 此分数跳过重写）。
    """

    def __init__(
        self,
        rag_pipeline: RAGPipeline | None = None,
        llm_client: OpenAI | None = None,
        model: str = "glm-4-flash",
        relevance_threshold: float = 3.0,
    ) -> None:
        super().__init__(
            name="reflection_agent",
            description="LangGraph 容错检索：检索→评估→重写→生成",
        )
        self._rag = rag_pipeline
        self._client = llm_client
        self._model = model
        self._rewriter = QueryRewriter(llm_client, model)
        self._evaluator = RelevanceEvaluator(llm_client, model, relevance_threshold)
        self._memory_manager: MemoryManager | None = None
        self._graph = self._build_graph()

    def inject_memory(self, memory_manager: MemoryManager) -> None:
        """注入记忆管理器。"""
        self._memory_manager = memory_manager

    # ── BaseAgent 入口 ────────────────────────────────

    async def run(
        self, subtask: SubTask, session_id: str, user_id: str,
    ) -> AgentResult:
        """执行 LangGraph 容错检索工作流。"""
        initial_state: ReflectionState = {
            "query": subtask.description,
            "original_query": subtask.description,
            "user_id": user_id,
            "session_id": session_id,
            "results": [],
            "score": 0.0,
            "rewrite_count": 0,
            "context_text": "",
            "final_answer": "",
            "error": "",
        }

        try:
            final_state = await self._graph.ainvoke(initial_state)

            if final_state.get("error"):
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    error=final_state["error"],
                    duration_ms=0,
                    trace_id=subtask.subtask_id,
                )

            answer = final_state.get("final_answer", "")
            output_parts = [answer]
            if final_state.get("rewrite_count", 0) > 0:
                output_parts.append(
                    f"\n（经过 {final_state['rewrite_count']} 轮 query 重写优化检索）"
                )

            return AgentResult(
                agent_name=self.name,
                success=True,
                output="\n".join(output_parts),
                duration_ms=0,
                trace_id=subtask.subtask_id,
            )

        except Exception as exc:
            logger.error("reflection_agent_error", error=str(exc))
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=f"反思检索工作流异常: {exc}",
                duration_ms=0,
                trace_id=subtask.subtask_id,
            )

    # ── LangGraph 构建 ───────────────────────────────

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph StateGraph 工作流。

        Nodes: retrieve → evaluate → rewrite → generate
        Edges:
          - evaluate → generate  (score ≥ threshold)
          - evaluate → rewrite   (score < threshold AND rewrite_count < MAX)
          - evaluate → generate  (score < threshold AND rewrite_count ≥ MAX)
          - rewrite  → retrieve
        """
        graph = StateGraph(ReflectionState)

        # 注册节点
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("evaluate", self._node_evaluate)
        graph.add_node("rewrite", self._node_rewrite)
        graph.add_node("generate", self._node_generate)

        # 入口
        graph.set_entry_point("retrieve")

        # 边
        graph.add_edge("retrieve", "evaluate")
        graph.add_conditional_edges(
            "evaluate",
            self._should_rewrite,
            {"rewrite": "rewrite", "generate": "generate"},
        )
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("generate", END)

        return graph.compile()

    # ── 节点实现 ─────────────────────────────────────

    async def _node_retrieve(self, state: ReflectionState) -> dict:
        """检索节点：调用 RAG Pipeline。"""
        try:
            results = await self._rag.retrieve(
                query=state["query"],
                user_id=state["user_id"],
            )
            context_text = self._rag.format_context(results)
            logger.info(
                "reflection.retrieve",
                query=state["query"][:50],
                result_count=len(results),
                rewrite_round=state["rewrite_count"],
            )
            return {"results": results, "context_text": context_text}

        except Exception as exc:
            logger.warning("reflection.retrieve_failed", error=str(exc))
            return {"results": [], "context_text": "", "error": str(exc)}

    async def _node_evaluate(self, state: ReflectionState) -> dict:
        """评估节点：LLM 打分检索结果的相关性。"""
        if state.get("error"):
            return {"score": 0.0}

        if not state["results"]:
            return {"score": 1.0}

        try:
            score = await self._evaluator.evaluate(
                state["query"], state["results"],
            )
            logger.info(
                "reflection.evaluate",
                score=score,
                threshold=self._evaluator.threshold,
            )
            return {"score": score}

        except Exception as exc:
            logger.warning("reflection.evaluate_failed", error=str(exc))
            # 评估失败时给中性分数，让 generate 处理
            return {"score": self._evaluator.threshold}

    async def _node_rewrite(self, state: ReflectionState) -> dict:
        """重写节点：LLM 优化 query。"""
        rewrite_count = state["rewrite_count"] + 1

        try:
            rewritten = await self._rewriter.rewrite(
                original_query=state["original_query"],
                context=state["context_text"][:500],
            )
            logger.info(
                "reflection.rewrite",
                round=rewrite_count,
                original=state["original_query"][:50],
                rewritten=rewritten[:50],
            )
            return {"query": rewritten, "rewrite_count": rewrite_count}

        except Exception as exc:
            logger.warning("reflection.rewrite_failed", error=str(exc))
            return {"rewrite_count": rewrite_count}

    async def _node_generate(self, state: ReflectionState) -> dict:
        """生成节点：基于检索上下文用 LLM 生成最终回答。"""
        context = state.get("context_text", "")
        query = state.get("original_query", state["query"])

        if not context and state.get("error"):
            return {"final_answer": f"检索过程中遇到错误：{state['error']}，无法生成回答。"}

        prompt = _GENERATE_PROMPT.format(query=query, context=context or "（无检索结果）")

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048,
            )
            answer = resp.choices[0].message.content.strip()
            return {"final_answer": answer}

        except Exception as exc:
            logger.error("reflection.generate_failed", error=str(exc))
            if context:
                return {"final_answer": f"生成回答失败，以下是检索到的相关内容：\n\n{context}"}
            return {"final_answer": f"抱歉，检索和生成均失败: {exc}"}

    # ── 条件路由 ──────────────────────────────────────

    def _should_rewrite(self, state: ReflectionState) -> str:
        """条件路由：决定是重写还是直接生成。"""
        if state.get("error"):
            return "generate"

        if state["score"] >= self._evaluator.threshold:
            return "generate"

        if state["rewrite_count"] >= _MAX_REWRITES:
            logger.info(
                "reflection.max_rewrites_reached",
                rewrite_count=state["rewrite_count"],
                score=state["score"],
            )
            return "generate"

        return "rewrite"
