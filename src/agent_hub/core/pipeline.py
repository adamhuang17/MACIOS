"""主控 Pipeline。

接收 ``TaskInput``，串联 **路由 → 权限 → Agent 执行 → 记忆写入**，
返回 ``TaskOutput``。所有步骤均在 :class:`SpanContext` 追踪下运行。

无依赖的子任务通过 ``asyncio.gather`` 并行执行（DAG 拓扑排序分层）。

Example::

    from agent_hub.config.settings import get_settings
    from agent_hub.core.pipeline import AgentPipeline
    from agent_hub.core.models import SourceChatType, SourceContext, TaskInput, UserContext
    from agent_hub.core.enums import UserRole

    pipeline = AgentPipeline(get_settings())
    output = await pipeline.run(TaskInput(
        user_context=UserContext(user_id="u1", role=UserRole.USER),
        source_context=SourceContext(
            channel="api",
            chat_id="u1",
            chat_type=SourceChatType.DIRECT,
            sender_id="u1",
        ),
        raw_message="帮我写一个二分查找",
    ))
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from agent_hub.agents import (
    LLMAgent,
    ReflectionAgent,
    RetrievalAgent,
    ToolAgent,
    ToolRegistry,
)
from agent_hub.core.binding import BindingResult, SourceBindingResolver
from agent_hub.core.enums import ExecutionMode, UserRole
from agent_hub.core.models import (
    AgentResult,
    Artifact,
    GuardResult,
    MemoryEntry,
    RoutingDecision,
    SourceChatType,
    SourceContext,
    SubTask,
    TaskInput,
    TaskOutput,
)
from agent_hub.core.plan_validator import SubTaskPlanValidator
from agent_hub.core.rate_limiter import RateLimiter
from agent_hub.core.risk import RiskDecision, RiskPolicy
from agent_hub.core.router import DecisionRouter
from agent_hub.core.tracer import SpanContext, get_current_trace_id
from agent_hub.memory import MemoryManager
from agent_hub.rag.pipeline import RAGPipeline
from agent_hub.security.guard import PromptGuard

if TYPE_CHECKING:
    from agent_hub.config.settings import Settings
    from agent_hub.core.base_agent import BaseAgent
    from agent_hub.core.cache import CacheLayer

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class _PreparedExecution:
    source_context: SourceContext
    binding: BindingResult | None = None
    routing: RoutingDecision | None = None
    risk: RiskDecision | None = None
    guard_result: GuardResult | None = None
    blocked_response: str | None = None
    blocked_status: str | None = None


# ── DAG 拓扑排序工具 ─────────────────────────────────


def _topological_layers(sub_tasks: list[SubTask]) -> list[list[SubTask]]:
    """将子任务按 DAG 依赖分层。

    同层内的子任务无相互依赖，可安全并行执行。

    Args:
        sub_tasks: 包含 ``depends_on`` 关系的子任务列表。

    Returns:
        分层后的子任务列表，每层为一组可并行执行的任务。
    """
    if not sub_tasks:
        return []

    task_map: dict[str, SubTask] = {t.subtask_id: t for t in sub_tasks}
    in_degree: dict[str, int] = {t.subtask_id: 0 for t in sub_tasks}
    dependents: dict[str, list[str]] = defaultdict(list)

    for t in sub_tasks:
        for dep in t.depends_on:
            if dep in task_map:
                in_degree[t.subtask_id] += 1
                dependents[dep].append(t.subtask_id)

    layers: list[list[SubTask]] = []
    ready = [tid for tid, deg in in_degree.items() if deg == 0]

    while ready:
        layer = [task_map[tid] for tid in ready]
        layers.append(layer)
        next_ready: list[str] = []
        for tid in ready:
            for child in dependents[tid]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    next_ready.append(child)
        ready = next_ready

    return layers


# ── AgentPipeline ────────────────────────────────────


class AgentPipeline:
    """Agent Pipeline：接收 TaskInput，串联路由→记忆→Agent执行→结果输出。

    Args:
        settings: 全局配置对象。
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        # 路由器
        self._router = DecisionRouter(settings=settings)
        self._binding = SourceBindingResolver()
        self._risk_policy = RiskPolicy()
        self._plan_validator = SubTaskPlanValidator()

        # 记忆管理器
        self._memory = MemoryManager(vault_path=settings.obsidian_vault_path)

        # 工具注册表
        self._registry = ToolRegistry()
        self._registry.register_defaults()

        # RAG Pipeline is cheap to construct: embedding weights and database
        # connections are opened lazily on first use.
        self._rag_pipeline: RAGPipeline | None = None
        self._ensure_rag_pipeline()

        # Agent 实例
        llm_agent = LLMAgent(
            default_model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            max_react_rounds=settings.react_max_rounds,
        )
        self._agents: dict[str, BaseAgent] = {
            "llm_agent": llm_agent,
            "retrieval_agent": RetrievalAgent(
                rag_pipeline=self._rag_pipeline,
                top_k=settings.rag_top_k,
            ),
            "tool_agent": ToolAgent(self._registry),
            "reflection_agent": ReflectionAgent(
                rag_pipeline=self._rag_pipeline,
                llm_client=llm_agent._primary,
                model=settings.llm_model,
            ),
        }

        # 注入记忆管理器
        for agent in self._agents.values():
            if hasattr(agent, "inject_memory"):
                agent.inject_memory(self._memory)

        # 注入工具注册表到 LLMAgent（启用 ReAct 推理循环）
        llm_agent = self._agents["llm_agent"]
        if hasattr(llm_agent, "inject_registry"):
            llm_agent.inject_registry(self._registry)

        # Prompt 注入守卫
        self._guard: PromptGuard | None = None
        if settings.guard_enabled:
            self._guard = PromptGuard(
                settings=settings,
                llm_enabled=settings.guard_llm_enabled,
            )

        # 三级缓存（feature flag 控制；连接失败时静默降级）
        self._cache: CacheLayer | None = None
        if getattr(settings, "cache_enabled", False) is True:
            try:
                from agent_hub.core.cache import CacheLayer

                self._cache = CacheLayer(
                    redis_url=settings.redis_url,
                    embedder=(
                        self._rag_pipeline.embedder
                        if self._rag_pipeline is not None
                        else None
                    ),
                    ttl=settings.cache_ttl,
                    semantic_threshold=settings.semantic_cache_threshold,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("cache_layer_init_failed", error=str(exc))

        # LLM 并发限流（暂作为属性持有，后续在 LLMAgent 调用包裹）
        self._rate_limiter: RateLimiter | None = None
        if getattr(settings, "rate_limiter_enabled", False) is True:
            self._rate_limiter = RateLimiter(
                max_concurrent=settings.llm_max_concurrent,
                default_timeout=float(settings.llm_call_timeout),
            )

        # 向量记忆层（可选；连接失败时静默降级）
        if getattr(settings, "vector_memory_enabled", False) is True:
            self._try_inject_vector_memory()

        logger.info(
            "pipeline_initialized",
            agents=list(self._agents.keys()),
            rag_enabled=self._rag_pipeline is not None,
            cache_enabled=self._cache is not None,
            rate_limiter_enabled=self._rate_limiter is not None,
        )

    # ── 可选基础设施访问器 ──────────────────────────

    @property
    def cache(self) -> CacheLayer | None:
        return self._cache

    @property
    def rate_limiter(self) -> RateLimiter | None:
        return self._rate_limiter

    @property
    def rag_pipeline(self) -> RAGPipeline:
        return self._ensure_rag_pipeline()

    def _ensure_rag_pipeline(self) -> RAGPipeline:
        if self._rag_pipeline is None:
            self._rag_pipeline = RAGPipeline(self._settings)
        for agent_name in ("retrieval_agent", "reflection_agent"):
            agent = getattr(self, "_agents", {}).get(agent_name)
            if agent is not None and hasattr(agent, "_rag"):
                agent._rag = self._rag_pipeline
        return self._rag_pipeline

    def _try_inject_vector_memory(self) -> None:
        """尝试装配 VectorMemory + ConflictDetector + MemoryOperator。

        任意一步失败都静默降级，pipeline 继续以 session+persistent 两层运行。
        """
        try:
            from agent_hub.memory.conflict_detector import ConflictDetector
            from agent_hub.memory.memory_ops import MemoryOperator
            from agent_hub.memory.vector_memory import VectorMemory
            from agent_hub.rag.embedder import Embedder

            embedder = Embedder(model_name=self._settings.embedding_model)
            vector_memory = VectorMemory(
                pg_dsn=self._settings.pg_dsn,
                embedder=embedder,
            )
            conflict_detector = ConflictDetector(embedder=embedder)
            memory_operator = MemoryOperator(
                vector_memory=vector_memory,
                conflict_detector=conflict_detector,
            )
            self._memory.inject_vector_memory(
                vector_memory=vector_memory,
                conflict_detector=conflict_detector,
                memory_operator=memory_operator,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("vector_memory_inject_failed", error=str(exc))

    # ── 执行准备：guard → binding → router → risk ─────────

    async def _prepare_execution(
        self,
        task_input: TaskInput,
        trace_id: str,
        root_span: SpanContext | None = None,
    ) -> _PreparedExecution:
        """Build once, consume in both ``run`` and ``run_stream``."""
        source_context = self._resolve_source_context(task_input)
        user_ctx = task_input.user_context

        guard_result: GuardResult | None = None
        if self._guard is not None:
            with SpanContext("step.guard", trace_id):
                guard_result = await self._guard.check(task_input.raw_message)
                if root_span is not None:
                    root_span.add_event("guard_done", {
                        "risk_level": guard_result.risk_level,
                        "is_safe": guard_result.is_safe,
                    })
                if not guard_result.is_safe:
                    return _PreparedExecution(
                        source_context=source_context,
                        guard_result=guard_result,
                        blocked_response=self._guard_blocked_response(guard_result),
                        blocked_status="blocked",
                    )

        with SpanContext("step.binding", trace_id):
            binding = self._binding.resolve(source_context, user_ctx)
            if root_span is not None:
                root_span.add_event("binding_done", {
                    "agent_id": binding.agent_id,
                    "route_source": binding.route_source,
                    "session_key": binding.session_key,
                })

        with SpanContext("step.router", trace_id):
            routing = await self._router.route(
                task_input.raw_message,
                user_ctx,
                source_context=source_context,
            )
            routing = self._promote_addressed_ignore(task_input, routing)
            risk = self._risk_policy.assess(
                user_context=user_ctx,
                source_context=source_context,
                binding=binding,
                routing=routing,
                guard_result=guard_result,
            )
            routing = self._risk_policy.apply(
                routing=routing,
                binding=binding,
                decision=risk,
            )
            if root_span is not None:
                root_span.add_event("routing_done", {
                    "mode": routing.mode.value,
                    "confidence": routing.confidence,
                    "plan_count": len(routing.plan),
                    "route_source": routing.route_source,
                    "tool_profile": routing.tool_profile,
                    "risk_level": routing.risk_level,
                    "requires_approval": routing.requires_approval,
                })

        return _PreparedExecution(
            source_context=source_context,
            binding=binding,
            routing=routing,
            risk=risk,
            guard_result=guard_result,
        )

    @staticmethod
    def _resolve_source_context(task_input: TaskInput) -> SourceContext:
        return task_input.source_context

    @staticmethod
    def _guard_blocked_response(guard_result: GuardResult) -> str:
        return (
            "⚠️ 检测到潜在安全风险，请求已被拦截。"
            f"（规则命中: {guard_result.matched_rules}）"
        )

    @staticmethod
    def _configure_agent_context(
        agent: BaseAgent,
        user_ctx: object,
        routing: RoutingDecision,
    ) -> None:
        role = getattr(getattr(user_ctx, "role", None), "value", None) or getattr(
            user_ctx, "role", "user",
        )
        if hasattr(agent, "set_user_role"):
            agent.set_user_role(str(role))
        if hasattr(agent, "set_tool_profile"):
            agent.set_tool_profile(routing.tool_profile)

    # ── 流式入口 ─────────────────────────────────────

    async def run_stream(
        self,
        task_input: TaskInput,
    ) -> AsyncGenerator[dict[str, str], None]:
        """Pipeline 流式入口：Guard → 路由 → 权限 → Agent → 逐 token yield。

        与 ``run()`` 流程保持一致：先做安全检测、admin 权限校验，再串
        Router → DAG 流式执行；执行完成后写入记忆。仅 LLMAgent 支持
        真正的 token 级流式，其他 Agent 一次性返回。

        Args:
            task_input: 用户请求。

        Yields:
            dict: ``{"type": "status"|"token"|"error"|"done", "content": "..."}``
        """
        trace_id = task_input.trace_id or get_current_trace_id()
        user_ctx = task_input.user_context
        prepared = await self._prepare_execution(task_input, trace_id)
        if prepared.blocked_response is not None:
            yield {"type": "error", "content": prepared.blocked_response}
            return
        if prepared.routing is None or prepared.binding is None:
            yield {"type": "error", "content": "执行准备失败：缺少路由结果"}
            return

        routing = prepared.routing
        session_id = routing.session_key or prepared.binding.session_key

        if routing.mode == ExecutionMode.IGNORE or routing.low_confidence:
            yield {"type": "token", "content": self._build_fallback_response(routing)}
            return

        # ── Step 2: admin 权限校验（与 run() 一致） ──
        if routing.requires_admin and user_ctx.role != UserRole.ADMIN:
            yield {
                "type": "error",
                "content": "⚠️ 权限不足：该操作仅限管理员使用。",
            }
            return

        if not routing.plan:
            routing = self._ensure_default_subtask(routing, task_input.raw_message)

        routing = self._postprocess_subtasks(routing)
        validation = self._plan_validator.validate(
            routing.plan,
            tool_profile=routing.tool_profile,
        )
        if not validation.valid:
            yield {
                "type": "error",
                "content": f"计划校验失败：{'; '.join(validation.errors)}",
            }
            return

        # ── Step 3: 流式执行 ────────────────────────
        collected_outputs: list[str] = []
        for subtask in routing.plan:
            agent_type = subtask.required_agents[0] if subtask.required_agents else "llm_agent"
            agent = self._agents.get(agent_type)

            if agent is None:
                yield {"type": "token", "content": f"[未知 Agent: {agent_type}]"}
                continue

            self._configure_agent_context(agent, user_ctx, routing)

            # 如果 agent 支持 run_stream，使用流式
            if hasattr(agent, "run_stream"):
                async for chunk in agent.run_stream(subtask, session_id, user_ctx.user_id):
                    if chunk.get("type") == "token" and chunk.get("content"):
                        collected_outputs.append(str(chunk["content"]))
                    yield chunk
            else:
                result = await agent.run(subtask, session_id, user_ctx.user_id)
                if result.success and result.output:
                    text = str(result.output)
                    collected_outputs.append(text)
                    yield {"type": "token", "content": text}
                elif result.error:
                    yield {"type": "error", "content": result.error}

        # ── Step 4: 记忆写入（与 run() 对齐） ────────
        try:
            self._memory.add(
                session_id=session_id,
                user_id=user_ctx.user_id,
                content=f"用户: {task_input.raw_message}",
                memory_type="short_term",
                tags=["user_request", routing.mode.value, "stream"],
            )
            if collected_outputs:
                self._memory.add(
                    session_id=session_id,
                    user_id=user_ctx.user_id,
                    content="".join(collected_outputs),
                    memory_type="long_term",
                    tags=["execution_summary", routing.mode.value, "stream"],
                )
        except Exception as exc:  # 记忆失败不应阻塞流式响应
            logger.warning("stream_memory_save_failed", error=str(exc))

    # ── 主入口 ───────────────────────────────────────

    async def run(self, task_input: TaskInput) -> TaskOutput:
        """Pipeline 主入口。

        完整流程：

        1. 路由决策（DecisionRouter）
        2. GROUP_CHAT / 低置信度 fallback
        3. 权限校验（ADMIN_COMMAND）
        4. Agent 执行（DAG 拓扑排序 → 同层 asyncio.gather 并行）
        5. 构建最终回复
        6. 记忆写入

        Args:
            task_input: 用户请求。

        Returns:
            TaskOutput: 端到端结果。
        """
        trace_id = task_input.trace_id or get_current_trace_id()
        user_ctx = task_input.user_context
        start_time = time.monotonic()

        with SpanContext("pipeline.run", trace_id) as root_span:
            try:
                prepared = await self._prepare_execution(
                    task_input,
                    trace_id,
                    root_span,
                )
                if prepared.blocked_response is not None:
                    return TaskOutput(
                        trace_id=trace_id,
                        user_id=user_ctx.user_id,
                        response=prepared.blocked_response,
                        agent_results=[],
                        memory_saved=[],
                        total_duration_ms=self._elapsed_ms(start_time),
                        status=prepared.blocked_status or "blocked",
                    )
                if prepared.routing is None or prepared.binding is None:
                    return TaskOutput(
                        trace_id=trace_id,
                        user_id=user_ctx.user_id,
                        response="执行准备失败：缺少路由结果",
                        agent_results=[],
                        memory_saved=[],
                        total_duration_ms=self._elapsed_ms(start_time),
                        status="error",
                    )

                routing = prepared.routing
                session_id = routing.session_key or prepared.binding.session_key

                # ── Step 1.5: 三级缓存查询（启用时） ─────
                if self._cache is not None:
                    try:
                        hit = await self._cache.get(
                            user_ctx.user_id, task_input.raw_message,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("cache_get_failed", error=str(exc))
                        hit = None
                    if hit is not None:
                        root_span.add_event("cache_hit", {"level": hit.level})
                        return TaskOutput(
                            trace_id=trace_id,
                            user_id=user_ctx.user_id,
                            response=hit.response,
                            agent_results=[],
                            memory_saved=[],
                            total_duration_ms=self._elapsed_ms(start_time),
                            status="cache_hit",
                        )

                # ── Step 2: ignore / 低置信度降级 ────────
                if routing.mode == ExecutionMode.IGNORE or routing.low_confidence:
                    return TaskOutput(
                        trace_id=trace_id,
                        user_id=user_ctx.user_id,
                        response=self._build_fallback_response(routing),
                        agent_results=[],
                        memory_saved=[],
                        total_duration_ms=self._elapsed_ms(start_time),
                        status="fallback",
                    )

                # ── Step 3: 权限校验 ─────────────────────
                if routing.requires_admin and user_ctx.role != UserRole.ADMIN:
                    return TaskOutput(
                        trace_id=trace_id,
                        user_id=user_ctx.user_id,
                        response="⚠️ 权限不足：该操作仅限管理员使用。",
                        agent_results=[],
                        memory_saved=[],
                        total_duration_ms=self._elapsed_ms(start_time),
                        status="permission_denied",
                    )

                # ── Step 3.5: 兜底 — 无计划但 mode 可执行时，自动生成默认子任务
                if not routing.plan and routing.mode != ExecutionMode.IGNORE:
                    routing = self._ensure_default_subtask(
                        routing, task_input.raw_message,
                    )

                # ── Step 3.6: 子任务后处理 — 修正 required_agents 以触发 ReAct
                routing = self._postprocess_subtasks(routing)
                validation = self._plan_validator.validate(
                    routing.plan,
                    tool_profile=routing.tool_profile,
                )
                if not validation.valid:
                    return TaskOutput(
                        trace_id=trace_id,
                        user_id=user_ctx.user_id,
                        response=f"计划校验失败：{'; '.join(validation.errors)}",
                        agent_results=[],
                        memory_saved=[],
                        total_duration_ms=self._elapsed_ms(start_time),
                        status="invalid_plan",
                    )

                # ── Step 4: Agent 执行（DAG 并行） ───────
                agent_results = await self._execute_dag(
                    routing.plan, task_input, trace_id, routing,
                )

                # ── Step 5: 构建回复 ─────────────────────
                response = self._build_response(routing, agent_results)
                artifacts = self._collect_artifacts(agent_results)

                # ── Step 5.5: 缓存回写（启用时） ─────────
                if self._cache is not None and response:
                    try:
                        await self._cache.set(
                            user_ctx.user_id, task_input.raw_message, response,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("cache_set_failed", error=str(exc))

                # ── Step 6: 记忆写入 ─────────────────────
                memory_saved = self._save_memory(
                    task_input, routing, agent_results, session_id,
                )

                total_ms = self._elapsed_ms(start_time)
                logger.info(
                    "pipeline_complete",
                    trace_id=trace_id,
                    mode=routing.mode.value,
                    duration_ms=total_ms,
                    status="success",
                )
                return TaskOutput(
                    trace_id=trace_id,
                    user_id=user_ctx.user_id,
                    response=response,
                    agent_results=agent_results,
                    memory_saved=memory_saved,
                    artifacts=artifacts,
                    total_duration_ms=total_ms,
                    status="success",
                )

            except Exception as exc:
                total_ms = self._elapsed_ms(start_time)
                logger.error("pipeline_error", trace_id=trace_id, error=str(exc))
                return TaskOutput(
                    trace_id=trace_id,
                    user_id=user_ctx.user_id,
                    response=f"系统异常：{exc}",
                    agent_results=[],
                    memory_saved=[],
                    total_duration_ms=total_ms,
                    status="error",
                )

    # ── DAG 并行执行 ─────────────────────────────────

    async def _execute_dag(
        self,
        sub_tasks: list[SubTask],
        task_input: TaskInput,
        trace_id: str,
        routing: RoutingDecision,
    ) -> list[AgentResult]:
        """按 DAG 拓扑排序分层执行子任务，同层 ``asyncio.gather`` 并行。"""
        layers = _topological_layers(sub_tasks)
        all_results: list[AgentResult] = []

        for layer in layers:
            coros = [
                self._execute_subtask(st, task_input, trace_id, routing) for st in layer
            ]
            layer_results = await asyncio.gather(*coros, return_exceptions=False)
            all_results.extend(layer_results)

        return all_results

    async def _execute_subtask(
        self,
        subtask: SubTask,
        task_input: TaskInput,
        trace_id: str,
        routing: RoutingDecision,
    ) -> AgentResult:
        """将单个子任务分发到对应 Agent。"""
        user_ctx = task_input.user_context
        session_id = routing.session_key or user_ctx.user_id

        # 选择 Agent（优先 required_agents 第一个）
        agent_type = subtask.required_agents[0] if subtask.required_agents else "llm_agent"

        if agent_type not in self._agents:
            return AgentResult(
                agent_name=agent_type,
                success=False,
                error=f"未知 Agent 类型: {agent_type}",
                duration_ms=0,
                trace_id=trace_id,
            )

        agent = self._agents[agent_type]

        self._configure_agent_context(agent, user_ctx, routing)

        with SpanContext(f"agent.{subtask.subtask_id}", trace_id):
            try:
                return await agent.run(subtask, session_id, user_ctx.user_id)
            except Exception as exc:
                return AgentResult(
                    agent_name=agent_type,
                    success=False,
                    error=str(exc),
                    duration_ms=0,
                    trace_id=trace_id,
                )

    # ── 回复构建 ─────────────────────────────────────

    @staticmethod
    def _build_response(routing: RoutingDecision, results: list[AgentResult]) -> str:
        """从 Agent 结果构建最终回复。"""
        if not results:
            return "任务已接收，未触发具体执行。"

        parts: list[str] = []
        for r in results:
            if r.success and r.output:
                parts.append(str(r.output))
            elif r.error:
                parts.append(f"[{r.agent_name}] 执行失败：{r.error}")

        return "\n\n".join(parts) if parts else "任务执行完毕，但无有效输出。"

    @staticmethod
    def _collect_artifacts(results: list[AgentResult]) -> list[Artifact]:
        """Extract lightweight Artifact descriptions from agent outputs."""
        artifacts: list[Artifact] = []
        for result in results:
            output = result.output
            payloads: list[object] = []
            if isinstance(output, dict):
                if "artifact" in output:
                    payloads.append(output["artifact"])
                if "artifacts" in output and isinstance(output["artifacts"], list):
                    payloads.extend(output["artifacts"])
                if "artifact_payload" in output:
                    payloads.append(output["artifact_payload"])
            for payload in payloads:
                if isinstance(payload, Artifact):
                    artifacts.append(payload)
                elif isinstance(payload, dict):
                    try:
                        artifacts.append(Artifact.model_validate(payload))
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("artifact_payload_ignored", error=str(exc))
        return artifacts

    @staticmethod
    def _build_fallback_response(routing: RoutingDecision) -> str:
        """低置信度 / ignore 模式时的兜底回复。"""
        if routing.low_confidence:
            return (
                "我不太确定您的意图（置信度较低）。"
                "您是想让我：① 生成内容？② 检索知识？③ 执行某个操作？"
                "请说得更具体一些~"
            )
        return "收到消息，暂无需处理的指令。"

    # ── 记忆写入 ─────────────────────────────────────

    def _save_memory(
        self,
        task_input: TaskInput,
        routing: RoutingDecision,
        results: list[AgentResult],
        session_id: str,
    ) -> list[MemoryEntry]:
        """将本次对话写入记忆（会话态 + 持久态）。"""
        user_id = task_input.user_context.user_id
        saved: list[MemoryEntry] = []

        # 写入用户请求（会话态）
        req_entry = self._memory.add(
            session_id=session_id,
            user_id=user_id,
            content=f"用户: {task_input.raw_message}",
            memory_type="short_term",
            tags=["user_request", routing.mode.value],
        )
        saved.append(req_entry)

        # 写入执行摘要（持久态）
        summary_parts = [f"模式: {routing.mode.value}（置信度{routing.confidence}）"]
        for r in results:
            status_mark = "✓" if r.success else "✗"
            summary_parts.append(f"[{r.agent_name}] {status_mark} {r.output or r.error}")

        summary_entry = self._memory.add(
            session_id=session_id,
            user_id=user_id,
            content="\n".join(summary_parts),
            memory_type="long_term",
            tags=["execution_summary", routing.mode.value],
        )
        saved.append(summary_entry)

        return saved

    # ── 工具方法 ─────────────────────────────────────

    # ── 执行模式 → 默认 Agent 映射 ──────────────────

    _MODE_AGENT_MAP: dict[ExecutionMode, list[str]] = {
        ExecutionMode.PLAN: ["llm_agent"],
        ExecutionMode.ACT: ["llm_agent", "tool_agent"],
        ExecutionMode.QA: ["retrieval_agent"],
        ExecutionMode.DELEGATE: ["tool_agent"],
        ExecutionMode.REPAIR: ["reflection_agent"],
    }

    def _postprocess_subtasks(self, routing: RoutingDecision) -> RoutingDecision:
        """修正子任务 required_agents，确保 act/tool 走 ReAct 路径。

        Router LLM 可能生成 required_agents=["tool_agent"]，导致任务直接交给
        ToolAgent 而非 LLMAgent 的 ReAct 循环。此方法将其修正为
        ["llm_agent", "tool_agent"]，确保 LLMAgent 通过 ReAct 调用工具。
        """
        needs_react = (
            routing.mode == ExecutionMode.ACT
            or "tool" in routing.capabilities
        )
        if not needs_react or not routing.plan:
            return routing

        patched = False
        new_tasks = []
        for st in routing.plan:
            agents = st.required_agents
            if agents and agents[0] == "tool_agent" and "llm_agent" not in agents:
                agents = ["llm_agent", *agents]
                patched = True
            new_tasks.append(st.model_copy(update={"required_agents": agents}))

        if patched:
            logger.info(
                "pipeline.postprocess_subtasks",
                mode=routing.mode.value,
                note="required_agents patched to trigger ReAct",
            )
            return routing.model_copy(update={"plan": new_tasks})
        return routing

    def _ensure_default_subtask(
        self,
        routing: RoutingDecision,
        raw_message: str,
    ) -> RoutingDecision:
        """当 plan 为空且 mode 在映射表中时，自动生成默认子任务。"""
        agents = self._MODE_AGENT_MAP.get(routing.mode)
        if not agents:
            return routing

        logger.info(
            "pipeline.fallback_subtask",
            mode=routing.mode.value,
            agents=agents,
        )
        default_subtask = SubTask(
            subtask_id="subtask_fallback_1",
            description=raw_message,
            required_agents=list(agents),
            priority=1,
            depends_on=[],
        )
        return routing.model_copy(update={"plan": [default_subtask]})

    @staticmethod
    def _promote_addressed_ignore(
        task_input: TaskInput,
        routing: RoutingDecision,
    ) -> RoutingDecision:
        """飞书中明确发给机器人的消息，即使被 Router 判 ignore，也给出普通回复。"""
        source_ctx = task_input.source_context
        is_feishu_addressed = (
            source_ctx.channel == "feishu"
            and (
                source_ctx.chat_type is SourceChatType.DIRECT
                or source_ctx.is_at_bot
            )
        )
        if (
            not is_feishu_addressed
            or routing.mode != ExecutionMode.IGNORE
            or routing.low_confidence
        ):
            return routing

        logger.info(
            "pipeline.addressed_ignore_promoted",
            channel=source_ctx.channel,
            chat_type=source_ctx.chat_type.value,
            is_at_bot=source_ctx.is_at_bot,
        )
        default_subtask = SubTask(
            subtask_id="subtask_addressed_reply_1",
            description=task_input.raw_message,
            required_agents=["llm_agent"],
            priority=1,
            depends_on=[],
        )
        return routing.model_copy(
            update={
                "mode": ExecutionMode.QA,
                "reasoning": f"{routing.reasoning}; addressed feishu message promoted",
                "plan": [default_subtask],
            }
        )

    @staticmethod
    def _elapsed_ms(start: float) -> int:
        return int((time.monotonic() - start) * 1000)
