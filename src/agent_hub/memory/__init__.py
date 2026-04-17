"""记忆管理模块：会话记忆 / 长期记忆 / 向量记忆。

提供三层记忆体系和统一管理入口：

- :class:`SessionMemory` — 内存级滑动窗口，维持最近 N 轮对话
- :class:`PersistentMemory` — Obsidian Markdown 持久化，长期知识沉淀
- :class:`VectorMemory` — pgvector 向量记忆，语义检索
- :class:`MemoryManager` — 统一管理入口，Mem0 式冲突检测 + 自动分发
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import structlog

from agent_hub.core.models import MemoryEntry
from agent_hub.memory.persistent import PersistentMemory
from agent_hub.memory.session import SessionMemory

if TYPE_CHECKING:
    from agent_hub.memory.conflict_detector import ConflictDetector
    from agent_hub.memory.memory_ops import MemoryOperator
    from agent_hub.memory.vector_memory import VectorMemory

logger = structlog.get_logger(__name__)

__all__ = [
    "MemoryManager",
    "PersistentMemory",
    "SessionMemory",
]


class MemoryManager:
    """统一管理多层记忆的入口。

    封装 :class:`SessionMemory`、:class:`PersistentMemory` 和可选的
    :class:`VectorMemory`，提供 ``add`` / ``add_with_conflict_check`` /
    ``get_context`` / ``get_recent`` 等高层 API。

    当配置了向量记忆时，``add_with_conflict_check()`` 会经过
    ConflictDetector 进行 Mem0 式冲突检测。

    Args:
        vault_path: Obsidian Vault 目录路径。
        max_window_size: 滑动窗口大小，默认 20。

    Example::

        mm = MemoryManager(vault_path="./data/obsidian")
        mm.add(
            session_id="sess-001",
            user_id="u001",
            content="帮我写一个二分查找",
            memory_type="short_term",
            tags=["algorithm"],
        )
        ctx = mm.get_context(session_id="sess-001", user_id="u001")
    """

    def __init__(
        self,
        vault_path: str = "./data/obsidian",
        max_window_size: int = 20,
    ) -> None:
        self.session = SessionMemory(max_window_size)
        self.persistent = PersistentMemory(vault_path)

        # 向量记忆层（可选，需显式注入）
        self._vector_memory: VectorMemory | None = None
        self._conflict_detector: ConflictDetector | None = None
        self._memory_operator: MemoryOperator | None = None

    def inject_vector_memory(
        self,
        vector_memory: VectorMemory,
        conflict_detector: ConflictDetector,
        memory_operator: MemoryOperator,
    ) -> None:
        """注入向量记忆层及冲突检测组件。

        由 Pipeline 在初始化时调用。
        """
        self._vector_memory = vector_memory
        self._conflict_detector = conflict_detector
        self._memory_operator = memory_operator
        logger.info("memory_vector_layer_injected")

    def add(
        self,
        session_id: str,
        user_id: str,
        content: str,
        memory_type: str = "short_term",
        tags: list[str] | None = None,
    ) -> MemoryEntry:
        """添加记忆（同步版本，不做冲突检测）。

        会话态始终写入；当 ``memory_type == "long_term"`` 时同时
        写入 Obsidian 持久化存储。

        Args:
            session_id: 会话唯一标识。
            user_id: 用户唯一标识。
            content: 记忆内容文本。
            memory_type: ``"short_term"`` 或 ``"long_term"``，默认前者。
            tags: 标签列表。

        Returns:
            创建的 MemoryEntry 实例。
        """
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            session_id=session_id,
            tags=tags or [],
            created_at=datetime.now(),
        )

        # 始终写入会话态
        evicted = self.session.add(session_id, entry)
        _ = evicted

        # 长期记忆同时写入 Obsidian
        if memory_type == "long_term":
            self.persistent.save(entry, user_id)

        return entry

    async def add_with_conflict_check(
        self,
        session_id: str,
        user_id: str,
        content: str,
        memory_type: str = "short_term",
        tags: list[str] | None = None,
    ) -> MemoryEntry:
        """添加记忆（异步版本，带 Mem0 式冲突检测）。

        流程：
        1. 会话态始终写入
        2. 长期记忆经过 ConflictDetector 决定 ADD/UPDATE/DELETE/NOOP
        3. 通过 MemoryOperator 执行决定的操作

        Args:
            session_id: 会话唯一标识。
            user_id: 用户唯一标识。
            content: 记忆内容文本。
            memory_type: ``"short_term"`` 或 ``"long_term"``。
            tags: 标签列表。

        Returns:
            创建或更新后的 MemoryEntry 实例。
        """
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            session_id=session_id,
            tags=tags or [],
            created_at=datetime.now(),
        )

        # 始终写入会话态
        self.session.add(session_id, entry)

        # 长期记忆走冲突检测 → 操作执行
        if memory_type == "long_term" and self._conflict_detector and self._memory_operator:
            op = await self._conflict_detector.detect(entry, user_id)
            logger.info(
                "memory_conflict_op",
                action=op.action,
                target_id=op.target_id,
                reason=op.reason,
            )
            entry = await self._memory_operator.execute(op, entry, user_id)
        elif memory_type == "long_term":
            # 无向量层，降级到纯 Obsidian 写入
            self.persistent.save(entry, user_id)

        return entry

    def get_context(
        self,
        session_id: str,
        user_id: str,
        session_tokens: int = 4000,
        persistent_tokens: int = 2000,
    ) -> str:
        """获取完整上下文字符串，供 LLM 使用。

        合并持久态和会话态上下文。

        Args:
            session_id: 会话唯一标识。
            user_id: 用户唯一标识。
            session_tokens: 会话态允许的最大 token 数。
            persistent_tokens: 持久态允许的最大 token 数。

        Returns:
            合并后的上下文字符串。
        """
        session_ctx = self.session.get_context_for_llm(session_id, session_tokens)
        persistent_ctx = self.persistent.get_context_for_llm(user_id, persistent_tokens)

        if persistent_ctx:
            return f"【持久记忆】\n{persistent_ctx}\n\n【当前会话】\n{session_ctx}"
        return f"【当前会话】\n{session_ctx}"

    def get_recent(self, session_id: str, n: int = 10) -> list[MemoryEntry]:
        """获取最近 N 条会话记忆。

        Args:
            session_id: 会话唯一标识。
            n: 返回条数，默认 10。

        Returns:
            按时间正序排列的记忆列表。
        """
        return self.session.get_recent(session_id, n)

    async def cleanup_expired(self, user_id: str) -> int:
        """清理过期记忆（TTL 机制）。

        扫描持久记忆，将 ttl > 0 且已过期的标记为 is_deleted。

        Args:
            user_id: 用户 ID。

        Returns:
            清理的记忆数量。
        """
        entries = self.persistent.load_recent(user_id, days=365)
        now = datetime.now()
        cleaned = 0

        for entry in entries:
            if entry.ttl > 0:
                elapsed = (now - entry.created_at).total_seconds()
                if elapsed > entry.ttl and not entry.is_deleted:
                    self.persistent.soft_delete(entry.memory_id, user_id)
                    cleaned += 1

        if cleaned:
            logger.info("memory_ttl_cleanup", user_id=user_id, cleaned=cleaned)
        return cleaned
