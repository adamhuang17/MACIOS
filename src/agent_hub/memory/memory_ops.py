"""记忆操作封装：ADD / UPDATE / DELETE / NOOP。

每次写入前经过 ConflictDetector 决定操作类型，
所有 DELETE/UPDATE 操作记录到审计日志。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from agent_hub.core.models import MemoryEntry
from agent_hub.memory.conflict_detector import MemoryOp

if TYPE_CHECKING:
    from agent_hub.memory.persistent import PersistentMemory
    from agent_hub.memory.vector_memory import VectorMemory

logger = structlog.get_logger(__name__)

# 审计日志独立 handler
_audit_logger = logging.getLogger("memory_audit")


def _init_audit_logger(log_dir: str = "./data") -> None:
    """初始化审计日志（独立文件 handler）。"""
    log_path = Path(log_dir) / "memory_audit.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    target = str(log_path)

    # 已经指向同一文件则跳过
    for h in _audit_logger.handlers:
        if isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path.resolve()):
            return

    # 如果目录不同，替换旧 handler
    _audit_logger.handlers.clear()

    handler = logging.FileHandler(target, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
    ))
    _audit_logger.addHandler(handler)
    _audit_logger.setLevel(logging.INFO)


class MemoryOperator:
    """记忆操作执行器。

    根据 MemoryOp 的 action 分发到对应操作。

    Args:
        persistent: Obsidian 持久化记忆实例。
        vector_memory: 向量记忆存储实例（可选）。
        audit_log_dir: 审计日志目录。
    """

    def __init__(
        self,
        persistent: PersistentMemory,
        vector_memory: VectorMemory | None = None,
        audit_log_dir: str = "./data",
    ) -> None:
        self._persistent = persistent
        self._vector_memory = vector_memory
        _init_audit_logger(audit_log_dir)

    async def execute(
        self,
        op: MemoryOp,
        entry: MemoryEntry,
        user_id: str,
    ) -> MemoryEntry:
        """执行记忆操作。

        Args:
            op: 冲突检测返回的操作描述。
            entry: 要写入的记忆条目。
            user_id: 用户 ID。

        Returns:
            最终保存的 MemoryEntry。
        """
        if op.action == "ADD":
            return await self._do_add(entry, user_id)
        elif op.action == "UPDATE":
            return await self._do_update(op, entry, user_id)
        elif op.action == "DELETE":
            return await self._do_delete(op, user_id)
        else:  # NOOP
            logger.info("memory_noop", memory_id=entry.memory_id, reason=op.reason)
            return entry

    async def _do_add(self, entry: MemoryEntry, user_id: str) -> MemoryEntry:
        """新增记忆。"""
        # 持久化到 Obsidian
        if entry.memory_type == "long_term":
            self._persistent.save(entry, user_id)

        # 向量化存储
        if self._vector_memory is not None:
            await self._vector_memory.store(entry, user_id)

        logger.info("memory_add", memory_id=entry.memory_id, user_id=user_id)
        return entry

    async def _do_update(
        self, op: MemoryOp, entry: MemoryEntry, user_id: str,
    ) -> MemoryEntry:
        """更新记忆（软删旧 + 新增新）。"""
        # 审计日志
        _audit_logger.info(
            "UPDATE | user=%s | old_id=%s | new_id=%s | reason=%s",
            user_id, op.target_id, entry.memory_id, op.reason,
        )

        # 软删除旧记忆
        if op.target_id:
            self._persistent.soft_delete(op.target_id, user_id)
            if self._vector_memory is not None:
                await self._vector_memory.delete(op.target_id, user_id)

        # 写入新记忆
        return await self._do_add(entry, user_id)

    async def _do_delete(self, op: MemoryOp, user_id: str) -> MemoryEntry:
        """删除记忆。"""
        _audit_logger.info(
            "DELETE | user=%s | target_id=%s | reason=%s",
            user_id, op.target_id, op.reason,
        )

        if op.target_id:
            self._persistent.soft_delete(op.target_id, user_id)
            if self._vector_memory is not None:
                await self._vector_memory.delete(op.target_id, user_id)

        # 返回一个标记为已删除的占位 entry
        return MemoryEntry(
            memory_id=op.target_id or "",
            content="[已删除]",
            memory_type="long_term",
            is_deleted=True,
        )
