"""Mem0 式冲突检测器。

新记忆写入前，与现有记忆做语义相似度比较，
决定操作类型：ADD / UPDATE / DELETE / NOOP。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from agent_hub.core.models import MemoryEntry

if TYPE_CHECKING:
    from agent_hub.memory.vector_memory import VectorMemory
    from agent_hub.rag.embedder import Embedder

logger = structlog.get_logger(__name__)


@dataclass
class MemoryOp:
    """记忆操作描述。"""

    action: str  # ADD / UPDATE / DELETE / NOOP
    target_id: str | None = None  # UPDATE/DELETE 时指向目标 memory_id
    reason: str = ""


class ConflictDetector:
    """Mem0 式记忆冲突检测器。

    写入前先做 cosine_similarity(new_embedding, existing_embeddings)，
    阈值 > threshold 触发 UPDATE 判断。

    Args:
        vector_memory: 向量记忆存储。
        embedder: 文本向量化服务。
        threshold: 冲突检测相似度阈值，默认 0.85。
        llm_confirm: 是否使用 LLM 二次确认（可选，默认 False）。
    """

    def __init__(
        self,
        vector_memory: VectorMemory,
        embedder: Embedder,
        threshold: float = 0.85,
        llm_confirm: bool = False,
    ) -> None:
        self._vector_memory = vector_memory
        self._embedder = embedder
        self._threshold = threshold
        self._llm_confirm = llm_confirm

    async def detect(self, new_entry: MemoryEntry, user_id: str) -> MemoryOp:
        """检测新记忆与现有记忆是否冲突。

        流程：
        1. 计算新记忆 embedding
        2. 在向量记忆中搜索相似条目
        3. 相似度 > threshold → UPDATE 候选
        4. 无相似 → ADD

        Args:
            new_entry: 待写入的新记忆。
            user_id: 用户 ID。

        Returns:
            MemoryOp 描述应执行的操作。
        """
        try:
            # 搜索相似记忆
            similar = await self._vector_memory.search_similar(
                query=new_entry.content,
                user_id=user_id,
                top_k=3,
                threshold=self._threshold,
            )

            if not similar:
                return MemoryOp(action="ADD", reason="无相似记忆，新增")

            best_match, best_score = similar[0]

            # 内容完全相同 → NOOP
            if best_match.content.strip() == new_entry.content.strip():
                return MemoryOp(
                    action="NOOP",
                    target_id=best_match.memory_id,
                    reason=f"内容完全相同 (score={best_score:.4f})",
                )

            # 高相似度 → UPDATE
            logger.info(
                "conflict_detected",
                new_preview=new_entry.content[:50],
                existing_id=best_match.memory_id,
                score=best_score,
            )

            return MemoryOp(
                action="UPDATE",
                target_id=best_match.memory_id,
                reason=f"语义相似 (score={best_score:.4f})，合并更新",
            )

        except Exception as exc:
            # 冲突检测失败不阻塞写入
            logger.warning("conflict_detection_failed", error=str(exc))
            return MemoryOp(action="ADD", reason=f"检测异常，降级为新增: {exc}")
