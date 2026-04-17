"""会话态滑动窗口记忆。

基于 ``collections.deque`` 实现内存级滑动窗口，
自动淘汰超出窗口的老消息，并支持将其沉淀到持久态。
"""

from __future__ import annotations

from collections import deque

from agent_hub.core.models import MemoryEntry


class SessionMemory:
    """会话态记忆管理器。

    每个 ``session_id`` 维护一个独立的消息队列，队列长度由
    ``max_window_size`` 控制。窗口满时新消息入队会导致最旧的
    消息被淘汰；调用方可通过 :meth:`add` 的返回值获取被淘汰的条目，
    用于沉淀到持久态。

    Args:
        max_window_size: 滑动窗口大小，默认 20 条。
    """

    def __init__(self, max_window_size: int = 20) -> None:
        self._sessions: dict[str, deque[MemoryEntry]] = {}
        self._max_window = max_window_size

    def add(self, session_id: str, entry: MemoryEntry) -> MemoryEntry | None:
        """添加一条记忆，自动维护滑动窗口。

        当窗口已满时，最旧的条目会被淘汰并作为返回值返回，
        调用方可决定是否将其沉淀到持久态。

        Args:
            session_id: 会话唯一标识。
            entry: 要添加的记忆条目。

        Returns:
            被淘汰的记忆条目；窗口未满时返回 ``None``。
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = deque(maxlen=self._max_window)

        queue = self._sessions[session_id]
        evicted: MemoryEntry | None = None

        # 窗口已满，取出即将被淘汰的最旧条目
        if len(queue) == self._max_window:
            evicted = queue[0]

        queue.append(entry)
        return evicted

    def get_recent(self, session_id: str, n: int = 10) -> list[MemoryEntry]:
        """获取最近 *n* 条记忆。

        Args:
            session_id: 会话唯一标识。
            n: 返回条数，默认 10。

        Returns:
            按时间正序排列的记忆列表（最旧在前）。
        """
        queue = self._sessions.get(session_id)
        if not queue:
            return []
        items = list(queue)
        return items[-n:]

    def get_context_for_llm(
        self, session_id: str, max_tokens: int = 4000
    ) -> str:
        """将最近的记忆拼接为字符串，供 LLM 上下文使用。

        使用 ``len(text) // 4`` 近似估算 token 数量，
        从最新条目开始向前追溯，直到累计 token 达到上限。

        Args:
            session_id: 会话唯一标识。
            max_tokens: 允许的最大 token 数，默认 4000。

        Returns:
            拼接后的上下文字符串。
        """
        queue = self._sessions.get(session_id)
        if not queue:
            return ""

        lines: list[str] = []
        used_chars = 0
        max_chars = max_tokens * 4  # 1 token ≈ 4 chars

        # 从最新开始向前追溯
        for entry in reversed(queue):
            line = f"[{entry.memory_type}] {entry.content}"
            line_chars = len(line)
            if used_chars + line_chars > max_chars:
                break
            lines.append(line)
            used_chars += line_chars

        # 反转回时间正序
        lines.reverse()
        return "\n".join(lines)

    def should_persist(self, session_id: str) -> list[MemoryEntry]:
        """返回窗口中即将被淘汰的条目（窗口已满时的最旧条目）。

        此方法不会修改队列，仅用于预判。实际淘汰发生在
        :meth:`add` 被调用时。

        Args:
            session_id: 会话唯一标识。

        Returns:
            包含即将被淘汰条目的列表（最多 1 条）；窗口未满时返回空列表。
        """
        queue = self._sessions.get(session_id)
        if not queue or len(queue) < self._max_window:
            return []
        return [queue[0]]

    def clear(self, session_id: str) -> None:
        """清空指定会话的全部记忆。

        Args:
            session_id: 会话唯一标识。
        """
        self._sessions.pop(session_id, None)
