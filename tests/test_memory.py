"""记忆管理模块的单元测试。

覆盖：
- SessionMemory 滑动窗口、淘汰回收、上下文拼接
- PersistentMemory Obsidian 写入格式、标签/关键词/日期搜索
- MemoryManager 统一接口
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml

from agent_hub.core.models import MemoryEntry
from agent_hub.memory import MemoryManager
from agent_hub.memory.persistent import (
    PersistentMemory,
    _entry_to_markdown,
    _markdown_to_entry,
    _parse_frontmatter,
)
from agent_hub.memory.session import SessionMemory


# ── helpers ───────────────────────────────────────────


def _make_entry(
    content: str = "hello",
    memory_type: str = "short_term",
    session_id: str = "sess-1",
    tags: list[str] | None = None,
    created_at: datetime | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        content=content,
        memory_type=memory_type,
        session_id=session_id,
        tags=tags or [],
        created_at=created_at or datetime.now(),
    )


# ══════════════════════════════════════════════════════
#  SessionMemory 测试
# ══════════════════════════════════════════════════════


class TestSessionMemory:
    def test_add_and_get_recent(self) -> None:
        sm = SessionMemory(max_window_size=5)
        for i in range(3):
            sm.add("s1", _make_entry(content=f"msg-{i}"))
        recent = sm.get_recent("s1", n=10)
        assert len(recent) == 3
        assert recent[0].content == "msg-0"
        assert recent[-1].content == "msg-2"

    def test_sliding_window_eviction(self) -> None:
        sm = SessionMemory(max_window_size=3)
        evicted_list: list[MemoryEntry | None] = []
        for i in range(5):
            evicted = sm.add("s1", _make_entry(content=f"msg-{i}"))
            evicted_list.append(evicted)

        # 前 3 条不触发淘汰
        assert evicted_list[0] is None
        assert evicted_list[1] is None
        assert evicted_list[2] is None
        # 第 4 条淘汰 msg-0
        assert evicted_list[3] is not None
        assert evicted_list[3].content == "msg-0"
        # 第 5 条淘汰 msg-1
        assert evicted_list[4] is not None
        assert evicted_list[4].content == "msg-1"

        # 窗口内仅保留最新 3 条
        recent = sm.get_recent("s1")
        assert len(recent) == 3
        assert [e.content for e in recent] == ["msg-2", "msg-3", "msg-4"]

    def test_should_persist_empty(self) -> None:
        sm = SessionMemory(max_window_size=3)
        assert sm.should_persist("nonexistent") == []

    def test_should_persist_not_full(self) -> None:
        sm = SessionMemory(max_window_size=5)
        sm.add("s1", _make_entry(content="a"))
        assert sm.should_persist("s1") == []

    def test_should_persist_full(self) -> None:
        sm = SessionMemory(max_window_size=3)
        for i in range(3):
            sm.add("s1", _make_entry(content=f"msg-{i}"))
        to_persist = sm.should_persist("s1")
        assert len(to_persist) == 1
        assert to_persist[0].content == "msg-0"

    def test_get_context_for_llm_basic(self) -> None:
        sm = SessionMemory(max_window_size=10)
        sm.add("s1", _make_entry(content="hi", memory_type="short_term"))
        sm.add("s1", _make_entry(content="hello", memory_type="long_term"))
        ctx = sm.get_context_for_llm("s1")
        assert "[short_term] hi" in ctx
        assert "[long_term] hello" in ctx

    def test_get_context_for_llm_token_limit(self) -> None:
        sm = SessionMemory(max_window_size=100)
        # 每条约 30 chars → ~7.5 tokens
        for i in range(50):
            sm.add("s1", _make_entry(content=f"message number {i:03d} padding"))
        # 限制 50 tokens → ~200 chars → 大约能容纳 6 条
        ctx = sm.get_context_for_llm("s1", max_tokens=50)
        lines = [ln for ln in ctx.split("\n") if ln.strip()]
        assert len(lines) < 50  # 远少于 50 条

    def test_get_context_for_llm_empty(self) -> None:
        sm = SessionMemory()
        assert sm.get_context_for_llm("nonexistent") == ""

    def test_get_recent_empty(self) -> None:
        sm = SessionMemory()
        assert sm.get_recent("nonexistent") == []

    def test_clear(self) -> None:
        sm = SessionMemory()
        sm.add("s1", _make_entry(content="x"))
        sm.clear("s1")
        assert sm.get_recent("s1") == []

    def test_clear_nonexistent(self) -> None:
        sm = SessionMemory()
        sm.clear("nonexistent")  # 不应抛异常

    def test_multiple_sessions_isolated(self) -> None:
        sm = SessionMemory(max_window_size=5)
        sm.add("s1", _make_entry(content="s1-msg"))
        sm.add("s2", _make_entry(content="s2-msg"))
        assert len(sm.get_recent("s1")) == 1
        assert len(sm.get_recent("s2")) == 1
        assert sm.get_recent("s1")[0].content == "s1-msg"


# ══════════════════════════════════════════════════════
#  PersistentMemory 测试
# ══════════════════════════════════════════════════════


class TestPersistentMemory:
    def test_save_creates_file(self, tmp_path: Path) -> None:
        pm = PersistentMemory(vault_path=str(tmp_path))
        entry = _make_entry(
            content="二分查找实现",
            memory_type="long_term",
            session_id="sess-abc123",
            tags=["python", "algorithm"],
        )
        path = pm.save(entry, user_id="u001")
        assert path.exists()
        assert path.suffix == ".md"
        # 路径包含 user_id 和月份目录
        assert "u001" in str(path)

    def test_save_file_format(self, tmp_path: Path) -> None:
        pm = PersistentMemory(vault_path=str(tmp_path))
        entry = _make_entry(
            content="测试内容",
            memory_type="long_term",
            session_id="sess-xyz",
            tags=["test", "demo"],
        )
        path = pm.save(entry, user_id="u001")
        text = path.read_text(encoding="utf-8")

        # 检查 YAML frontmatter 结构
        assert text.startswith("---\n")
        fm = _parse_frontmatter(text)
        assert fm["memory_type"] == "long_term"
        assert fm["session_id"] == "sess-xyz"
        assert "test" in fm["tags"]
        assert "demo" in fm["tags"]
        # created_at 是 ISO 格式字符串
        datetime.fromisoformat(str(fm["created_at"]))

        # 正文
        assert "测试内容" in text

    def test_save_batch(self, tmp_path: Path) -> None:
        pm = PersistentMemory(vault_path=str(tmp_path))
        entries = [_make_entry(content=f"batch-{i}") for i in range(3)]
        paths = pm.save_batch(entries, user_id="u001")
        assert len(paths) == 3
        assert all(p.exists() for p in paths)

    def test_save_filename_collision(self, tmp_path: Path) -> None:
        """同一秒内多次写入不应覆盖。"""
        pm = PersistentMemory(vault_path=str(tmp_path))
        now = datetime.now()
        e1 = _make_entry(content="first", session_id="sess-a", created_at=now)
        e2 = _make_entry(content="second", session_id="sess-a", created_at=now)
        p1 = pm.save(e1, user_id="u001")
        p2 = pm.save(e2, user_id="u001")
        assert p1 != p2
        assert p1.exists() and p2.exists()

    def test_search_by_tag(self, tmp_path: Path) -> None:
        pm = PersistentMemory(vault_path=str(tmp_path))
        pm.save(_make_entry(content="有python标签", tags=["python"]), "u001")
        pm.save(_make_entry(content="无python标签", tags=["java"]), "u001")
        pm.save(_make_entry(content="也有python", tags=["python", "algo"]), "u001")

        results = pm.search_by_tag("u001", "python")
        assert len(results) == 2
        contents = [r.content for r in results]
        assert "有python标签" in contents
        assert "也有python" in contents

    def test_search_by_tag_no_match(self, tmp_path: Path) -> None:
        pm = PersistentMemory(vault_path=str(tmp_path))
        pm.save(_make_entry(tags=["python"]), "u001")
        assert pm.search_by_tag("u001", "rust") == []

    def test_search_by_tag_nonexistent_user(self, tmp_path: Path) -> None:
        pm = PersistentMemory(vault_path=str(tmp_path))
        assert pm.search_by_tag("ghost", "python") == []

    def test_search_by_keyword(self, tmp_path: Path) -> None:
        pm = PersistentMemory(vault_path=str(tmp_path))
        pm.save(_make_entry(content="二分查找算法"), "u001")
        pm.save(_make_entry(content="快速排序实现"), "u001")
        pm.save(_make_entry(content="另一个二分法"), "u001")

        results = pm.search_by_keyword("u001", "二分")
        assert len(results) == 2

    def test_search_by_keyword_limit(self, tmp_path: Path) -> None:
        pm = PersistentMemory(vault_path=str(tmp_path))
        for i in range(5):
            pm.save(_make_entry(content=f"match-{i}"), "u001")
        results = pm.search_by_keyword("u001", "match", limit=3)
        assert len(results) == 3

    def test_load_recent(self, tmp_path: Path) -> None:
        pm = PersistentMemory(vault_path=str(tmp_path))
        now = datetime.now()
        old = now - timedelta(days=30)

        pm.save(_make_entry(content="recent", created_at=now), "u001")
        pm.save(_make_entry(content="old", created_at=old), "u001")

        results = pm.load_recent("u001", days=7)
        assert len(results) == 1
        assert results[0].content == "recent"

    def test_load_recent_nonexistent_user(self, tmp_path: Path) -> None:
        pm = PersistentMemory(vault_path=str(tmp_path))
        assert pm.load_recent("ghost") == []

    def test_get_context_for_llm(self, tmp_path: Path) -> None:
        pm = PersistentMemory(vault_path=str(tmp_path))
        pm.save(
            _make_entry(content="记忆内容A", tags=["tag1"]),
            "u001",
        )
        pm.save(
            _make_entry(content="记忆内容B", tags=["tag2"]),
            "u001",
        )
        ctx = pm.get_context_for_llm("u001")
        assert "记忆内容A" in ctx
        assert "记忆内容B" in ctx

    def test_get_context_for_llm_empty(self, tmp_path: Path) -> None:
        pm = PersistentMemory(vault_path=str(tmp_path))
        assert pm.get_context_for_llm("ghost") == ""


# ══════════════════════════════════════════════════════
#  序列化 / 反序列化辅助函数测试
# ══════════════════════════════════════════════════════


class TestSerialization:
    def test_entry_to_markdown_roundtrip(self) -> None:
        entry = _make_entry(
            content="roundtrip test",
            memory_type="long_term",
            session_id="sess-rt",
            tags=["a", "b"],
        )
        md = _entry_to_markdown(entry)
        restored = _markdown_to_entry(md)
        assert restored is not None
        assert restored.content == "roundtrip test"
        assert restored.memory_type == "long_term"
        assert restored.session_id == "sess-rt"
        assert restored.tags == ["a", "b"]

    def test_parse_frontmatter_valid(self) -> None:
        text = "---\ntags: [a, b]\nkey: value\n---\nbody"
        fm = _parse_frontmatter(text)
        assert fm["tags"] == ["a", "b"]
        assert fm["key"] == "value"

    def test_parse_frontmatter_invalid(self) -> None:
        assert _parse_frontmatter("no frontmatter here") == {}

    def test_markdown_to_entry_invalid(self) -> None:
        assert _markdown_to_entry("no frontmatter") is None


# ══════════════════════════════════════════════════════
#  MemoryManager 测试
# ══════════════════════════════════════════════════════


class TestMemoryManager:
    def test_add_short_term(self, tmp_path: Path) -> None:
        mm = MemoryManager(vault_path=str(tmp_path))
        entry = mm.add(
            session_id="sess-1",
            user_id="u001",
            content="短期记忆",
            memory_type="short_term",
        )
        assert entry.content == "短期记忆"
        assert entry.memory_type == "short_term"
        # 会话态有数据
        assert len(mm.get_recent("sess-1")) == 1
        # 持久态无文件
        assert not list(tmp_path.rglob("*.md"))

    def test_add_long_term(self, tmp_path: Path) -> None:
        mm = MemoryManager(vault_path=str(tmp_path))
        mm.add(
            session_id="sess-1",
            user_id="u001",
            content="长期记忆",
            memory_type="long_term",
            tags=["important"],
        )
        # 会话态有数据
        assert len(mm.get_recent("sess-1")) == 1
        # 持久态有文件
        md_files = list(tmp_path.rglob("*.md"))
        assert len(md_files) == 1
        text = md_files[0].read_text(encoding="utf-8")
        assert "长期记忆" in text
        fm = _parse_frontmatter(text)
        assert "important" in fm["tags"]

    def test_get_context_combined(self, tmp_path: Path) -> None:
        mm = MemoryManager(vault_path=str(tmp_path))
        mm.add("sess-1", "u001", "短期消息", memory_type="short_term")
        mm.add("sess-1", "u001", "长期知识", memory_type="long_term")

        ctx = mm.get_context("sess-1", "u001")
        assert "【持久记忆】" in ctx
        assert "【当前会话】" in ctx
        assert "长期知识" in ctx
        assert "短期消息" in ctx

    def test_get_context_session_only(self, tmp_path: Path) -> None:
        mm = MemoryManager(vault_path=str(tmp_path))
        mm.add("sess-1", "u001", "仅短期", memory_type="short_term")
        ctx = mm.get_context("sess-1", "u001")
        assert "【当前会话】" in ctx
        assert "仅短期" in ctx
        # 无持久记忆时不显示该标题
        assert "【持久记忆】" not in ctx

    def test_get_recent(self, tmp_path: Path) -> None:
        mm = MemoryManager(vault_path=str(tmp_path))
        for i in range(5):
            mm.add("sess-1", "u001", f"msg-{i}")
        recent = mm.get_recent("sess-1", n=3)
        assert len(recent) == 3
        assert recent[-1].content == "msg-4"

    def test_custom_window_size(self, tmp_path: Path) -> None:
        mm = MemoryManager(vault_path=str(tmp_path), max_window_size=3)
        for i in range(5):
            mm.add("sess-1", "u001", f"msg-{i}")
        recent = mm.get_recent("sess-1", n=10)
        assert len(recent) == 3
        assert [e.content for e in recent] == ["msg-2", "msg-3", "msg-4"]

    def test_usage_example(self, tmp_path: Path) -> None:
        """文档示例代码的可运行验证。"""
        mm = MemoryManager(vault_path=str(tmp_path))

        mm.add(
            session_id="sess-001",
            user_id="u001",
            content="帮我写一个二分查找",
            memory_type="short_term",
            tags=["algorithm"],
        )
        mm.add(
            session_id="sess-001",
            user_id="u001",
            content="LLM 输出了完整的 Python 二分查找代码",
            memory_type="long_term",
            tags=["python", "algorithm"],
        )

        ctx = mm.get_context(session_id="sess-001", user_id="u001")
        assert "二分查找" in ctx
        assert "【当前会话】" in ctx
