"""记忆管理 v2 测试。

覆盖 Mem0 式冲突检测、语义去重、TTL 过期、软删除、审计日志。
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_hub.core.models import MemoryEntry
from agent_hub.memory import MemoryManager
from agent_hub.memory.conflict_detector import ConflictDetector, MemoryOp
from agent_hub.memory.memory_ops import MemoryOperator
from agent_hub.memory.persistent import PersistentMemory


# ═══════════════════════════════════════════════════════
# MemoryEntry 新字段测试
# ═══════════════════════════════════════════════════════


class TestMemoryEntryNewFields:
    """MemoryEntry 新增字段测试。"""

    def test_default_memory_id(self) -> None:
        entry = MemoryEntry(content="test", memory_type="short_term")
        assert entry.memory_id
        assert len(entry.memory_id) > 10  # UUID

    def test_default_ttl(self) -> None:
        entry = MemoryEntry(content="test", memory_type="short_term")
        assert entry.ttl == -1

    def test_default_is_deleted(self) -> None:
        entry = MemoryEntry(content="test", memory_type="short_term")
        assert entry.is_deleted is False

    def test_custom_ttl(self) -> None:
        entry = MemoryEntry(content="test", memory_type="short_term", ttl=3600)
        assert entry.ttl == 3600

    def test_embedding_field(self) -> None:
        entry = MemoryEntry(
            content="test", memory_type="short_term",
            embedding=[0.1, 0.2, 0.3],
        )
        assert entry.embedding == [0.1, 0.2, 0.3]

    def test_embedding_excluded_from_dict(self) -> None:
        entry = MemoryEntry(
            content="test", memory_type="short_term",
            embedding=[0.1, 0.2, 0.3],
        )
        d = entry.model_dump()
        assert "embedding" not in d

    def test_unique_memory_ids(self) -> None:
        e1 = MemoryEntry(content="a", memory_type="short_term")
        e2 = MemoryEntry(content="b", memory_type="short_term")
        assert e1.memory_id != e2.memory_id


# ═══════════════════════════════════════════════════════
# PersistentMemory 新方法测试
# ═══════════════════════════════════════════════════════


class TestPersistentMemoryV2:
    """PersistentMemory soft_delete / get_by_id 测试。"""

    def test_soft_delete(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        entry = MemoryEntry(
            content="要删除的记忆",
            memory_type="long_term",
            session_id="sess1",
        )
        pm.save(entry, "u1")

        result = pm.soft_delete(entry.memory_id, "u1")
        assert result is True

        # 再次读取应该标记为 deleted
        loaded = pm.get_by_id(entry.memory_id, "u1")
        assert loaded is not None
        assert loaded.is_deleted is True

    def test_soft_delete_not_found(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        result = pm.soft_delete("nonexistent", "u1")
        assert result is False

    def test_get_by_id(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        entry = MemoryEntry(
            content="测试记忆",
            memory_type="long_term",
            session_id="sess1",
        )
        pm.save(entry, "u1")

        loaded = pm.get_by_id(entry.memory_id, "u1")
        assert loaded is not None
        assert loaded.content == "测试记忆"
        assert loaded.memory_id == entry.memory_id

    def test_get_by_id_not_found(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        assert pm.get_by_id("nonexistent", "u1") is None

    def test_soft_deleted_excluded_from_search(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        entry = MemoryEntry(
            content="会被删除的内容",
            memory_type="long_term",
            session_id="sess1",
            tags=["target"],
        )
        pm.save(entry, "u1")
        pm.soft_delete(entry.memory_id, "u1")

        results = pm.search_by_tag("u1", "target")
        assert len(results) == 0

    def test_soft_deleted_excluded_from_keyword_search(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        entry = MemoryEntry(
            content="独特关键词XYZ123",
            memory_type="long_term",
            session_id="sess1",
        )
        pm.save(entry, "u1")
        pm.soft_delete(entry.memory_id, "u1")

        results = pm.search_by_keyword("u1", "XYZ123")
        assert len(results) == 0

    def test_frontmatter_roundtrip_new_fields(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        entry = MemoryEntry(
            content="测试新字段",
            memory_type="long_term",
            session_id="sess1",
            ttl=3600,
        )
        pm.save(entry, "u1")

        loaded = pm.get_by_id(entry.memory_id, "u1")
        assert loaded is not None
        assert loaded.ttl == 3600
        assert loaded.memory_id == entry.memory_id


# ═══════════════════════════════════════════════════════
# ConflictDetector 测试
# ═══════════════════════════════════════════════════════


class TestConflictDetector:
    """Mem0 式冲突检测测试。"""

    @pytest.mark.asyncio
    async def test_no_similar_returns_add(self) -> None:
        mock_vm = AsyncMock()
        mock_vm.search_similar.return_value = []

        mock_embedder = MagicMock()
        detector = ConflictDetector(mock_vm, mock_embedder, threshold=0.85)

        entry = MemoryEntry(content="全新的记忆", memory_type="long_term")
        op = await detector.detect(entry, "u1")

        assert op.action == "ADD"

    @pytest.mark.asyncio
    async def test_similar_content_returns_update(self) -> None:
        existing = MemoryEntry(
            memory_id="existing-123",
            content="Python 是解释型语言",
            memory_type="long_term",
        )

        mock_vm = AsyncMock()
        mock_vm.search_similar.return_value = [(existing, 0.92)]

        mock_embedder = MagicMock()
        detector = ConflictDetector(mock_vm, mock_embedder, threshold=0.85)

        entry = MemoryEntry(content="Python 是一种解释型编程语言", memory_type="long_term")
        op = await detector.detect(entry, "u1")

        assert op.action == "UPDATE"
        assert op.target_id == "existing-123"

    @pytest.mark.asyncio
    async def test_identical_content_returns_noop(self) -> None:
        existing = MemoryEntry(
            memory_id="existing-456",
            content="测试内容",
            memory_type="long_term",
        )

        mock_vm = AsyncMock()
        mock_vm.search_similar.return_value = [(existing, 0.99)]

        mock_embedder = MagicMock()
        detector = ConflictDetector(mock_vm, mock_embedder, threshold=0.85)

        entry = MemoryEntry(content="测试内容", memory_type="long_term")
        op = await detector.detect(entry, "u1")

        assert op.action == "NOOP"

    @pytest.mark.asyncio
    async def test_detection_error_fallback_to_add(self) -> None:
        mock_vm = AsyncMock()
        mock_vm.search_similar.side_effect = ConnectionError("db down")

        mock_embedder = MagicMock()
        detector = ConflictDetector(mock_vm, mock_embedder)

        entry = MemoryEntry(content="test", memory_type="long_term")
        op = await detector.detect(entry, "u1")

        assert op.action == "ADD"
        assert "异常" in op.reason or "降级" in op.reason


# ═══════════════════════════════════════════════════════
# MemoryOperator 测试
# ═══════════════════════════════════════════════════════


class TestMemoryOperator:
    """记忆操作执行器测试。"""

    @pytest.mark.asyncio
    async def test_execute_add(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        operator = MemoryOperator(pm, audit_log_dir=str(tmp_path))

        entry = MemoryEntry(
            content="新记忆", memory_type="long_term", session_id="s1",
        )
        op = MemoryOp(action="ADD", reason="新增")
        result = await operator.execute(op, entry, "u1")

        assert result.content == "新记忆"

    @pytest.mark.asyncio
    async def test_execute_update(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        # 先保存一条旧记忆
        old_entry = MemoryEntry(
            content="旧内容", memory_type="long_term", session_id="s1",
        )
        pm.save(old_entry, "u1")

        operator = MemoryOperator(pm, audit_log_dir=str(tmp_path))
        new_entry = MemoryEntry(
            content="新内容", memory_type="long_term", session_id="s1",
        )
        op = MemoryOp(action="UPDATE", target_id=old_entry.memory_id, reason="更新")
        result = await operator.execute(op, new_entry, "u1")

        assert result.content == "新内容"
        # 旧记忆应被软删除
        old = pm.get_by_id(old_entry.memory_id, "u1")
        assert old is not None
        assert old.is_deleted is True

    @pytest.mark.asyncio
    async def test_execute_noop(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        operator = MemoryOperator(pm, audit_log_dir=str(tmp_path))

        entry = MemoryEntry(content="不变", memory_type="long_term")
        op = MemoryOp(action="NOOP", reason="内容相同")
        result = await operator.execute(op, entry, "u1")

        assert result.content == "不变"

    @pytest.mark.asyncio
    async def test_execute_delete(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        entry = MemoryEntry(
            content="将被删除", memory_type="long_term", session_id="s1",
        )
        pm.save(entry, "u1")

        operator = MemoryOperator(pm, audit_log_dir=str(tmp_path))
        op = MemoryOp(action="DELETE", target_id=entry.memory_id, reason="用户删除")
        result = await operator.execute(op, entry, "u1")

        assert result.is_deleted is True

    @pytest.mark.asyncio
    async def test_audit_log_written(self, tmp_path: Path) -> None:
        pm = PersistentMemory(str(tmp_path))
        operator = MemoryOperator(pm, audit_log_dir=str(tmp_path))

        entry = MemoryEntry(content="test", memory_type="long_term")
        op = MemoryOp(action="UPDATE", target_id="old-id", reason="测试审计")
        await operator.execute(op, entry, "u1")

        log_file = tmp_path / "memory_audit.log"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "UPDATE" in content
        assert "old-id" in content


# ═══════════════════════════════════════════════════════
# MemoryManager v2 集成测试
# ═══════════════════════════════════════════════════════


class TestMemoryManagerV2:
    """MemoryManager 升级版集成测试。"""

    def test_add_sync_still_works(self, tmp_path: Path) -> None:
        mm = MemoryManager(vault_path=str(tmp_path))
        entry = mm.add("sess1", "u1", "测试", memory_type="short_term")
        assert entry.content == "测试"
        assert entry.memory_id

    def test_add_long_term_sync(self, tmp_path: Path) -> None:
        mm = MemoryManager(vault_path=str(tmp_path))
        entry = mm.add("sess1", "u1", "长期记忆", memory_type="long_term", tags=["test"])
        assert entry.memory_type == "long_term"

    @pytest.mark.asyncio
    async def test_add_with_conflict_check_no_vector(self, tmp_path: Path) -> None:
        """无向量层时降级到 Obsidian 纯写入。"""
        mm = MemoryManager(vault_path=str(tmp_path))
        entry = await mm.add_with_conflict_check(
            "sess1", "u1", "降级测试", memory_type="long_term",
        )
        assert entry.content == "降级测试"

    @pytest.mark.asyncio
    async def test_add_with_conflict_check_add_op(self, tmp_path: Path) -> None:
        """有向量层，冲突检测返回 ADD。"""
        mm = MemoryManager(vault_path=str(tmp_path))

        mock_vm = AsyncMock()
        mock_detector = AsyncMock()
        mock_detector.detect.return_value = MemoryOp(action="ADD", reason="新增")

        mock_operator = AsyncMock()
        mock_operator.execute.side_effect = lambda op, entry, uid: entry

        mm.inject_vector_memory(mock_vm, mock_detector, mock_operator)

        entry = await mm.add_with_conflict_check(
            "sess1", "u1", "新记忆", memory_type="long_term",
        )
        assert entry.content == "新记忆"
        mock_detector.detect.assert_called_once()
        mock_operator.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, tmp_path: Path) -> None:
        """TTL 过期清理测试。"""
        mm = MemoryManager(vault_path=str(tmp_path))

        # 创建一条 TTL=1 秒的记忆，created_at 设为过去
        entry = MemoryEntry(
            content="过期记忆",
            memory_type="long_term",
            session_id="sess1",
            ttl=1,
            created_at=datetime.now() - timedelta(seconds=10),
        )
        mm.persistent.save(entry, "u1")

        cleaned = await mm.cleanup_expired("u1")
        assert cleaned == 1

        # 验证已被软删除
        loaded = mm.persistent.get_by_id(entry.memory_id, "u1")
        assert loaded is not None
        assert loaded.is_deleted is True

    @pytest.mark.asyncio
    async def test_cleanup_non_expired_untouched(self, tmp_path: Path) -> None:
        """未过期记忆不被清理。"""
        mm = MemoryManager(vault_path=str(tmp_path))

        entry = MemoryEntry(
            content="未过期记忆",
            memory_type="long_term",
            session_id="sess1",
            ttl=99999,
        )
        mm.persistent.save(entry, "u1")

        cleaned = await mm.cleanup_expired("u1")
        assert cleaned == 0

    @pytest.mark.asyncio
    async def test_cleanup_no_ttl_untouched(self, tmp_path: Path) -> None:
        """ttl=-1 的记忆永不过期。"""
        mm = MemoryManager(vault_path=str(tmp_path))

        entry = MemoryEntry(
            content="永不过期",
            memory_type="long_term",
            session_id="sess1",
            ttl=-1,
            created_at=datetime.now() - timedelta(days=365),
        )
        mm.persistent.save(entry, "u1")

        cleaned = await mm.cleanup_expired("u1")
        assert cleaned == 0
