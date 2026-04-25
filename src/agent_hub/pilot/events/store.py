"""EventStore / SnapshotStore 协议与默认实现。

设计原则：
- EventStore 只管事件追加与读取，``sequence`` 由存储分配。
- SnapshotStore 只管实体快照读写，支持 ``expected_version`` 乐观并发。
- 存储层不调用 ``transition``，不做业务判断。
- InMemory 与 SQLite 实现行为一致，用于跨场景共享契约测试。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, runtime_checkable

from agent_hub.pilot.domain.errors import ConcurrencyConflict
from agent_hub.pilot.domain.events import ExecutionEvent
from agent_hub.pilot.domain.models import (
    Approval,
    Artifact,
    Plan,
    PlanStep,
    Task,
    Workspace,
)


_ENTITY_REGISTRY: dict[str, type[Any]] = {
    "workspace": Workspace,
    "task": Task,
    "plan": Plan,
    "plan_step": PlanStep,
    "approval": Approval,
    "artifact": Artifact,
}


def _entity_type_of(obj: Any) -> str:
    for name, cls in _ENTITY_REGISTRY.items():
        if isinstance(obj, cls):
            return name
    msg = f"未注册的实体类型: {type(obj).__name__}"
    raise TypeError(msg)


def _id_of(obj: Any) -> str:
    for attr in ("workspace_id", "task_id", "plan_id", "step_id", "approval_id", "artifact_id"):
        if hasattr(obj, attr) and isinstance(getattr(obj, attr), str):
            # 优先返回与实体名匹配的主键
            mapping = {
                Workspace: "workspace_id",
                Task: "task_id",
                Plan: "plan_id",
                PlanStep: "step_id",
                Approval: "approval_id",
                Artifact: "artifact_id",
            }
            return getattr(obj, mapping[type(obj)])
    msg = f"无法提取主键: {type(obj).__name__}"
    raise TypeError(msg)


def _workspace_id_of(obj: Any) -> str:
    if isinstance(obj, Workspace):
        return obj.workspace_id
    return obj.workspace_id  # 其它实体均带 workspace_id


# ── 协议 ─────────────────────────────────────────────


@runtime_checkable
class EventStore(Protocol):
    """事件存储合约。"""

    async def append_event(self, event: ExecutionEvent) -> ExecutionEvent: ...

    async def list_events(
        self,
        workspace_id: str,
        *,
        since_sequence: int = 0,
        limit: int = 500,
    ) -> list[ExecutionEvent]: ...


@runtime_checkable
class SnapshotStore(Protocol):
    """实体快照存储合约。"""

    async def put_snapshot(
        self,
        entity: Any,
        *,
        expected_version: Optional[int] = None,
    ) -> None: ...

    async def get_snapshot(
        self,
        entity_type: str,
        entity_id: str,
    ) -> Optional[Any]: ...

    async def list_snapshots(
        self,
        entity_type: str,
        *,
        workspace_id: Optional[str] = None,
    ) -> list[Any]: ...


# ── InMemory 实现 ────────────────────────────────────


@dataclass
class _MemoryEventRecord:
    event: ExecutionEvent


class InMemoryEventStore:
    """进程内 EventStore + SnapshotStore，用于测试与 demo mode。"""

    def __init__(self) -> None:
        self._events: dict[str, list[ExecutionEvent]] = {}
        self._next_sequence: dict[str, int] = {}
        self._idempotency: dict[tuple[str, str], ExecutionEvent] = {}
        self._snapshots: dict[tuple[str, str], Any] = {}
        self._lock = asyncio.Lock()

    # ── EventStore ────────────────────────────────

    async def append_event(self, event: ExecutionEvent) -> ExecutionEvent:
        async with self._lock:
            ws = event.workspace_id
            if event.idempotency_key:
                key = (ws, event.idempotency_key)
                existing = self._idempotency.get(key)
                if existing is not None:
                    return existing

            seq = self._next_sequence.get(ws, 0) + 1
            self._next_sequence[ws] = seq
            stored = event.model_copy(update={"sequence": seq})
            self._events.setdefault(ws, []).append(stored)
            if event.idempotency_key:
                self._idempotency[(ws, event.idempotency_key)] = stored
            return stored

    async def list_events(
        self,
        workspace_id: str,
        *,
        since_sequence: int = 0,
        limit: int = 500,
    ) -> list[ExecutionEvent]:
        async with self._lock:
            events = self._events.get(workspace_id, [])
            filtered = [e for e in events if e.sequence > since_sequence]
            return filtered[:limit]

    # ── SnapshotStore ─────────────────────────────

    async def put_snapshot(
        self,
        entity: Any,
        *,
        expected_version: Optional[int] = None,
    ) -> None:
        entity_type = _entity_type_of(entity)
        entity_id = _id_of(entity)
        async with self._lock:
            existing = self._snapshots.get((entity_type, entity_id))
            if expected_version is not None:
                actual = existing.version if existing is not None else 0
                if actual != expected_version:
                    raise ConcurrencyConflict(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        expected_version=expected_version,
                        actual_version=actual,
                    )
            self._snapshots[(entity_type, entity_id)] = entity

    async def get_snapshot(
        self,
        entity_type: str,
        entity_id: str,
    ) -> Optional[Any]:
        async with self._lock:
            return self._snapshots.get((entity_type, entity_id))

    async def list_snapshots(
        self,
        entity_type: str,
        *,
        workspace_id: Optional[str] = None,
    ) -> list[Any]:
        async with self._lock:
            results = [
                v for (t, _), v in self._snapshots.items() if t == entity_type
            ]
            if workspace_id is not None:
                results = [v for v in results if _workspace_id_of(v) == workspace_id]
            return results


# ── SQLite 实现 ──────────────────────────────────────


_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS pilot_events (
    workspace_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    event_id TEXT NOT NULL,
    idempotency_key TEXT,
    type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (workspace_id, sequence)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_pilot_events_event_id
    ON pilot_events(event_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_pilot_events_idem
    ON pilot_events(workspace_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS pilot_snapshots (
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (entity_type, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_pilot_snapshots_ws
    ON pilot_snapshots(entity_type, workspace_id);
"""


class SQLiteEventStore:
    """SQLite 实现的 EventStore + SnapshotStore。

    使用标准库 ``sqlite3``，所有写入在事务内完成。
    通过 ``threading.Lock`` 保护 sqlite 连接的多线程访问，
    并用 ``asyncio.to_thread`` 桥接到异步接口。
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            isolation_level=None,  # 手动管理事务
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SQLITE_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── EventStore ────────────────────────────────

    async def append_event(self, event: ExecutionEvent) -> ExecutionEvent:
        return await asyncio.to_thread(self._append_event_sync, event)

    def _append_event_sync(self, event: ExecutionEvent) -> ExecutionEvent:
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE")

                if event.idempotency_key:
                    row = cur.execute(
                        "SELECT payload_json FROM pilot_events "
                        "WHERE workspace_id = ? AND idempotency_key = ?",
                        (event.workspace_id, event.idempotency_key),
                    ).fetchone()
                    if row is not None:
                        cur.execute("COMMIT")
                        return ExecutionEvent.model_validate_json(row[0])

                row = cur.execute(
                    "SELECT COALESCE(MAX(sequence), 0) FROM pilot_events "
                    "WHERE workspace_id = ?",
                    (event.workspace_id,),
                ).fetchone()
                seq = int(row[0]) + 1
                stored = event.model_copy(update={"sequence": seq})
                payload = stored.model_dump_json()
                cur.execute(
                    "INSERT INTO pilot_events "
                    "(workspace_id, sequence, event_id, idempotency_key, "
                    "type, payload_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        stored.workspace_id,
                        stored.sequence,
                        stored.event_id,
                        stored.idempotency_key,
                        stored.type.value,
                        payload,
                        stored.created_at.astimezone(timezone.utc).isoformat(),
                    ),
                )
                cur.execute("COMMIT")
                return stored
            except Exception:
                cur.execute("ROLLBACK")
                raise

    async def list_events(
        self,
        workspace_id: str,
        *,
        since_sequence: int = 0,
        limit: int = 500,
    ) -> list[ExecutionEvent]:
        return await asyncio.to_thread(
            self._list_events_sync, workspace_id, since_sequence, limit,
        )

    def _list_events_sync(
        self,
        workspace_id: str,
        since_sequence: int,
        limit: int,
    ) -> list[ExecutionEvent]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload_json FROM pilot_events "
                "WHERE workspace_id = ? AND sequence > ? "
                "ORDER BY sequence ASC LIMIT ?",
                (workspace_id, since_sequence, limit),
            ).fetchall()
        return [ExecutionEvent.model_validate_json(r[0]) for r in rows]

    # ── SnapshotStore ─────────────────────────────

    async def put_snapshot(
        self,
        entity: Any,
        *,
        expected_version: Optional[int] = None,
    ) -> None:
        await asyncio.to_thread(self._put_snapshot_sync, entity, expected_version)

    def _put_snapshot_sync(
        self,
        entity: Any,
        expected_version: Optional[int],
    ) -> None:
        entity_type = _entity_type_of(entity)
        entity_id = _id_of(entity)
        workspace_id = _workspace_id_of(entity)
        payload = entity.model_dump_json()
        updated_at = entity.updated_at.astimezone(timezone.utc).isoformat()

        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE")
                row = cur.execute(
                    "SELECT version FROM pilot_snapshots "
                    "WHERE entity_type = ? AND entity_id = ?",
                    (entity_type, entity_id),
                ).fetchone()
                actual = int(row[0]) if row is not None else 0
                if expected_version is not None and actual != expected_version:
                    cur.execute("ROLLBACK")
                    raise ConcurrencyConflict(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        expected_version=expected_version,
                        actual_version=actual,
                    )
                if row is None:
                    cur.execute(
                        "INSERT INTO pilot_snapshots "
                        "(entity_type, entity_id, workspace_id, version, "
                        "payload_json, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            entity_type,
                            entity_id,
                            workspace_id,
                            entity.version,
                            payload,
                            updated_at,
                        ),
                    )
                else:
                    cur.execute(
                        "UPDATE pilot_snapshots SET workspace_id = ?, "
                        "version = ?, payload_json = ?, updated_at = ? "
                        "WHERE entity_type = ? AND entity_id = ?",
                        (
                            workspace_id,
                            entity.version,
                            payload,
                            updated_at,
                            entity_type,
                            entity_id,
                        ),
                    )
                cur.execute("COMMIT")
            except ConcurrencyConflict:
                raise
            except Exception:
                cur.execute("ROLLBACK")
                raise

    async def get_snapshot(
        self,
        entity_type: str,
        entity_id: str,
    ) -> Optional[Any]:
        return await asyncio.to_thread(self._get_snapshot_sync, entity_type, entity_id)

    def _get_snapshot_sync(
        self,
        entity_type: str,
        entity_id: str,
    ) -> Optional[Any]:
        cls = _ENTITY_REGISTRY.get(entity_type)
        if cls is None:
            msg = f"未知 entity_type: {entity_type}"
            raise ValueError(msg)
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM pilot_snapshots "
                "WHERE entity_type = ? AND entity_id = ?",
                (entity_type, entity_id),
            ).fetchone()
        if row is None:
            return None
        return cls.model_validate_json(row[0])

    async def list_snapshots(
        self,
        entity_type: str,
        *,
        workspace_id: Optional[str] = None,
    ) -> list[Any]:
        return await asyncio.to_thread(
            self._list_snapshots_sync, entity_type, workspace_id,
        )

    def _list_snapshots_sync(
        self,
        entity_type: str,
        workspace_id: Optional[str],
    ) -> list[Any]:
        cls = _ENTITY_REGISTRY.get(entity_type)
        if cls is None:
            msg = f"未知 entity_type: {entity_type}"
            raise ValueError(msg)
        with self._lock:
            if workspace_id is None:
                rows = self._conn.execute(
                    "SELECT payload_json FROM pilot_snapshots WHERE entity_type = ?",
                    (entity_type,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT payload_json FROM pilot_snapshots "
                    "WHERE entity_type = ? AND workspace_id = ?",
                    (entity_type, workspace_id),
                ).fetchall()
        return [cls.model_validate_json(r[0]) for r in rows]


__all__ = [
    "EventStore",
    "InMemoryEventStore",
    "SQLiteEventStore",
    "SnapshotStore",
]


# 防止未使用警告
_ = (json, datetime, _MemoryEventRecord)
