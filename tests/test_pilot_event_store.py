"""EventStore / SnapshotStore 契约测试。

InMemory 与 SQLite 共享同一组测试，行为不得分叉。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from agent_hub.pilot import (
    ConcurrencyConflict,
    EventType,
    ExecutionEvent,
    InMemoryEventStore,
    SQLiteEventStore,
    Task,
    TaskAction,
    Workspace,
    transition,
)

# ── 参数化两种实现 ───────────────────────────────────


@pytest_asyncio.fixture(params=["memory", "sqlite"])
async def store(request, tmp_path) -> AsyncIterator[object]:
    if request.param == "memory":
        s = InMemoryEventStore()
        yield s
    else:
        db = tmp_path / "pilot.db"
        s = SQLiteEventStore(str(db))
        try:
            yield s
        finally:
            s.close()


# ── Helpers ──────────────────────────────────────────


def _evt(workspace_id: str, *, idem: str | None = None, msg: str = "") -> ExecutionEvent:
    return ExecutionEvent(
        workspace_id=workspace_id,
        type=EventType.TASK_CREATED,
        message=msg,
        idempotency_key=idem,
    )


def _make_workspace() -> Workspace:
    return Workspace(title="t", source_channel="api", created_by="u1")


# ── EventStore 契约 ──────────────────────────────────


@pytest.mark.asyncio
async def test_append_assigns_monotonic_sequence(store) -> None:
    ws = "ws_a"
    a = await store.append_event(_evt(ws, msg="1"))
    b = await store.append_event(_evt(ws, msg="2"))
    c = await store.append_event(_evt(ws, msg="3"))
    assert (a.sequence, b.sequence, c.sequence) == (1, 2, 3)


@pytest.mark.asyncio
async def test_workspaces_have_independent_sequences(store) -> None:
    a1 = await store.append_event(_evt("ws_a"))
    b1 = await store.append_event(_evt("ws_b"))
    a2 = await store.append_event(_evt("ws_a"))
    assert a1.sequence == 1 and b1.sequence == 1 and a2.sequence == 2


@pytest.mark.asyncio
async def test_idempotency_returns_first_event(store) -> None:
    e1 = await store.append_event(_evt("ws", idem="k1", msg="first"))
    e2 = await store.append_event(_evt("ws", idem="k1", msg="second"))
    assert e1.event_id == e2.event_id
    assert e2.message == "first"
    # 不应推进 sequence
    e3 = await store.append_event(_evt("ws", msg="third"))
    assert e3.sequence == 2


@pytest.mark.asyncio
async def test_list_events_since_sequence(store) -> None:
    ws = "ws_x"
    for i in range(5):
        await store.append_event(_evt(ws, msg=str(i)))
    got = await store.list_events(ws, since_sequence=2)
    assert [e.sequence for e in got] == [3, 4, 5]


@pytest.mark.asyncio
async def test_list_events_limit(store) -> None:
    ws = "ws_y"
    for _ in range(10):
        await store.append_event(_evt(ws))
    got = await store.list_events(ws, limit=3)
    assert len(got) == 3
    assert [e.sequence for e in got] == [1, 2, 3]


# ── SnapshotStore 契约 ───────────────────────────────


@pytest.mark.asyncio
async def test_put_get_snapshot_roundtrip(store) -> None:
    ws = _make_workspace()
    await store.put_snapshot(ws)
    got = await store.get_snapshot("workspace", ws.workspace_id)
    assert got == ws


@pytest.mark.asyncio
async def test_snapshot_expected_version_conflict(store) -> None:
    ws = _make_workspace()
    await store.put_snapshot(ws, expected_version=0)  # 首次插入
    with pytest.raises(ConcurrencyConflict):
        await store.put_snapshot(ws, expected_version=99)


@pytest.mark.asyncio
async def test_snapshot_update_via_transition(store) -> None:
    ws = _make_workspace()
    await store.put_snapshot(ws, expected_version=0)

    task = Task(workspace_id=ws.workspace_id, origin_text="x", requester_id="u1")
    await store.put_snapshot(task, expected_version=0)

    task2 = transition(task, TaskAction.START_PLANNING)
    await store.put_snapshot(task2, expected_version=1)

    got = await store.get_snapshot("task", task.task_id)
    assert got.version == 2
    assert got.status.value == "planning"


@pytest.mark.asyncio
async def test_list_snapshots_by_workspace(store) -> None:
    ws1 = _make_workspace()
    ws2 = _make_workspace()
    await store.put_snapshot(ws1)
    await store.put_snapshot(ws2)
    t1 = Task(workspace_id=ws1.workspace_id, origin_text="a", requester_id="u")
    t2 = Task(workspace_id=ws2.workspace_id, origin_text="b", requester_id="u")
    await store.put_snapshot(t1)
    await store.put_snapshot(t2)

    got = await store.list_snapshots("task", workspace_id=ws1.workspace_id)
    assert [t.task_id for t in got] == [t1.task_id]


@pytest.mark.asyncio
async def test_get_unknown_snapshot_returns_none(store) -> None:
    assert await store.get_snapshot("task", "nope") is None


# ── SQLite 持久化专项 ────────────────────────────────


@pytest.mark.asyncio
async def test_sqlite_persists_after_reopen(tmp_path) -> None:
    db = tmp_path / "p.db"
    s1 = SQLiteEventStore(str(db))
    ws = _make_workspace()
    await s1.put_snapshot(ws)
    await s1.append_event(_evt(ws.workspace_id, msg="hi"))
    s1.close()

    s2 = SQLiteEventStore(str(db))
    try:
        got_ws = await s2.get_snapshot("workspace", ws.workspace_id)
        assert got_ws is not None and got_ws.workspace_id == ws.workspace_id
        events = await s2.list_events(ws.workspace_id)
        assert len(events) == 1 and events[0].message == "hi"
    finally:
        s2.close()
