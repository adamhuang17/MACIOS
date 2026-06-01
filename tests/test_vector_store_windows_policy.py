from __future__ import annotations

from agent_hub.rag import vector_store


def test_windows_proactor_policy_switches_to_selector(monkeypatch) -> None:
    class FakeProactorPolicy:
        pass

    class FakeSelectorPolicy:
        pass

    proactor = FakeProactorPolicy()
    selector = FakeSelectorPolicy()
    applied: list[object] = []

    monkeypatch.setattr(vector_store.sys, "platform", "win32")
    monkeypatch.setattr(
        vector_store.asyncio,
        "WindowsProactorEventLoopPolicy",
        FakeProactorPolicy,
        raising=False,
    )
    monkeypatch.setattr(
        vector_store.asyncio,
        "WindowsSelectorEventLoopPolicy",
        lambda: selector,
        raising=False,
    )
    monkeypatch.setattr(vector_store.asyncio, "get_event_loop_policy", lambda: proactor)
    monkeypatch.setattr(vector_store.asyncio, "set_event_loop_policy", applied.append)

    vector_store._ensure_windows_selector_event_loop_policy()

    assert applied == [selector]
