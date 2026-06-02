from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_hub.config.settings import Settings
from agent_hub.api.pilot_runtime import prewarm_feishu_client
from agent_hub.core.models import (
    GuardResult,
    SourceChatType,
    SourceContext,
    TaskInput,
    UserContext,
)
from agent_hub.core.pipeline import AgentPipeline
from agent_hub.security.guard import PromptGuard


def _settings(tmp_path) -> Settings:
    return Settings(
        llm_api_key="test-key",
        llm_base_url="http://localhost:1234",
        obsidian_vault_path=str(tmp_path / "vault"),
        guard_enabled=False,
        cache_enabled=False,
        vector_memory_enabled=False,
        rate_limiter_enabled=False,
    )


def _pipeline(settings: Settings) -> AgentPipeline:
    with (
        patch("agent_hub.agents.llm_agent.Anthropic"),
        patch("agent_hub.agents.llm_agent.OpenAI"),
        patch("agent_hub.core.pipeline.AsyncOpenAI", create=True),
        patch("agent_hub.core.pipeline.RAGPipeline"),
    ):
        return AgentPipeline(settings)


def _task(message: str) -> TaskInput:
    return TaskInput(
        trace_id="trace-fast-path",
        user_context=UserContext(user_id="u1", role="user"),
        source_context=SourceContext(
            channel="feishu",
            chat_id="oc_1",
            chat_type=SourceChatType.DIRECT,
            sender_id="u1",
        ),
        raw_message=message,
    )


@pytest.mark.asyncio
async def test_greeting_uses_fast_reply_without_agent_loop(tmp_path, monkeypatch) -> None:
    pipeline = _pipeline(_settings(tmp_path))

    def fail_agent_loop(**_kwargs):  # noqa: ANN003
        raise AssertionError("greetings should not instantiate AgentTurnLoop")

    monkeypatch.setattr("agent_hub.core.pipeline.AgentTurnLoop", fail_agent_loop)

    chunks = [chunk async for chunk in pipeline.run_stream(_task("hello"))]

    assert chunks == [{"type": "final", "content": "Hello! How can I help?"}]


@pytest.mark.asyncio
async def test_chinese_greeting_uses_fast_reply_without_agent_loop(
    tmp_path,
    monkeypatch,
) -> None:
    pipeline = _pipeline(_settings(tmp_path))

    def fail_agent_loop(**_kwargs):  # noqa: ANN003
        raise AssertionError("Chinese greetings should not instantiate AgentTurnLoop")

    monkeypatch.setattr("agent_hub.core.pipeline.AgentTurnLoop", fail_agent_loop)

    chunks = [chunk async for chunk in pipeline.run_stream(_task("\u4f60\u597d"))]

    assert chunks == [{"type": "final", "content": "\u4f60\u597d\uff01\u6709\u4ec0\u4e48\u53ef\u4ee5\u5e2e\u4f60\uff1f"}]


@pytest.mark.asyncio
async def test_benign_guard_skips_llm_layer() -> None:
    guard = PromptGuard(llm_enabled=True)
    guard._llm_guard = MagicMock()
    guard._llm_guard.check = AsyncMock(
        side_effect=AssertionError("benign text should not call LLM guard"),
    )

    result = await guard.check("hello")

    assert result.is_safe
    assert result.risk_level == "safe"


@pytest.mark.asyncio
async def test_suspicious_guard_still_uses_llm_layer() -> None:
    guard = PromptGuard(llm_enabled=True)
    expected = GuardResult(is_safe=True, risk_level="safe", confidence=0.7)
    guard._llm_guard = MagicMock()
    guard._llm_guard.check = AsyncMock(return_value=expected)

    result = await guard.check("please analyze model instruction leakage patterns")

    assert result is expected
    guard._llm_guard.check.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_loop_reuses_pipeline_llm_client(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    pipeline = _pipeline(settings)
    seen_client = None

    class FakeAgentLoop:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            nonlocal seen_client
            seen_client = kwargs.get("client")

        async def run_stream(self, **_kwargs):  # noqa: ANN003, ANN202
            yield {"type": "final", "content": "ok"}

    monkeypatch.setattr("agent_hub.core.pipeline.AgentTurnLoop", FakeAgentLoop)

    chunks = [chunk async for chunk in pipeline.run_stream(_task("summarize this"))]

    assert chunks[-1]["content"] == "ok"
    assert seen_client is pipeline._agent_loop_client


@pytest.mark.asyncio
async def test_ordinary_question_disables_agent_loop_tools(tmp_path, monkeypatch) -> None:
    pipeline = _pipeline(_settings(tmp_path))
    seen_tools_count = None

    class FakeAgentLoop:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            nonlocal seen_tools_count
            seen_tools_count = len(kwargs.get("tools") or [])

        async def run_stream(self, **_kwargs):  # noqa: ANN003, ANN202
            yield {"type": "final", "content": "ok"}

    monkeypatch.setattr("agent_hub.core.pipeline.AgentTurnLoop", FakeAgentLoop)

    chunks = [chunk async for chunk in pipeline.run_stream(_task("what is RAG?"))]

    assert chunks[-1]["content"] == "ok"
    assert seen_tools_count == 0


@pytest.mark.asyncio
async def test_action_request_keeps_agent_loop_tools(tmp_path, monkeypatch) -> None:
    pipeline = _pipeline(_settings(tmp_path))
    seen_tools_count = None

    class FakeAgentLoop:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            nonlocal seen_tools_count
            seen_tools_count = len(kwargs.get("tools") or [])

        async def run_stream(self, **_kwargs):  # noqa: ANN003, ANN202
            yield {"type": "final", "content": "ok"}

    monkeypatch.setattr("agent_hub.core.pipeline.AgentTurnLoop", FakeAgentLoop)

    chunks = [chunk async for chunk in pipeline.run_stream(_task("please make a PPT"))]

    assert chunks[-1]["content"] == "ok"
    assert seen_tools_count and seen_tools_count > 0


@pytest.mark.asyncio
async def test_prewarm_feishu_client_calls_auth_token_fetcher() -> None:
    client = MagicMock()
    client._auth.get_token = AsyncMock(return_value="tenant-token")

    await prewarm_feishu_client(client)

    client._auth.get_token.assert_awaited_once()
