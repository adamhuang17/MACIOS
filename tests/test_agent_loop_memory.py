from __future__ import annotations

from agent_hub.config.settings import Settings
from agent_hub.contracts.identity import SourceChatType, SourceContext, UserContext, UserRole
from agent_hub.core.agent_loop import AgentTurnLoop


def test_system_prompt_includes_memory_context_when_available() -> None:
    loop = AgentTurnLoop(
        settings=Settings(llm_api_key="test-key", llm_base_url="http://localhost"),
        tools=[],
        client=object(),
        max_rounds=1,
    )

    prompt = loop._build_system_prompt(
        user_context=UserContext(user_id="u1", role=UserRole.USER),
        source_context=SourceContext(
            channel="feishu",
            chat_id="oc_1",
            chat_type=SourceChatType.DIRECT,
            sender_id="u1",
        ),
        session_id="feishu:oc_1:u1",
        memory_context="用户偏好：正式、简洁、大气的中文 PPT。",
    )

    assert "Relevant memory" in prompt
    assert "用户偏好：正式、简洁、大气的中文 PPT。" in prompt


def test_system_prompt_omits_memory_section_when_context_empty() -> None:
    loop = AgentTurnLoop(
        settings=Settings(llm_api_key="test-key", llm_base_url="http://localhost"),
        tools=[],
        client=object(),
        max_rounds=1,
    )

    prompt = loop._build_system_prompt(
        user_context=UserContext(user_id="u1", role=UserRole.USER),
        source_context=SourceContext(
            channel="api",
            chat_id="chat-1",
            chat_type=SourceChatType.DIRECT,
            sender_id="u1",
        ),
        session_id="api:chat-1:u1",
        memory_context="",
    )

    assert "Relevant memory" not in prompt
