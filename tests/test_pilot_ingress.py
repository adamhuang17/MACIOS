"""``PilotIngressService`` 单元测试。

覆盖：
- classify() 对常见消息形态（空文本、任务关键词、进度查询、问答、群聊
  无 mention 等）的分类结果。
- dispatch() 在注入 / 不注入 AgentPipeline / PilotQueryService 时的
  reply_text 行为。
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_hub.connectors.feishu.models import (
    FeishuChatType,
    FeishuInboundMessage,
    FeishuMessageType,
)
from agent_hub.pilot.services.ingress import (
    IngressIntent,
    PilotIngressService,
)


def _msg(
    *,
    text: str = "",
    chat_type: FeishuChatType = FeishuChatType.P2P,
    bot_mentioned: bool = False,
    message_type: FeishuMessageType = FeishuMessageType.TEXT,
) -> FeishuInboundMessage:
    return FeishuInboundMessage(
        event_id="evt-1",
        event_type="im.message.receive_v1",
        tenant_key="t1",
        app_id="app1",
        chat_id="oc_chat_1",
        chat_type=chat_type,
        message_id="om_1",
        message_type=message_type,
        sender_id="ou_1",
        text=text,
        bot_mentioned=bot_mentioned,
        create_time=datetime.now(UTC),
    )


# ── classify ──────────────────────────────────────


class TestClassify:
    def setup_method(self) -> None:
        self.svc = PilotIngressService()

    def test_empty_text_is_ignored(self) -> None:
        assert self.svc.classify(_msg(text="")).intent is IngressIntent.IGNORE
        assert self.svc.classify(_msg(text="   ")).intent is IngressIntent.IGNORE

    def test_non_text_message_falls_back_to_task(self) -> None:
        d = self.svc.classify(_msg(message_type=FeishuMessageType.FILE))
        assert d.intent is IngressIntent.START_TASK

    def test_task_keyword_routes_to_task(self) -> None:
        for t in [
            "帮我做一份项目汇报 PPT",
            "写一份产品 Brief",
            "整理一下会议纪要",
            "生成一个介绍文档",
        ]:
            d = self.svc.classify(_msg(text=t))
            assert d.intent is IngressIntent.START_TASK, t
            assert d.matched_keywords

    def test_progress_query_routes(self) -> None:
        for t in [
            "刚才那个任务进度怎么样？",
            "PPT 做完了吗",
            "汇报的状态如何",
            "刚才那份方案搞完没",
        ]:
            d = self.svc.classify(_msg(text=t))
            assert d.intent is IngressIntent.PROGRESS_QUERY, t

    def test_progress_word_alone_without_subject_treated_as_qa(self) -> None:
        # "进度" 不带任何任务名 / 问号 → 不应误判为查询
        d = self.svc.classify(_msg(text="进度"))
        assert d.intent is IngressIntent.ORDINARY_QA

    def test_ordinary_qa(self) -> None:
        for t in [
            "RAG 是什么？",
            "今天天气怎么样",
            "解释一下 transformer",
        ]:
            d = self.svc.classify(_msg(text=t))
            assert d.intent is IngressIntent.ORDINARY_QA, t

    def test_group_without_mention_is_ignored_for_qa(self) -> None:
        d = self.svc.classify(
            _msg(
                text="RAG 是什么？",
                chat_type=FeishuChatType.GROUP,
                bot_mentioned=False,
            )
        )
        assert d.intent is IngressIntent.IGNORE

    def test_group_with_mention_qa_passes(self) -> None:
        d = self.svc.classify(
            _msg(
                text="RAG 是什么？",
                chat_type=FeishuChatType.GROUP,
                bot_mentioned=True,
            )
        )
        assert d.intent is IngressIntent.ORDINARY_QA


# ── dispatch ──────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_qa_uses_pipeline() -> None:
    pipeline = MagicMock()
    pipeline.run = AsyncMock(
        return_value=MagicMock(response="42", status="ok"),
    )
    svc = PilotIngressService(pipeline=pipeline)
    decision = await svc.dispatch(_msg(text="生命的意义是什么"))
    assert decision.intent is IngressIntent.ORDINARY_QA
    assert decision.reply_text == "42"
    pipeline.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatch_qa_without_pipeline_returns_fallback() -> None:
    svc = PilotIngressService()
    decision = await svc.dispatch(_msg(text="解释一下 attention"))
    assert decision.intent is IngressIntent.ORDINARY_QA
    assert decision.reply_text and "暂未启用" in decision.reply_text


@pytest.mark.asyncio
async def test_dispatch_qa_pipeline_error_is_caught() -> None:
    pipeline = MagicMock()
    pipeline.run = AsyncMock(side_effect=RuntimeError("boom"))
    svc = PilotIngressService(pipeline=pipeline)
    decision = await svc.dispatch(_msg(text="解释一下 attention"))
    assert decision.intent is IngressIntent.ORDINARY_QA
    assert decision.reply_text and "暂时失败" in decision.reply_text


@pytest.mark.asyncio
async def test_dispatch_progress_query_with_no_task() -> None:
    queries = MagicMock()
    queries.list_tasks = AsyncMock(return_value=[])
    workspace = MagicMock(workspace_id="ws-1")
    lookup = AsyncMock(return_value=workspace)
    svc = PilotIngressService(queries=queries, repository_lookup=lookup)
    decision = await svc.dispatch(_msg(text="刚才那个任务进度如何？"))
    assert decision.intent is IngressIntent.PROGRESS_QUERY
    assert decision.reply_text and "没有进行中" in decision.reply_text


@pytest.mark.asyncio
async def test_dispatch_progress_query_with_task() -> None:
    queries = MagicMock()
    summary = MagicMock(
        title="测试任务",
        pending_approvals=2,
        artifact_count=3,
    )
    summary.status = MagicMock(value="running")
    queries.list_tasks = AsyncMock(return_value=[summary])
    workspace = MagicMock(workspace_id="ws-1")
    lookup = AsyncMock(return_value=workspace)
    svc = PilotIngressService(queries=queries, repository_lookup=lookup)
    decision = await svc.dispatch(_msg(text="进度怎么样？"))
    assert decision.intent is IngressIntent.PROGRESS_QUERY
    assert decision.reply_text and "测试任务" in decision.reply_text
    assert "running" in decision.reply_text


@pytest.mark.asyncio
async def test_dispatch_progress_no_workspace() -> None:
    queries = MagicMock()
    lookup = AsyncMock(return_value=None)
    svc = PilotIngressService(queries=queries, repository_lookup=lookup)
    decision = await svc.dispatch(_msg(text="进度如何？"))
    assert decision.intent is IngressIntent.PROGRESS_QUERY
    assert decision.reply_text and "还没有创建过任务" in decision.reply_text


@pytest.mark.asyncio
async def test_dispatch_task_passthrough() -> None:
    svc = PilotIngressService()
    decision = await svc.dispatch(_msg(text="帮我做一份汇报 PPT"))
    assert decision.intent is IngressIntent.START_TASK
    assert decision.reply_text is None


@pytest.mark.asyncio
async def test_dispatch_ignore_passthrough() -> None:
    svc = PilotIngressService()
    decision = await svc.dispatch(_msg(text="   "))
    assert decision.intent is IngressIntent.IGNORE
    assert decision.reply_text is None
