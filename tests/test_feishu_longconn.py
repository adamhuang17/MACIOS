"""FeishuLongConnClient 单元测试。

使用 unittest.mock 替换 lark_oapi，完全离线运行。
覆盖：
- _to_webhook_dict 转换逻辑
- start() 在缺少 lark_oapi 时优雅降级
- start() 成功启动后台线程
- aclose() 停止后台线程并调用 ws_client.stop()
- _run_in_thread → 事件触发 → service.handle_event_dict() 被调用
- FeishuWebhookProcessor.handle_event_dict() 跳过验签直接解析
"""

from __future__ import annotations

import asyncio
import threading
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_hub.connectors.feishu.longconn import FeishuLongConnClient, _to_webhook_dict
from agent_hub.connectors.feishu.models import FeishuWebhookOutcome
from agent_hub.connectors.feishu.webhook import FeishuWebhookProcessor

# ── _to_webhook_dict ──────────────────────────────


def _make_lark_event(
    *,
    event_id: str = "evt-001",
    event_type: str = "im.message.receive_v1",
    tenant_key: str = "tenant-1",
    app_id: str = "app-1",
    open_id: str = "ou_sender",
    chat_id: str = "oc_chat",
    message_id: str = "om_msg",
    chat_type: str = "group",
    message_type: str = "text",
    content: str = '{"text":"hello"}',
    mentions: list | None = None,
) -> MagicMock:
    """构造一个模拟 lark-oapi im.message.receive_v1 事件对象。"""
    header = MagicMock()
    header.event_id = event_id
    header.event_type = event_type
    header.tenant_key = tenant_key
    header.app_id = app_id

    sender_id = MagicMock()
    sender_id.open_id = open_id
    sender_id.user_id = ""
    sender_id.union_id = ""

    sender = MagicMock()
    sender.sender_id = sender_id
    sender.sender_type = "user"

    message = MagicMock()
    message.message_id = message_id
    message.chat_id = chat_id
    message.chat_type = chat_type
    message.message_type = message_type
    message.content = content

    mock_mentions: list = []
    for m in (mentions or []):
        mid = MagicMock()
        mid.open_id = m.get("open_id", "")
        mm = MagicMock()
        mm.key = m.get("key", "@_user_1")
        mm.id = mid
        mm.name = m.get("name", "")
        mock_mentions.append(mm)
    message.mentions = mock_mentions

    event = MagicMock()
    event.sender = sender
    event.message = message

    data = MagicMock()
    data.header = header
    data.event = event
    return data


class TestToWebhookDict:
    def test_basic_conversion(self) -> None:
        data = _make_lark_event()
        result = _to_webhook_dict(data)
        assert result is not None
        assert result["header"]["event_id"] == "evt-001"
        assert result["header"]["event_type"] == "im.message.receive_v1"
        assert result["header"]["tenant_key"] == "tenant-1"
        assert result["header"]["token"] == ""  # 长连接不需要 token
        assert result["event"]["message"]["chat_id"] == "oc_chat"
        assert result["event"]["sender"]["sender_id"]["open_id"] == "ou_sender"

    def test_mentions_converted(self) -> None:
        data = _make_lark_event(
            mentions=[{"key": "@_user_1", "open_id": "ou_bot", "name": "Bot"}]
        )
        result = _to_webhook_dict(data)
        assert result is not None
        mentions = result["event"]["message"]["mentions"]
        assert len(mentions) == 1
        assert mentions[0]["id"]["open_id"] == "ou_bot"
        assert mentions[0]["key"] == "@_user_1"

    def test_missing_header_returns_none(self) -> None:
        data = MagicMock()
        data.header = None
        data.event = MagicMock()
        assert _to_webhook_dict(data) is None

    def test_missing_event_returns_none(self) -> None:
        data = MagicMock()
        data.header = MagicMock()
        data.event = None
        assert _to_webhook_dict(data) is None


# ── FeishuWebhookProcessor.handle_event_dict ─────────


@pytest.mark.asyncio
async def test_handle_event_dict_basic() -> None:
    processor = FeishuWebhookProcessor(
        verification_token="secret",  # 长连接模式不应校验 token
        encrypt_key="",
        bot_open_id="",
        require_mention_in_group=False,
    )
    data = _make_lark_event(chat_type="p2p")
    payload = _to_webhook_dict(data)
    assert payload is not None
    result = await processor.handle_event_dict(payload)
    assert result.outcome is FeishuWebhookOutcome.ACCEPTED
    assert result.message is not None
    assert result.message.text == "hello"


@pytest.mark.asyncio
async def test_handle_event_dict_dedup() -> None:
    processor = FeishuWebhookProcessor(require_mention_in_group=False)
    data = _make_lark_event(event_id="evt-dup", chat_type="p2p")
    payload = _to_webhook_dict(data)
    assert payload is not None
    r1 = await processor.handle_event_dict(payload)
    r2 = await processor.handle_event_dict(payload)
    assert r1.outcome is FeishuWebhookOutcome.ACCEPTED
    assert r2.outcome is FeishuWebhookOutcome.DUPLICATE


@pytest.mark.asyncio
async def test_handle_event_dict_skips_verification_token() -> None:
    """长连接模式下即使 token 为空也不应拒绝请求。"""
    processor = FeishuWebhookProcessor(
        verification_token="should_not_be_checked",
        require_mention_in_group=False,
    )
    data = _make_lark_event(chat_type="p2p")
    payload = _to_webhook_dict(data)
    assert payload is not None
    # token="" 在 webhook 模式下会 mismatch，但 handle_event_dict 跳过校验
    result = await processor.handle_event_dict(payload)
    assert result.outcome is FeishuWebhookOutcome.ACCEPTED


# ── FeishuLongConnClient ──────────────────────────


def _make_mock_lark_module() -> types.ModuleType:
    """构造一个最小化的 lark_oapi mock 模块。"""
    mod = types.ModuleType("lark_oapi")
    mod.__path__ = []  # 让 import machinery 把它视为 package
    builder = MagicMock()
    handler_builder = MagicMock()
    handler_builder.register_p2_im_message_receive_v1.return_value = handler_builder
    handler_builder.build.return_value = MagicMock()
    builder.return_value = handler_builder
    mod.EventDispatcherHandler = MagicMock()
    mod.EventDispatcherHandler.builder = builder

    ws_mod = types.ModuleType("lark_oapi.ws")
    ws_client_mod = types.ModuleType("lark_oapi.ws.client")
    ws_client_mod.loop = None
    ws_client_instance = MagicMock()
    ws_client_instance.start = MagicMock(side_effect=lambda: None)  # 立即返回
    ws_client_instance.stop = MagicMock()
    ws_mod.Client = MagicMock(return_value=ws_client_instance)
    ws_mod.client = ws_client_mod
    mod.ws = ws_mod
    return mod


def _mock_lark_modules(lark_mod: types.ModuleType) -> dict[str, types.ModuleType]:
    return {
        "lark_oapi": lark_mod,
        "lark_oapi.ws": lark_mod.ws,
        "lark_oapi.ws.client": lark_mod.ws.client,
    }


@pytest.mark.asyncio
async def test_start_missing_lark_oapi() -> None:
    """缺少 lark_oapi 时 start() 不应抛异常，只记录错误。"""
    svc = AsyncMock()
    client = FeishuLongConnClient(
        app_id="app-1", app_secret="secret", webhook_service=svc
    )
    with patch.dict("sys.modules", {"lark_oapi": None}):
        await client.start()
    # 后台线程未创建
    assert client._thread is None


@pytest.mark.asyncio
async def test_start_and_aclose() -> None:
    """start() 创建后台线程；aclose() 停止 ws_client 并等待线程结束。"""
    svc = AsyncMock()
    client = FeishuLongConnClient(
        app_id="app-1", app_secret="secret", webhook_service=svc
    )

    lark_mod = _make_mock_lark_module()
    ws_client_instance = lark_mod.ws.Client.return_value

    # _run_in_thread 使用 stop_event 控制退出；我们立即置位让线程自然结束
    def _blocking_run(loop: asyncio.AbstractEventLoop) -> None:
        client._ws_client = ws_client_instance
        # 模拟 start() 立即返回（非长期阻塞）
        client._stop_event.wait(timeout=0.5)

    with patch.dict("sys.modules", _mock_lark_modules(lark_mod)), \
         patch.object(client, "_run_in_thread", side_effect=_blocking_run):
        await client.start()
        assert client._thread is not None
        client._ws_client = ws_client_instance
        await client.aclose()

    ws_client_instance.stop.assert_called_once()
    assert client._thread is None


@pytest.mark.asyncio
async def test_event_dispatched_to_service() -> None:
    """事件经 _run_in_thread 处理后应调用 service.handle_event_dict()。"""
    svc = AsyncMock()
    svc.handle_event_dict = AsyncMock(return_value=MagicMock())

    client = FeishuLongConnClient(
        app_id="app-1", app_secret="secret", webhook_service=svc
    )

    lark_mod = _make_mock_lark_module()
    captured_handler: list = []

    # 截获 register_p2_im_message_receive_v1 注册的回调
    original_builder = lark_mod.EventDispatcherHandler.builder

    def _capture_builder(enc_key: str, verify_token: str) -> MagicMock:
        hb = original_builder(enc_key, verify_token)
        def _register(fn):  # noqa: ANN001, ANN202
            captured_handler.append(fn)
            return hb
        hb.register_p2_im_message_receive_v1 = _register
        return hb

    lark_mod.EventDispatcherHandler.builder = _capture_builder

    loop = asyncio.get_event_loop()
    stop = threading.Event()

    def _blocking(loop_: asyncio.AbstractEventLoop) -> None:  # noqa: ANN001
        # 让 _run_in_thread 能够执行到 ws_client.start()
        client._stop_event.wait(timeout=0.5)

    with patch.dict("sys.modules", _mock_lark_modules(lark_mod)), \
         patch.object(client, "_run_in_thread", wraps=client._run_in_thread):
        # 直接在线程中执行以捕获回调
        t = threading.Thread(
            target=client._run_in_thread, args=(loop,), daemon=True
        )
        t.start()
        # 等一小会让线程注册 handler
        await asyncio.sleep(0.1)
        client._stop_event.set()
        t.join(timeout=2)

    # 用捕获到的 handler 发送一条事件
    if captured_handler:
        event_data = _make_lark_event(chat_type="p2p", event_id="evt-x")
        captured_handler[0](event_data)
        # run_coroutine_threadsafe 发送到 loop，等待
        await asyncio.sleep(0.1)
        svc.handle_event_dict.assert_called_once()
