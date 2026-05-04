"""飞书 WebSocket 长连接客户端 — 无需公网地址。

原理：
  ``lark-oapi`` SDK 主动向飞书 WebSocket 服务器（wss://sdkws.larkoffice.com）
  发起出站连接；飞书通过该连接推送事件，无需公网 webhook URL。

典型用法（在 PilotRuntime lifespan 中）::

    lc = FeishuLongConnClient(
        app_id=settings.feishu_app_id,
        app_secret=settings.feishu_app_secret,
        webhook_service=runtime.feishu_webhook_service,
    )
    await lc.start()   # 非阻塞，后台线程持续运行
    ...
    await lc.aclose()  # 优雅停止

前提：
  - 飞书开放平台 → 事件订阅 → 接收方式改为「长连接」。
  - 安装依赖：``pip install lark-oapi``（或在 pyproject.toml 声明）。

注意：
  ``lark-oapi.ws`` 模块在 import 时会调用 ``asyncio.get_event_loop()``
  并缓存为模块级变量 ``loop``。如果主线程已有运行中的事件循环（如 uvicorn），
  后续 ``ws.Client.start()`` 调用 ``loop.run_until_complete()`` 会报
  "This event loop is already running"。

  因此必须在**独立线程**中 import 并运行 SDK，使其拿到线程专属的
  新事件循环。
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from agent_hub.connectors.feishu.service import FeishuWebhookService

logger = structlog.get_logger(__name__)


# ── SDK 事件 → webhook dict 转换 ─────────────────────


def _to_webhook_dict(data: Any) -> dict[str, Any] | None:  # noqa: ANN401
    """把 lark-oapi ``im.message.receive_v1`` 事件对象转成 webhook v2 格式 dict。

    长连接接收到的结构化 SDK 对象与 HTTP webhook payload 等价，
    转换后直接交给 :meth:`FeishuWebhookProcessor.handle_event_dict` 处理。
    """
    header = getattr(data, "header", None)
    event = getattr(data, "event", None)
    if header is None or event is None:
        return None

    sender = getattr(event, "sender", None) or object()
    sender_id_obj = getattr(sender, "sender_id", None) or object()
    message = getattr(event, "message", None) or object()

    mentions: list[dict[str, Any]] = []
    for m in getattr(message, "mentions", None) or []:
        mid = getattr(m, "id", None)
        mentions.append(
            {
                "key": getattr(m, "key", ""),
                "id": {"open_id": getattr(mid, "open_id", "") if mid else ""},
                "name": getattr(m, "name", ""),
            }
        )

    return {
        "header": {
            "event_id": getattr(header, "event_id", ""),
            "event_type": getattr(header, "event_type", "im.message.receive_v1"),
            "tenant_key": getattr(header, "tenant_key", ""),
            "app_id": getattr(header, "app_id", ""),
            "token": "",
        },
        "event": {
            "sender": {
                "sender_id": {
                    "open_id": getattr(sender_id_obj, "open_id", ""),
                    "user_id": getattr(sender_id_obj, "user_id", ""),
                    "union_id": getattr(sender_id_obj, "union_id", ""),
                },
                "sender_type": getattr(sender, "sender_type", "user"),
            },
            "message": {
                "message_id": getattr(message, "message_id", ""),
                "chat_id": getattr(message, "chat_id", ""),
                "chat_type": getattr(message, "chat_type", ""),
                "message_type": getattr(message, "message_type", ""),
                "content": getattr(message, "content", "{}"),
                "mentions": mentions,
            },
        },
    }


# ── 长连接客户端 ──────────────────────────────────────


class FeishuLongConnClient:
    """飞书 WebSocket 长连接客户端（不需要公网地址）。

    通过 ``lark-oapi`` SDK 主动连接飞书 WebSocket 服务器，接收事件后
    通过 :class:`~agent_hub.connectors.feishu.service.FeishuWebhookService`
    处理，与 HTTP webhook 共用同一套去重 / 规范化 / 过滤 / 任务提交逻辑。
    """

    def __init__(
        self,
        *,
        app_id: str,
        app_secret: str,
        webhook_service: FeishuWebhookService,
    ) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._service = webhook_service
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._ws_client: Any = None

    # ── 公开接口 ─────────────────────────────────────

    async def start(self) -> None:
        """在后台启动 WebSocket 长连接（立即返回）。"""
        try:
            import lark_oapi  # noqa: F401 — 仅检查是否安装
        except ImportError:
            logger.error(
                "feishu.longconn.missing_dependency",
                hint="pip install 'lark-oapi>=1.3,<2'",
            )
            return

        self._stop_event.clear()
        main_loop = asyncio.get_running_loop()
        self._thread = threading.Thread(
            target=self._run_in_thread,
            args=(main_loop,),
            daemon=True,
            name="feishu-longconn",
        )
        self._thread.start()
        logger.info("feishu.longconn.started", app_id=self._app_id)

    async def aclose(self) -> None:
        """优雅停止长连接。"""
        self._stop_event.set()
        ws = self._ws_client
        if ws is not None:
            _stop = getattr(ws, "stop", None) or getattr(ws, "close", None)
            if callable(_stop):
                with suppress(Exception):
                    _stop()
            self._ws_client = None

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("feishu.longconn.stopped")

    # ── 内部 ─────────────────────────────────────────

    def _run_in_thread(self, main_loop: asyncio.AbstractEventLoop) -> None:
        """在独立线程中 import lark SDK 并运行 WebSocket 客户端。

        lark-oapi 的 ws/client.py 模块级代码在 import 时调用
        ``asyncio.get_event_loop()`` 并缓存为模块变量 ``loop``。
        此处为工作线程创建独立事件循环，并替换 SDK 缓存的 loop，
        使 ``ws.Client.start()`` 不再与 uvicorn 主循环冲突。
        """
        import lark_oapi as lark
        from lark_oapi.ws import client as _ws_mod

        thread_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(thread_loop)
        _ws_mod.loop = thread_loop

        def _handle_im(data: Any) -> None:  # noqa: ANN401
            payload = _to_webhook_dict(data)
            if payload is None:
                logger.warning("feishu.longconn.unparseable_event")
                return
            future = asyncio.run_coroutine_threadsafe(
                self._service.handle_event_dict(payload),
                main_loop,
            )
            try:
                future.result(timeout=30)
            except Exception as exc:  # noqa: BLE001
                logger.warning("feishu.longconn.event_error", error=str(exc))

        event_handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(_handle_im)
            .build()
        )
        ws_client = lark.ws.Client(
            self._app_id,
            self._app_secret,
            event_handler=event_handler,
        )
        self._ws_client = ws_client
        try:
            ws_client.start()
        except Exception as exc:  # noqa: BLE001
            if not self._stop_event.is_set():
                logger.error("feishu.longconn.client_error", error=str(exc))
        finally:
            self._ws_client = None
