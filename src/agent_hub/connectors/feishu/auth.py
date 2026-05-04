"""Feishu tenant_access_token 缓存与刷新。

约束：

- 只缓存 ``tenant_access_token``；不区分 user_access_token / app_access_token；
- 单进程内通过 :class:`asyncio.Lock` 保证同一时刻只有一次刷新；
- 提前 ``REFRESH_AHEAD_SECONDS`` 刷新，避免边界过期被开放平台拒绝；
- 对开放平台错误进行脱敏：``app_secret`` 不会出现在异常或日志中。
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import httpx
import structlog

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


REFRESH_AHEAD_SECONDS = 60
_DEFAULT_TIMEOUT_S = 10.0
_TOKEN_PATH = "/open-apis/auth/v3/tenant_access_token/internal"


class FeishuAuthError(RuntimeError):
    """tenant_access_token 获取失败。"""


TokenFetcher = "Callable[[str, str], Awaitable[tuple[str, int]]]"


class FeishuAuthProvider:
    """tenant_access_token 缓存提供者。

    Args:
        app_id: 飞书应用 App ID。
        app_secret: App Secret；只在网络请求中使用，不写入日志。
        api_base_url: 开放平台 base url；指向飞书国内 / 国际版。
        request_timeout_ms: 单次刷新请求超时；与全局 client 超时分离。
        token_fetcher: 自定义 fetcher（仅供测试）；签名为
            ``async (app_id, app_secret) -> (token, expire_seconds)``。
    """

    def __init__(
        self,
        *,
        app_id: str,
        app_secret: str,
        api_base_url: str = "https://open.feishu.cn",
        request_timeout_ms: int = 10_000,
        token_fetcher: object | None = None,
    ) -> None:
        if not app_id or not app_secret:
            msg = "app_id 与 app_secret 必须同时提供"
            raise ValueError(msg)
        self._app_id = app_id
        self._app_secret = app_secret
        self._api_base_url = api_base_url.rstrip("/")
        self._timeout_s = request_timeout_ms / 1000.0
        self._token_fetcher = token_fetcher
        self._lock = asyncio.Lock()
        self._token: str | None = None
        self._expires_at: float = 0.0

    @property
    def app_id(self) -> str:
        return self._app_id

    async def get_token(self) -> str:
        """获取 tenant_access_token；缓存命中则不发起远程请求。"""
        now = time.monotonic()
        if self._token is not None and now < self._expires_at:
            return self._token

        async with self._lock:
            now = time.monotonic()
            if self._token is not None and now < self._expires_at:
                return self._token

            token, expire_in = await self._fetch_token()
            self._token = token
            # expire_in 是开放平台返回的剩余秒数；提前 REFRESH_AHEAD_SECONDS 失效。
            self._expires_at = now + max(expire_in - REFRESH_AHEAD_SECONDS, 30.0)
            logger.info(
                "feishu.auth.token_refreshed",
                app_id=self._app_id,
                expire_in=expire_in,
            )
            return token

    async def invalidate(self) -> None:
        """显式作废缓存，下次 :meth:`get_token` 强制刷新。"""
        async with self._lock:
            self._token = None
            self._expires_at = 0.0

    # ── 内部 ───────────────────────────────────────

    async def _fetch_token(self) -> tuple[str, int]:
        if self._token_fetcher is not None:
            return await self._token_fetcher(self._app_id, self._app_secret)  # type: ignore[misc]

        url = f"{self._api_base_url}{_TOKEN_PATH}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.post(
                    url,
                    json={"app_id": self._app_id, "app_secret": self._app_secret},
                )
        except httpx.HTTPError as exc:
            msg = f"feishu auth http error: {exc.__class__.__name__}"
            raise FeishuAuthError(msg) from exc

        if resp.status_code != 200:
            msg = f"feishu auth http {resp.status_code}"
            raise FeishuAuthError(msg)
        data = resp.json()
        code = data.get("code")
        if code != 0:
            # 不要把 secret 或 raw response 全量塞进异常。
            msg = f"feishu auth biz error: code={code}"
            raise FeishuAuthError(msg)
        token = data.get("tenant_access_token")
        expire = int(data.get("expire", 0))
        if not token or expire <= 0:
            msg = "feishu auth response missing token / expire"
            raise FeishuAuthError(msg)
        return token, expire


__all__ = ["FeishuAuthError", "FeishuAuthProvider"]
