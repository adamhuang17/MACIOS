"""``FeishuAuthProvider`` 与 ``FeishuClient`` 的轻量单元测试。

走 :class:`httpx.MockTransport` 注入响应，不发起真实网络请求。
"""

from __future__ import annotations

import json

import httpx
import pytest

from agent_hub.connectors.feishu import FeishuApiError, FeishuAuthProvider, FeishuClient

# ── Auth ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_auth_provider_caches_token() -> None:
    calls: list[tuple[str, str]] = []

    async def fetcher(app_id: str, app_secret: str) -> tuple[str, int]:
        calls.append((app_id, app_secret))
        return f"token-{len(calls)}", 7200

    auth = FeishuAuthProvider(
        app_id="ai",
        app_secret="sec",
        token_fetcher=fetcher,
    )
    a = await auth.get_token()
    b = await auth.get_token()
    assert a == b == "token-1"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_auth_provider_invalidate_forces_refresh() -> None:
    calls: list[int] = []

    async def fetcher(_a: str, _s: str) -> tuple[str, int]:
        calls.append(1)
        return f"t{len(calls)}", 7200

    auth = FeishuAuthProvider(app_id="a", app_secret="s", token_fetcher=fetcher)
    await auth.get_token()
    await auth.invalidate()
    second = await auth.get_token()
    assert second == "t2"
    assert len(calls) == 2


# ── Client ─────────────────────────────────────────


def _build_client(handler) -> tuple[FeishuClient, list[httpx.Request]]:  # noqa: ANN001
    seen: list[httpx.Request] = []

    async def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return await handler(request)

    transport = httpx.MockTransport(wrapped)
    http_client = httpx.AsyncClient(transport=transport)

    async def fetcher(_a: str, _s: str) -> tuple[str, int]:
        return "tok", 7200

    auth = FeishuAuthProvider(app_id="a", app_secret="s", token_fetcher=fetcher)
    client = FeishuClient(auth=auth, http_client=http_client)
    return client, seen


@pytest.mark.asyncio
async def test_client_send_message_success() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "code": 0,
                "msg": "ok",
                "data": {
                    "message_id": "om_x",
                    "chat_id": "oc_x",
                    "create_time": "1700000000000",
                },
            },
        )

    client, seen = _build_client(handler)
    sent = await client.send_message(
        receive_id="oc_x",
        receive_id_type="chat_id",
        msg_type="text",
        content=json.dumps({"text": "hi"}),
    )
    assert sent.message_id == "om_x"
    assert sent.chat_id == "oc_x"
    assert seen[0].headers["Authorization"] == "Bearer tok"
    await client.aclose()


@pytest.mark.asyncio
async def test_client_create_doc_success() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "document": {"document_id": "doc_1", "revision_id": 3},
                },
            },
        )

    client, _ = _build_client(handler)
    created = await client.create_doc(title="hello")
    assert created.document_id == "doc_1"
    assert created.revision_id == 3
    assert created.url.endswith("doc_1")
    await client.aclose()


@pytest.mark.asyncio
async def test_client_business_error_raises() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 99991, "msg": "bad token"})

    client, _ = _build_client(handler)
    with pytest.raises(FeishuApiError) as ctx:
        await client.send_message(
            receive_id="oc",
            receive_id_type="chat_id",
            msg_type="text",
            content="{}",
        )
    assert ctx.value.code == 99991
    await client.aclose()


@pytest.mark.asyncio
async def test_client_unauthorized_invalidates_token() -> None:
    refresh_count = 0

    async def fetcher(_a: str, _s: str) -> tuple[str, int]:
        nonlocal refresh_count
        refresh_count += 1
        return f"tok{refresh_count}", 7200

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    auth = FeishuAuthProvider(app_id="a", app_secret="s", token_fetcher=fetcher)
    client = FeishuClient(auth=auth, http_client=http_client)
    with pytest.raises(FeishuApiError):
        await client.send_message(
            receive_id="oc",
            receive_id_type="chat_id",
            msg_type="text",
            content="{}",
        )
    # invalidate 后再次请求应该再 fetch 一次 token
    with pytest.raises(FeishuApiError):
        await client.send_message(
            receive_id="oc",
            receive_id_type="chat_id",
            msg_type="text",
            content="{}",
        )
    assert refresh_count == 2
    await client.aclose()
