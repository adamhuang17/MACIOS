"""``/api/feishu/webhook`` 路由。

只承担 HTTP 边界：

- 解析 JSON、捕捉 :class:`FeishuWebhookError` 并按状态码返回；
- 把 :class:`FeishuAckResult` 翻译成统一响应格式；
- challenge / duplicate / ignored 全部返回 200 + JSON，符合飞书事件
  订阅要求；token / 解密失败返回 403。

业务在 :class:`FeishuWebhookService` 中完成。
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request, status

from agent_hub.connectors.feishu import (
    FeishuAckResult,
    FeishuWebhookError,
    FeishuWebhookOutcome,
    FeishuWebhookService,
)

logger = structlog.get_logger(__name__)


def build_feishu_router(
    *,
    webhook_service: FeishuWebhookService,
    callback_path: str = "/api/feishu/webhook",
) -> APIRouter:
    """构造飞书 webhook FastAPI 路由。"""
    router = APIRouter(tags=["feishu"])

    @router.post(callback_path)
    async def feishu_webhook(request: Request) -> dict:
        try:
            payload = await request.json()
        except (ValueError, UnicodeDecodeError) as exc:
            logger.warning("feishu.webhook.bad_json", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="invalid json",
            ) from exc

        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="payload must be json object",
            )

        try:
            result: FeishuAckResult = await webhook_service.handle(payload)
        except FeishuWebhookError as exc:
            logger.warning("feishu.webhook.rejected", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(exc),
            ) from exc

        return _serialize_result(result)

    return router


def _serialize_result(result: FeishuAckResult) -> dict:
    if result.outcome is FeishuWebhookOutcome.CHALLENGE:
        return {"challenge": result.challenge or ""}
    body: dict = {"outcome": result.outcome.value}
    if result.reason:
        body["reason"] = result.reason
    if result.handle is not None:
        body["task"] = {
            "task_id": result.handle.task_id,
            "workspace_id": result.handle.workspace_id,
            "status": result.handle.status.value,
            "deduplicated": result.handle.deduplicated,
        }
    if result.ack_message_id:
        body["ack_message_id"] = result.ack_message_id
    return body


__all__ = ["build_feishu_router"]
