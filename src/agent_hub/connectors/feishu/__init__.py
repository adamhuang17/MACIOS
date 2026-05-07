"""Feishu / Lark connector（M4）。

只承担与飞书开放平台之间的边界：

- :mod:`auth`     —— tenant_access_token 缓存与刷新；
- :mod:`client`   —— im/drive/docs HTTP 客户端 + 单元测试用 Fake；
- :mod:`webhook`  —— URL verification、加密解包、event_id 去重、消息规范化；
- :mod:`models`   —— 严格内部 DTO，不直接外泄飞书原始 payload；
- :mod:`service`  —— 把规范化消息转成 :class:`TaskRequest` 提交给 ``TaskOrchestrator``。
- :mod:`longconn` —— WebSocket 长连接客户端（无公网地址时替代 webhook）。

connector 不直接读写 Pilot 实体或事件存储；所有业务副作用通过
``TaskOrchestrator`` / ``SkillRegistry`` 落地。
"""

from agent_hub.connectors.feishu.auth import FeishuAuthError, FeishuAuthProvider
from agent_hub.connectors.feishu.client import (
    FakeFeishuClient,
    FeishuApiError,
    FeishuClient,
    FeishuClientProtocol,
    FeishuDocCreated,
    FeishuFileUploaded,
    FeishuMessageRecord,
    FeishuMessageSent,
)
from agent_hub.connectors.feishu.longconn import FeishuLongConnClient
from agent_hub.connectors.feishu.models import (
    FeishuAttachment,
    FeishuChatType,
    FeishuInboundMessage,
    FeishuMessageType,
    FeishuWebhookOutcome,
)
from agent_hub.connectors.feishu.progress_notifier import FeishuProgressNotifier
from agent_hub.connectors.feishu.service import (
    FeishuAckResult,
    FeishuWebhookService,
)
from agent_hub.connectors.feishu.webhook import (
    FeishuWebhookError,
    FeishuWebhookProcessor,
    FeishuWebhookResult,
)

__all__ = [
    "FakeFeishuClient",
    "FeishuAckResult",
    "FeishuLongConnClient",
    "FeishuApiError",
    "FeishuAttachment",
    "FeishuAuthError",
    "FeishuAuthProvider",
    "FeishuChatType",
    "FeishuClient",
    "FeishuClientProtocol",
    "FeishuDocCreated",
    "FeishuFileUploaded",
    "FeishuInboundMessage",
    "FeishuMessageRecord",
    "FeishuMessageSent",
    "FeishuMessageType",
    "FeishuProgressNotifier",
    "FeishuWebhookError",
    "FeishuWebhookOutcome",
    "FeishuWebhookProcessor",
    "FeishuWebhookResult",
    "FeishuWebhookService",
]
