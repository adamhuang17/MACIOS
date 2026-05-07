"""审批通知协议。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent_hub.pilot.domain.models import Approval


@runtime_checkable
class ApprovalRequestNotifier(Protocol):
    """在审批进入 requested 后向外部渠道发送通知。"""

    async def notify_requested(self, approval: Approval) -> None: ...


__all__ = ["ApprovalRequestNotifier"]