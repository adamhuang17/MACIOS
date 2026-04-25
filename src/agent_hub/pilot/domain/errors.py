"""Pilot 领域 / 存储异常。"""

from __future__ import annotations


class PilotDomainError(Exception):
    """Pilot 领域基础异常。"""


class IllegalTransition(PilotDomainError):  # noqa: N818
    """状态机拒绝的非法迁移。"""

    def __init__(
        self,
        entity_type: str,
        entity_id: str,
        current_status: str,
        action: str,
    ) -> None:
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.current_status = current_status
        self.action = action
        super().__init__(
            f"非法状态迁移: {entity_type}({entity_id}) "
            f"status={current_status} action={action}"
        )


class PlanGraphError(PilotDomainError):
    """Plan DAG 校验失败。"""


class ConcurrencyConflict(PilotDomainError):  # noqa: N818
    """乐观并发版本冲突。"""

    def __init__(
        self,
        entity_type: str,
        entity_id: str,
        expected_version: int,
        actual_version: int,
    ) -> None:
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"版本冲突: {entity_type}({entity_id}) "
            f"expected={expected_version} actual={actual_version}"
        )
