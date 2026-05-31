"""跨层公共错误类型。

只定义跨边界传递的错误基类和通用错误码，
各子系统的具体异常保留在各自模块内。
"""

from __future__ import annotations


class AgentHubError(Exception):
    """Agent-Hub 所有受控异常的基类。"""

    code: str = "agent_hub_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        self.message = message

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


class ContractViolationError(AgentHubError):
    """调用方传入了违反契约的参数或状态。"""

    code = "contract_violation"


class NotFoundError(AgentHubError):
    """请求的资源不存在。"""

    code = "not_found"


class PermissionDeniedError(AgentHubError):
    """调用方权限不足。"""

    code = "permission_denied"


class ExternalServiceError(AgentHubError):
    """外部服务（LLM、存储、连接器）调用失败。"""

    code = "external_service_error"


__all__ = [
    "AgentHubError",
    "ContractViolationError",
    "ExternalServiceError",
    "NotFoundError",
    "PermissionDeniedError",
]
