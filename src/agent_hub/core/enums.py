"""意图类型、用户角色等枚举定义。"""

from enum import Enum


class IntentType(str, Enum):
    """用户意图分类。

    由路由层（Router Agent）对用户消息进行分类后输出。
    """

    TASK_GENERATION = "task_generation"    # 任务生成
    RETRIEVAL = "retrieval"               # 知识检索
    TOOL_EXECUTION = "tool_execution"     # 工具调用
    FILE_PROCESSING = "file_processing"   # 文件处理
    ADMIN_COMMAND = "admin_command"        # 管理员命令
    GROUP_CHAT = "group_chat"             # 群聊（不处理）
    REFLECTION = "reflection"             # 反思 / 纠错


class UserRole(str, Enum):
    """用户角色。

    ADMIN 拥有单独开会话操控远端 OpenClaw 的权限；
    USER 仅可在群聊中发起标准请求。
    """

    ADMIN = "admin"
    USER = "user"
