"""执行模式、节点类型、渠道类型、用户角色等枚举定义。"""

from enum import Enum


class ExecutionMode(str, Enum):
    """路由决策执行模式（Layer 2 输出）。

    代表"下一步如何处理"，而非用户语义意图分类。
    由 DecisionRouter 对用户请求进行决策后输出。
    """

    IGNORE = "ignore"       # 忽略 / 轻量回复（群聊闲聊、无需处理）
    QA = "qa"               # 知识问答 / 检索
    PLAN = "plan"           # 内容生成 / 复杂多步骤任务
    ACT = "act"             # 工具执行
    DELEGATE = "delegate"   # 委派给外部 Agent（如 OpenClaw）
    REPAIR = "repair"       # 反思 / 修复（用户质疑之前结果）


class NodeType(str, Enum):
    """工作流节点类型（基础设施节点，对标 Dify BlockEnum）。

    稳定的能力类型枚举，不是用户语义 intent taxonomy。
    各 Flow 内部通过局部分支标签做动态路由，
    此枚举仅描述"节点能做什么"。
    """

    LLM = "llm"                                      # LLM 生成
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"      # 知识库检索
    QUESTION_CLASSIFIER = "question_classifier"      # 动态问题分类节点
    IF_ELSE = "if_else"                              # 条件分支
    TOOL = "tool"                                    # 工具调用
    LOOP = "loop"                                    # 循环节点


class ChannelType(str, Enum):
    """消息来源渠道类型。

    Layer 1 Policy Gate 中的事实字段之一。
    """

    DINGTALK = "dingtalk"   # 钉钉
    QQ = "qq"               # QQ
    OPENCLAW = "openclaw"   # OpenClaw
    API = "api"             # 直接 API 调用
    WEBHOOK = "webhook"     # Webhook 回调


class UserRole(str, Enum):
    """用户角色。

    ADMIN 拥有单独开会话操控远端 OpenClaw 的权限；
    USER 仅可在群聊中发起标准请求。
    """

    ADMIN = "admin"
    USER = "user"
