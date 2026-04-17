"""应用配置管理。

通过 ``pydantic-settings`` 从 ``.env`` 文件和环境变量中读取配置。
优先级：环境变量 > .env 文件 > 默认值。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（pyproject.toml 所在位置）
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """全局配置。

    所有字段均可通过同名环境变量覆盖（不区分大小写）。

    Attributes:
        anthropic_api_key: Anthropic API 密钥。
        anthropic_model: 默认使用的 Claude 模型。
        dingtalk_app_key: 钉钉应用 App Key。
        dingtalk_app_secret: 钉钉应用 App Secret。
        dingtalk_robot_code: 钉钉机器人编码。
        qq_bot_app_id: QQ 机器人 App ID。
        qq_bot_token: QQ 机器人 Token。
        qq_bot_secret: QQ 机器人 Secret。
        neo4j_uri: Neo4j 数据库连接 URI。
        neo4j_user: Neo4j 用户名。
        neo4j_password: Neo4j 密码。
        openclaw_host: OpenClaw 虚拟机主机地址。
        openclaw_port: OpenClaw 服务端口。
        openclaw_api_key: OpenClaw API 密钥。
        vm_shared_dir: 虚拟机共享目录路径。
        host_data_dir: 宿主机数据目录路径。
        allowed_file_extensions: 允许上传的文件扩展名。
        log_level: 日志级别。
    """

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM（智谱 / OpenAI 兼容） ─────────────────────
    llm_api_key: str = ""
    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    llm_model: str = "glm-4-flash"

    # ── 钉钉 ─────────────────────────────────────────
    dingtalk_app_key: str = ""
    dingtalk_app_secret: str = ""
    dingtalk_robot_code: str = ""

    # ── QQ 机器人 ─────────────────────────────────────
    qq_bot_app_id: str = ""
    qq_bot_token: str = ""
    qq_bot_secret: str = ""

    # ── Neo4j ─────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # ── OpenClaw ──────────────────────────────────────
    openclaw_host: str = "127.0.0.1"
    openclaw_port: int = 8443
    openclaw_api_key: str = ""

    # ── RAG ────────────────────────────────────────────
    rag_api_base: str = "http://localhost:8000"

    # ── pgvector + Embedding ─────────────────────────
    pg_dsn: str = "postgresql://localhost:5432/agent_hub"
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    chunk_size: int = 512
    chunk_overlap: int = 64
    rag_top_k: int = 5
    rrf_k: int = 60

    # ── Obsidian 记忆持久化 ────────────────────────────
    obsidian_vault_path: str = "./data/obsidian"

    # ── 记忆管理（Mem0 式） ───────────────────────────
    memory_conflict_threshold: float = 0.85

    # ── CORS（逗号分隔） ──────────────────────────────
    cors_origins: str = "http://localhost:3000"

    # ── ReAct 推理循环 ────────────────────────────────
    react_max_rounds: int = 5

    # ── Prompt 注入防御 ───────────────────────────────
    guard_enabled: bool = True
    guard_llm_enabled: bool = True

    # ── Redis 缓存 ───────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl: int = 300
    semantic_cache_threshold: float = 0.95

    # ── LLM 并发控制 ─────────────────────────────────
    llm_max_concurrent: int = 5
    llm_call_timeout: int = 30

    # ── 文件路径 & 安全 ──────────────────────────────
    vm_shared_dir: str = "/mnt/shared"
    host_data_dir: str = "D:/Agent-Hub/data"
    allowed_file_extensions: str = ".txt,.pdf,.md,.docx,.csv,.json"

    # ── 日志 ──────────────────────────────────────────
    log_level: str = "INFO"

    @property
    def allowed_extensions_set(self) -> set[str]:
        """将逗号分隔的扩展名字符串解析为集合。"""
        return {ext.strip() for ext in self.allowed_file_extensions.split(",") if ext.strip()}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置单例。

    使用 ``lru_cache`` 确保整个进程生命周期内只解析一次配置。
    """
    return Settings()
