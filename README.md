# MACIOS — Multi-Agent Collaborative Intelligence Orchestration System

> 面向多用户多渠道对话场景的 AI 原生 Multi-Agent 协同编排系统
> 支持意图路由 · DAG 任务编排 · ReAct 推理 · 混合检索 · 三层记忆 · 双层安全防御

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green?logo=fastapi)
![Tests](https://img.shields.io/badge/Tests-330%20passed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 核心特性

| 能力 | 说明 |
|------|------|
| **结构化意图路由** | LLM Function Calling + Pydantic Schema，单次调用完成 7 类意图分类与 DAG 子任务拆解 |
| **DAG 并行编排** | Kahn 拓扑排序 + asyncio.gather，同层子任务并行执行、跨层结果传递 |
| **ReAct 推理闭环** | Thought→Action→Observation 多轮循环，最多 5 轮自主决策，格式容错 |
| **混合检索 RAG** | pgvector 稠密 + BM25 稀疏 + RRF 融合排序，LangGraph 驱动检索-评估-重写 |
| **三层记忆架构** | Session deque (短期) + Obsidian Markdown (持久) + pgvector (语义向量) |
| **双层注入防御** | 22 条正则规则 (~1ms) + LLM Function Calling 语义检测，覆盖 5 大攻击类别 |
| **SSE 流式输出** | FastAPI + SSE 逐 Token 推送，异步非阻塞 |
| **全链路追踪** | OpenTelemetry 可选集成 + structlog JSON Lines 兜底 |
| **评测框架** | Precision@K / Recall@K / MRR 自动评测 |

---

## 架构概览

```mermaid
graph TD
    subgraph 入口层
        API["FastAPI /chat<br/>SSE 流式"]
        DingTalk["钉钉 Connector"]
        QQ["QQ Connector"]
    end

    subgraph 安全层
        Guard["PromptGuard<br/>双层注入防御"]
    end

    subgraph 核心层
        Pipeline["AgentPipeline<br/>DAG 编排主控"]
        Router["IntentRouter<br/>7 类意图路由"]
        Tracer["SpanContext<br/>OTel 全链路追踪"]
        Cache["CacheLayer<br/>L1 精确 + L2 语义"]
    end

    subgraph 执行层
        LLM["LLMAgent<br/>ReAct 推理"]
        RAG["RetrievalAgent<br/>混合检索"]
        Tool["ToolAgent<br/>工具调用"]
        Reflect["ReflectionAgent<br/>LangGraph 反思"]
    end

    subgraph 记忆层
        Session["SessionMemory<br/>滑动窗口"]
        Persistent["PersistentMemory<br/>Obsidian MD"]
        Vector["VectorMemory<br/>pgvector 语义"]
        Conflict["ConflictDetector<br/>冲突检测"]
    end

    subgraph RAG Pipeline
        Chunker["语义切块"]
        Embedder["BGE 向量化"]
        VStore["pgvector 存储"]
        BM25["BM25 稀疏检索"]
        Ranker["RRF 融合排序"]
    end

    subgraph 工具注册
        Registry["ToolRegistry<br/>6 个预置工具"]
    end

    API --> Guard
    DingTalk -.-> Guard
    QQ -.-> Guard
    Guard -->|通过| Pipeline
    Guard -->|拦截| API

    Pipeline --> Router
    Pipeline --> Tracer
    Pipeline --> Cache
    Pipeline --> LLM
    Pipeline --> RAG
    Pipeline --> Tool
    Pipeline --> Reflect

    LLM --> Session
    LLM --> Persistent
    LLM --> Vector
    Vector --> Conflict
    Tool --> Registry

    RAG --> Chunker
    Chunker --> Embedder
    Embedder --> VStore
    RAG --> BM25
    VStore --> Ranker
    BM25 --> Ranker
```

## 数据流

```
用户消息 → TaskInput
    → PromptGuard.check()          # Step 0: 双层注入检测（规则+LLM）
    → IntentRouter.route()         # 7 类意图分类 + DAG 子任务拆解
    → 权限校验                      # ADMIN_COMMAND 需管理员角色
    → Pipeline._topological_layers() # Kahn 拓扑排序
    → asyncio.gather (同层并行)     # LLMAgent / RetrievalAgent / ToolAgent / ReflectionAgent
        → ReAct Loop (可选)         # Thought → Action → Observation × N
        → RAG Pipeline (可选)       # Dense + Sparse → RRF → Rewrite
    → MemoryManager.add()          # 三层记忆写入
    → TaskOutput / SSE Stream      # 最终响应
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| **LLM** | Anthropic Claude · OpenAI GPT · 智谱 GLM-4 |
| **框架** | FastAPI · Pydantic v2 · LangGraph · sentence-transformers |
| **向量** | pgvector · BAAI/BGE-large-zh-v1.5 |
| **检索** | BM25 (rank-bm25 + jieba) · RRF 融合 |
| **存储** | PostgreSQL · Redis · Obsidian Markdown Vault |
| **可观测** | OpenTelemetry · structlog JSON Lines |
| **安全** | 正则规则引擎 · LLM Function Calling 语义检测 |
| **测试** | pytest · pytest-asyncio · 330 用例 |

---

## 快速开始

### 安装

```bash
# 基础安装
pip install -e .

# 开发环境（含测试 + lint）
pip install -e ".[dev]"
```

### 配置

在项目根目录创建 `.env` 文件：

```env
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
LLM_MODEL=glm-4-flash
OBSIDIAN_VAULT_PATH=./data/obsidian
POSTGRES_DSN=postgresql://user:pass@localhost:5432/agent_hub
REDIS_URL=redis://localhost:6379/0
CORS_ORIGINS=http://localhost:3000
GUARD_ENABLED=true
```

### Docker 一键启动

```bash
docker compose up -d
```

### 手动启动 API 服务

```bash
uvicorn agent_hub.api.routes:app --reload --port 8080
```

### API 使用

```bash
# 健康检查
curl http://localhost:8080/health

# 对话
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "帮我写一个快速排序", "user_id": "u001", "role": "user"}'

# 查看可用工具（按角色过滤）
curl http://localhost:8080/tools?role=user

# 查看追踪链路
curl http://localhost:8080/trace/{trace_id}
```

### 运行测试

```bash
# 全量测试 (330 用例)
pytest tests/ -v --ignore=tests/test_manual.py

# 安全测试 (37 注入防御用例)
pytest tests/test_guard.py -v

# 基准测试
python scripts/benchmark.py
```

---

## 项目结构

```
src/agent_hub/
├── api/                        # HTTP 接口层
│   ├── routes.py               #   FastAPI 路由 (/chat /health /tools /trace)
│   └── streaming.py            #   SSE 流式输出
├── agents/                     # Agent 执行层
│   ├── llm_agent.py            #   LLM Agent + ReAct 推理循环
│   ├── reflection_agent.py     #   LangGraph 反思 Agent (检索-评估-重写)
│   ├── registry.py             #   工具注册表 (6 个预置工具)
│   ├── retrieval_agent.py      #   RAG 检索 Agent
│   └── tool_agent.py           #   工具调用 Agent (权限校验)
├── config/
│   └── settings.py             # pydantic-settings 配置管理
├── connectors/                 # 通讯渠道连接器 (DingTalk/QQ/OpenClaw)
├── core/                       # 核心框架
│   ├── base_agent.py           #   Agent 抽象基类 (模板方法)
│   ├── cache.py                #   L1 精确 + L2 语义三级缓存
│   ├── enums.py                #   IntentType (7类) / UserRole 枚举
│   ├── models.py               #   Pydantic 数据模型 (12+ 模型)
│   ├── observability.py        #   OTel Metrics 脚手架
│   ├── pipeline.py             #   AgentPipeline DAG 编排主控
│   ├── rate_limiter.py         #   异步并发限流 + 超时降级
│   ├── router.py               #   IntentRouter 意图路由器
│   └── tracer.py               #   SpanContext 全链路追踪
├── eval/                       # 评测框架
│   ├── dataset.py              #   评测数据集加载
│   ├── evaluator.py            #   Precision@K / Recall@K / MRR
│   └── reporter.py             #   评测报告生成
├── memory/                     # 三层记忆管理
│   ├── conflict_detector.py    #   余弦相似度冲突检测 (Mem0 式)
│   ├── memory_ops.py           #   记忆操作统一接口
│   ├── persistent.py           #   Obsidian Markdown 持久化
│   ├── session.py              #   滑动窗口会话记忆
│   └── vector_memory.py        #   pgvector 语义向量记忆
├── rag/                        # RAG 混合检索
│   ├── chunker.py              #   语义切块 (段落对齐 + 窗口重叠)
│   ├── embedder.py             #   BGE 向量化 (lazy-load)
│   ├── hybrid_ranker.py        #   RRF 融合排序
│   ├── pipeline.py             #   RAGPipeline (ingest + retrieve)
│   ├── query_rewriter.py       #   Query Rewrite (LLM 驱动)
│   ├── relevance_evaluator.py  #   检索相关性评估
│   ├── sparse_retriever.py     #   BM25 + jieba 稀疏检索
│   └── vector_store.py         #   pgvector HNSW 向量存储
└── security/                   # 安全模块
    └── guard.py                #   双层 Prompt 注入防御

scripts/
├── benchmark.py                # 端到端性能基准测试
├── bench_cache.py              # 缓存性能测试
├── bench_guard.py              # 安全防御基准测试
├── bench_rag.py                # RAG 检索基准测试
├── build_eval_dataset.py       # 评测数据集构建
└── evaluate.py                 # 评测脚本

tests/                          # 206+ 测试用例
├── test_agents.py              #   Agent 执行层测试 (41 用例)
├── test_api.py                 #   API 接口测试
├── test_cache.py               #   缓存层测试
├── test_eval.py                #   评测框架测试
├── test_guard.py               #   注入防御测试 (37 用例)
├── test_memory.py              #   记忆管理测试 (36 用例)
├── test_memory_v2.py           #   向量记忆 + 冲突检测测试
├── test_models.py              #   数据模型测试
├── test_pipeline.py            #   Pipeline 编排测试
├── test_pipeline_fallback.py   #   Pipeline 降级测试
├── test_rag.py                 #   RAG 检索测试
├── test_reflection.py          #   反思 Agent 测试
├── test_router.py              #   意图路由测试 (26 用例)
└── test_streaming.py           #   流式输出测试
```

---

## 安全设计

### 双层 Prompt 注入防御

```
用户输入
  ├─→ RuleBasedGuard (22 条正则, ~1ms)
  │     ├── 角色劫持检测
  │     ├── Prompt 泄露检测
  │     ├── 越权提升检测
  │     ├── 分隔符注入检测
  │     └── 编码绕过检测
  └─→ LLMGuard (Function Calling 语义分析)
        └── confidence > 0.8 → blocked
        └── 0.5~0.8 → suspicious (放行+标记)
        └── < 0.5 → safe
```

### 权限控制

- `UserRole.ADMIN` / `UserRole.USER` 角色分离
- `ADMIN_COMMAND` 意图仅管理员可执行，普通用户自动降级
- `ToolAgent` 执行前校验工具权限 (如 `system_command` 仅 admin)
- 会话级数据隔离：每用户独立 session / memory namespace

---

## 许可

MIT
