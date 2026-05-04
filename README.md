# Agent-Hub

> 面向多用户场景的多 Agent 协同任务系统，覆盖意图路由、DAG 编排、ReAct 推理、RAG 检索、记忆管理、安全防护与 SSE 流式响应。
>
> 当前仓库已经具备可运行的通用 Agent 执行内核；Agent-Pilot、飞书、审批流、PPT 产物链路等内容属于产品方向或在建规划，现统一收敛并明确边界。

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 项目定位

Agent-Hub 可以分成两个层次理解：

1. 当前主线是通用多 Agent 引擎，核心入口为 `IntentRouter + AgentPipeline + LLMAgent / RetrievalAgent / ToolAgent / ReflectionAgent + Memory + Guard`。
2. Agent-Pilot 是这个引擎之上的业务产品方向，目标是把 IM 入口、审批、技能调用、文档/PPT 产物和事件流串起来。

## 当前状态

| 范围 | 状态 | 说明 |
|------|------|------|
| Agent-Hub 核心引擎 | 已实现 | 多 Agent 编排、路由、工具调用、记忆、安全、API 已落地 |
| FastAPI + SSE | 已实现 | 提供 `/chat`、`/chat/stream`、`/health`、`/tools`、`/trace/{trace_id}` |
| RAG 与评测 | 已实现 | 混合检索、评测脚本与测试用例已在仓库中 |
| Pilot M0-M2 基础闭环 | 已实现基础 | 领域模型、事件存储、编排骨架、Skill Registry、Pilot API 与最小 Dashboard 已落地 |
| 飞书连接器、Brief/PPT 产物链路 | 规划中 | 飞书真实接入、文档生成、PPTX 渲染和分享链路仍属于后续模块 |

## 已实现能力

| 能力 | 说明 |
|------|------|
| 结构化意图路由 | 使用 LLM + Pydantic Schema 完成意图分类与子任务拆解 |
| DAG 并行编排 | `AgentPipeline` 按依赖分层，使用 `asyncio.gather` 并行执行同层子任务 |
| ReAct 推理闭环 | `LLMAgent` 支持 Thought -> Action -> Observation 多轮推理 |
| 混合检索 RAG | 支持稠密检索、BM25 稀疏检索与 RRF 融合排序 |
| 三层记忆 | 会话记忆、Obsidian Markdown 持久记忆、向量记忆模块 |
| Prompt 注入防御 | 规则检测 + LLM 语义检测的双层防护 |
| 流式输出 | FastAPI + SSE，支持逐 token 推送 |
| 可观测性 | 支持 trace id、结构化日志与可选 OpenTelemetry |
| 评测与基准 | 提供评测脚本、数据集构建与 benchmark 脚本 |

## 实现边界

当前代码库的真实边界如下：

- 主编排仍然是自研 DAG + `asyncio.gather`，不是 LangGraph 驱动全系统。
- LangGraph 目前主要用于 `reflection_agent` 内部的检索、评估与重写子流程。
- `connectors/` 仍是占位目录，钉钉、QQ、OpenClaw 等外部接入尚未真正落地。
- `pilot/` 已有领域模型、事件底座、编排服务、技能注册抽象和 FakeSkills；飞书适配层、Brief/PPT 真实产物链路仍未落地。
- 仓库内存在若干基础设施模块，但并非全部已经贯穿主 Pipeline，例如缓存、限流、向量记忆接入点仍需进一步整合。

## 架构概览

```mermaid
graph TD
    API[FastAPI API] --> Guard[PromptGuard]
    Guard --> Pipeline[AgentPipeline]

    Pipeline --> Router[IntentRouter]
    Pipeline --> LLM[LLMAgent]
    Pipeline --> RAG[RetrievalAgent]
    Pipeline --> Tool[ToolAgent]
    Pipeline --> Reflect[ReflectionAgent]

    LLM --> Session[SessionMemory]
    LLM --> Persistent[PersistentMemory]
    LLM --> Vector[VectorMemory]

    RAG --> Chunker[Chunker]
    RAG --> Sparse[BM25 Retriever]
    RAG --> Store[pgvector Store]
    RAG --> Ranker[Hybrid Ranker]

    Tool --> Registry[ToolRegistry]
```

## 快速开始

### 1. 安装依赖

```bash
pip install -e .
pip install -e ".[dev]"
```

### 2. 配置环境变量

在项目根目录创建 `.env`：

```env
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
LLM_MODEL=glm-4-flash

PG_DSN=postgresql://agent:agent@localhost:5432/agent_hub
REDIS_URL=redis://localhost:6379/0
OBSIDIAN_VAULT_PATH=./data/obsidian

RAG_API_BASE=http://localhost:8000
CORS_ORIGINS=http://localhost:8080,http://localhost:3000

PUBLIC_BASE_URL=http://localhost:8080
DASHBOARD_STATIC_DIR=web
PILOT_ENABLED=true
PILOT_DEMO_MODE=true
PILOT_STORE_PATH=./data/pilot.sqlite3
PILOT_ADMIN_TOKEN=change-me

GUARD_ENABLED=true
GUARD_LLM_ENABLED=true
```

说明：

- `PG_DSN`、`REDIS_URL`、`OBSIDIAN_VAULT_PATH` 是本地可运行所需的关键配置。
- `PUBLIC_BASE_URL` 是对外展示入口；本机演示可用 `http://localhost:8080`，局域网演示改成 `http://<部署机IP>:8080`，飞书回调阶段应换成 HTTPS 公网域名。
- `DASHBOARD_STATIC_DIR` 默认指向 `web`；当前保底方案由 FastAPI 同源挂载到 `/dashboard/`，不需要单独前端服务器。
- `PILOT_STORE_PATH` 配置后使用 SQLite 保存 Pilot 事件和快照；为空时使用进程内内存存储，重启后数据会丢失。
- `dingtalk_*`、`qq_*`、`openclaw_*` 等配置项虽然已经出现在 `Settings` 中，但对应连接器暂未完成，不建议写进最小启动配置。

### 3. 启动依赖服务

推荐本地开发时先启动 PostgreSQL 和 Redis：

```bash
docker compose up -d postgres redis
```

如果你已经准备好 `.env`，也可以直接启动完整容器栈：

```bash
docker compose up -d --build
```

### 4. 启动 API

```bash
uvicorn agent_hub.api.routes:app --reload --port 8080
```

### 5. 打开 Pilot Dashboard

M2 的 Dashboard 采用同源保底部署：后端启动后访问：

```text
http://localhost:8080/dashboard/
```

局域网演示时，把 `.env` 中的 `PUBLIC_BASE_URL` 改成部署机 IP，例如：

```env
PUBLIC_BASE_URL=http://192.168.1.23:8080
CORS_ORIGINS=http://192.168.1.23:8080
```

容器模式下，`Dockerfile` 会把 `web/` 复制进镜像，`docker-compose.yml` 默认将 Dashboard 静态目录设为 `/app/web`，访问路径仍是 `/dashboard/`。

Dashboard 是 Agent-Pilot 的状态面板和审批兜底入口：任务列表、步骤状态、审批按钮、工件列表和 SSE 事件流都在这里展示。正式飞书比赛展示时，飞书 IM/卡片是主交互入口；Dashboard 用来让评委看到 Agent 内部计划、执行轨迹、审批状态和失败恢复能力。

## API 示例

### 健康检查

```bash
curl http://localhost:8080/health
```

### 同步对话

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "帮我写一个快速排序",
    "user_id": "u001",
    "role": "user"
  }'
```

### SSE 流式对话

```bash
curl -N -X POST http://localhost:8080/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "给我一个 Python 装饰器示例",
    "user_id": "u001"
  }'
```

### 查看当前角色可用工具

```bash
curl "http://localhost:8080/tools?role=user"
```

### 查询 trace

```bash
curl http://localhost:8080/trace/<trace_id>
```

## 测试与脚本

```bash
pytest tests/ -v --ignore=tests/test_manual.py
pytest tests/test_guard.py -v
python scripts/benchmark.py
python scripts/evaluate.py
```

仓库中的 `scripts/` 还提供缓存、安全、RAG 与评测数据构建相关脚本，可按需单独运行。

## 项目结构

```text
src/agent_hub/
├── agents/         # LLM / Retrieval / Tool / Reflection Agent
├── api/            # FastAPI 路由与 SSE 流式输出
├── config/         # Settings 与配置管理
├── connectors/     # 外部连接器占位
├── core/           # Pipeline、Router、Models、Tracer 等核心框架
├── eval/           # 评测数据集、评估器、报告
├── memory/         # Session / Persistent / Vector Memory
├── pilot/          # Pilot 领域模型与事件底座（在建）
├── rag/            # Chunker、Embedder、Ranker、Vector Store
└── security/       # Prompt Guard 与安全防护

scripts/            # benchmark、评测、演示脚本
tests/              # API、Pipeline、Memory、RAG、安全、Pilot 等测试
data/               # Obsidian 记忆与测试数据
```

## Agent-Pilot 方向（规划中）

当前 README 中原有的 Agent-Pilot 目标仍然保留，但统一视为产品路线，不再与已实现能力混写。规划方向包括：

- 以飞书 IM 作为主要任务入口，Web/PWA 作为状态面板和辅助控制台。
- 显式建模 `Workspace`、`Task`、`Plan`、`PlanStep`、`Approval`、`Artifact`、`ExecutionEvent`。
- 在执行链路中引入审批、dry-run、审计和技能注册机制。
- 将文档、Brief、SlideSpec、PPTX 等产物纳入标准化产物流水线。
- 通过事件总线和事件存储支持实时状态同步与断点恢复。

如果后续继续推进 Pilot，建议在代码结构上补齐以下目录，而不是继续把规划直接写进现状描述：

```text
src/agent_hub/pilot/
├── services/
├── skills/
├── storage/
└── model_gateway.py

src/agent_hub/connectors/
└── feishu/
```

## 安全与权限

- Prompt 注入防御采用规则检测与 LLM 语义检测双层组合。
- `ADMIN_COMMAND` 会按用户角色做权限控制，普通用户会被自动降级处理。
- `ToolAgent` 会在执行前校验工具权限，避免高风险工具被普通角色直接调用。
- 会话数据按 `user_id`、`session_id`、`group_id` 进行隔离。

## 许可

MIT