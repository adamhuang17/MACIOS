# Agent-Hub 架构词汇表

本文档定义项目中使用的四种核心能力类型，消除命名歧义，统一术语。

## 四种能力类型

### 1. InstructionSkill（指令技能）

**别名**: `SkillPackage`（`agent_hub.capabilities.skills`）  
**规范导入**: `from agent_hub.capabilities.skills import InstructionSkill`

**含义**: 不可执行的提示词包，遵循 Claude Code / OpenClaw 的 `SKILL.md` 约定。  
用于向 LLM 注入结构化指令，教会模型何时及如何使用某个能力。

- 来源：磁盘上的 `SKILL.md` 文件或带 YAML frontmatter 的 `.md` 文件。
- 不可直接调用，不产生副作用。
- 通过 `format_skill_invocation()` 注入 LLM prompt。

---

### 2. ActionRegistry（动作注册表）

**别名**: `SkillRegistry`（`agent_hub.pilot.skills.registry`）  
**规范导入**: `from agent_hub.pilot.skills.registry import ActionRegistry`

**含义**: 可执行的业务动作集合，每个动作有 `side_effect`（read/write/share）、
`requires_approval`、`timeout_ms` 等策略约束。

- 动作由 async Python 函数实现（`SkillFunc`）。
- 写/分享类动作默认需要人工或自动审批（`requires_approval=True`）。
- 通过 `ActionRegistry.invoke(SkillInvocation)` 执行，返回 `SkillResult`。

---

### 3. ToolRegistry（工具注册表）

**模块**: `agent_hub.agents.registry`  
**规范导入**: `from agent_hub.agents.registry import ToolRegistry`

**含义**: Agent 可使用的工具集合，主要对接 MCP（Model Context Protocol）工具。

- 每个工具有 `ToolSpec`（schema）+ 可选的本地执行函数 `func`。
- MCP 工具的 `func=None`，执行在外部 MCP 服务器上完成。
- 通过 `register_tool()` / `list_tools_for_profile()` 管理工具权限。

---

### 4. CapabilityBundle（能力包）

**模块**: `agent_hub.capabilities.plugins`  
**规范导入**: `from agent_hub.capabilities.plugins import PluginBundle`

**含义**: 一整个 Claude Code 插件包（`.claude-plugin/plugin.json`），聚合了
InstructionSkill 列表、MCP 配置（`MCPServerSpec`）、自定义命令等。

- 由 `load_plugin_bundle()` 从磁盘加载。
- 不可直接执行；是 InstructionSkill + MCPConfig 的组合容器。

---

## 术语对照表

| 规范名称 | 旧/别名 | 模块 | 可执行 | 有副作用 |
|---|---|---|---|---|
| `InstructionSkill` | `SkillPackage` | `capabilities.skills` | ❌ | ❌ |
| `ActionRegistry` | `SkillRegistry` | `pilot.skills.registry` | ✅ | ✅ |
| `ToolRegistry` | ─ | `agents.registry` | 条件（MCP） | 条件 |
| `CapabilityBundle` | `PluginBundle` | `capabilities.plugins` | ❌ | ❌ |

---

## 分层边界规则

```
capabilities/                  ← 加载配置、不执行
  InstructionSkill (SkillPackage)
  MCPConfig
  CapabilityBundle (PluginBundle)

agents/                        ← LLM 推理 + MCP 工具调用
  ToolRegistry
  LLMAgent / ToolAgent / ...

pilot/skills/                  ← 可执行业务动作
  ActionRegistry (SkillRegistry)

core/                          ← SubTask DAG 执行内核
  AgentPipeline
  Router / Binding / RiskPolicy
```

**禁止**:
- `capabilities/` 不得 `import subprocess`、不得调用 `pilot`
- `core/` 不得 `import connectors.feishu` 具体实现
- `pilot/domain/` 不得 `import fastapi`、`anthropic`、`openai`
- `connectors/` 不得直接持有 `AgentPipeline`
