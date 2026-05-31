"""Agent-Hub runtime 层。

runtime 是「策略执行引擎」，包含：
- policy/ ── 源绑定解析器与风险策略
- observability/ ── 追踪与链路上报
- agent/ ── Agent Actor 基类与协议
- workflow/ ── DAG 工作流执行原语（Phase 4 补充）

导入约束：
- runtime 只可向下依赖 contracts，不得依赖 pilot / connectors / apps
- core.pipeline / core.binding / core.risk 将逐步成为 re-export 包装
"""
