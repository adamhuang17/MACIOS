"""Agent-Pilot 业务产品层。

M0 阶段只暴露领域模型、状态机、事件契约与事件底座。
不依赖 LLM、Feishu、FastAPI、`agent_hub.core.AgentPipeline`。
"""

from __future__ import annotations

from agent_hub.pilot.domain.enums import (
    ActorType,
    ApprovalAction,
    ApprovalStatus,
    ApprovalTargetType,
    ArtifactAction,
    ArtifactStatus,
    ArtifactType,
    EventLevel,
    EventType,
    PlanAction,
    PlanStatus,
    PlanStepAction,
    PlanStepKind,
    PlanStepStatus,
    RiskLevel,
    TaskAction,
    TaskStatus,
    WorkspaceAction,
    WorkspaceStatus,
)
from agent_hub.pilot.domain.errors import (
    ConcurrencyConflict,
    IllegalTransition,
    PilotDomainError,
    PlanGraphError,
)
from agent_hub.pilot.domain.events import ExecutionEvent
from agent_hub.pilot.domain.models import (
    Approval,
    Artifact,
    Plan,
    PlanStep,
    Task,
    Workspace,
)
from agent_hub.pilot.domain.state import transition, validate_plan_graph
from agent_hub.pilot.events.bus import EventBus, EventSubscription
from agent_hub.pilot.events.store import (
    EventStore,
    InMemoryEventStore,
    SnapshotStore,
    SQLiteEventStore,
)

__all__ = [
    # enums
    "ActorType",
    "ApprovalAction",
    "ApprovalStatus",
    "ApprovalTargetType",
    "ArtifactAction",
    "ArtifactStatus",
    "ArtifactType",
    "EventLevel",
    "EventType",
    "PlanAction",
    "PlanStatus",
    "PlanStepAction",
    "PlanStepKind",
    "PlanStepStatus",
    "RiskLevel",
    "TaskAction",
    "TaskStatus",
    "WorkspaceAction",
    "WorkspaceStatus",
    # errors
    "ConcurrencyConflict",
    "IllegalTransition",
    "PilotDomainError",
    "PlanGraphError",
    # models
    "Approval",
    "Artifact",
    "ExecutionEvent",
    "Plan",
    "PlanStep",
    "Task",
    "Workspace",
    # state
    "transition",
    "validate_plan_graph",
    # events / store
    "EventBus",
    "EventStore",
    "EventSubscription",
    "InMemoryEventStore",
    "SnapshotStore",
    "SQLiteEventStore",
]
