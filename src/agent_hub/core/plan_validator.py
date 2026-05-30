"""Validation helpers for router-produced SubTask DAGs."""

from __future__ import annotations

from collections import defaultdict, deque

from pydantic import BaseModel, Field

from agent_hub.core.models import SubTask
from agent_hub.core.risk import ToolProfile, parse_tool_profile


class PlanValidationResult(BaseModel):
    """Result of validating a SubTask plan."""

    valid: bool
    errors: list[str] = Field(default_factory=list)


class SubTaskPlanValidator:
    """Validate SubTask count, dependencies, cycles, and profile compatibility."""

    def __init__(
        self,
        *,
        max_subtasks: int = 8,
        known_agents: set[str] | None = None,
    ) -> None:
        self._max_subtasks = max_subtasks
        self._known_agents = known_agents

    def validate(
        self,
        subtasks: list[SubTask],
        *,
        tool_profile: ToolProfile | str | None = None,
    ) -> PlanValidationResult:
        errors: list[str] = []
        if len(subtasks) > self._max_subtasks:
            errors.append(f"子任务数量超过上限: {len(subtasks)} > {self._max_subtasks}")

        ids = [task.subtask_id for task in subtasks]
        id_set = set(ids)
        if len(id_set) != len(ids):
            errors.append("subtask_id 必须唯一")

        for task in subtasks:
            if self._known_agents is not None:
                unknown = [
                    agent
                    for agent in task.required_agents
                    if agent not in self._known_agents
                ]
                if unknown:
                    errors.append(f"{task.subtask_id} 引用了未知 Agent: {unknown}")
            for dep in task.depends_on:
                if dep == task.subtask_id:
                    errors.append(f"{task.subtask_id} 不能依赖自身")
                elif dep not in id_set:
                    errors.append(f"{task.subtask_id} 依赖未知子任务: {dep}")

        if not errors and _has_cycle(subtasks):
            errors.append("子任务依赖图存在环")

        profile = parse_tool_profile(tool_profile)
        if profile is ToolProfile.READ_ONLY:
            for task in subtasks:
                if "tool_agent" in task.required_agents:
                    errors.append(f"{task.subtask_id} 需要 tool_agent，但当前 profile=read_only")

        return PlanValidationResult(valid=not errors, errors=errors)


def validate_subtask_plan(
    subtasks: list[SubTask],
    *,
    max_subtasks: int = 8,
    tool_profile: ToolProfile | str | None = None,
) -> PlanValidationResult:
    """Convenience wrapper for validating a SubTask DAG."""
    return SubTaskPlanValidator(max_subtasks=max_subtasks).validate(
        subtasks,
        tool_profile=tool_profile,
    )


def _has_cycle(subtasks: list[SubTask]) -> bool:
    if not subtasks:
        return False

    in_degree = {task.subtask_id: 0 for task in subtasks}
    children: dict[str, list[str]] = defaultdict(list)
    ids = set(in_degree)
    for task in subtasks:
        for dep in task.depends_on:
            if dep in ids:
                in_degree[task.subtask_id] += 1
                children[dep].append(task.subtask_id)

    ready = deque(task_id for task_id, degree in in_degree.items() if degree == 0)
    visited = 0
    while ready:
        task_id = ready.popleft()
        visited += 1
        for child in children[task_id]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                ready.append(child)
    return visited != len(subtasks)


__all__ = [
    "PlanValidationResult",
    "SubTaskPlanValidator",
    "validate_subtask_plan",
]
