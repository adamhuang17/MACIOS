"""PilotRepository：在 M0 SnapshotStore 之上的 typed 包装。

服务层只通过本类读写实体快照；版本号默认按 ``entity.version - 1``
作为 ``expected_version`` 传给底层快照存储，从而强制乐观并发。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_hub.pilot.domain.models import (
    Approval,
    Artifact,
    Plan,
    PlanStep,
    Task,
    Workspace,
)

if TYPE_CHECKING:
    from agent_hub.pilot.events.store import SnapshotStore


_ENTITY_TYPE_MAP: dict[type, str] = {
    Workspace: "workspace",
    Task: "task",
    Plan: "plan",
    PlanStep: "plan_step",
    Approval: "approval",
    Artifact: "artifact",
}


class PilotRepository:
    """对快照存储的 typed 封装。"""

    def __init__(self, snapshots: SnapshotStore) -> None:
        self._snapshots = snapshots

    async def save(
        self,
        entity: Workspace | Task | Plan | PlanStep | Approval | Artifact,
        *,
        expected_version: int | None = -1,
    ) -> None:
        """保存实体快照。

        ``expected_version=-1`` 表示按 ``entity.version - 1`` 推断（默认）。
        显式传 ``None`` 表示不做版本校验。
        """
        if expected_version == -1:
            expected_version = entity.version - 1
        await self._snapshots.put_snapshot(entity, expected_version=expected_version)

    async def get_workspace(self, workspace_id: str) -> Workspace | None:
        return await self._snapshots.get_snapshot("workspace", workspace_id)

    async def get_task(self, task_id: str) -> Task | None:
        return await self._snapshots.get_snapshot("task", task_id)

    async def get_plan(self, plan_id: str) -> Plan | None:
        return await self._snapshots.get_snapshot("plan", plan_id)

    async def get_step(self, step_id: str) -> PlanStep | None:
        return await self._snapshots.get_snapshot("plan_step", step_id)

    async def get_approval(self, approval_id: str) -> Approval | None:
        return await self._snapshots.get_snapshot("approval", approval_id)

    async def get_artifact(self, artifact_id: str) -> Artifact | None:
        return await self._snapshots.get_snapshot("artifact", artifact_id)

    async def list_steps(
        self,
        plan_id: str,
        *,
        workspace_id: str | None = None,
    ) -> list[PlanStep]:
        steps = await self._snapshots.list_snapshots(
            "plan_step", workspace_id=workspace_id,
        )
        filtered = [s for s in steps if s.plan_id == plan_id]
        filtered.sort(key=lambda s: s.index)
        return filtered

    async def list_task_artifacts(
        self,
        task_id: str,
        *,
        workspace_id: str | None = None,
    ) -> list[Artifact]:
        items = await self._snapshots.list_snapshots(
            "artifact", workspace_id=workspace_id,
        )
        return [a for a in items if a.task_id == task_id]

    async def list_plans_for_task(
        self,
        task_id: str,
        *,
        workspace_id: str | None = None,
    ) -> list[Plan]:
        items = await self._snapshots.list_snapshots(
            "plan", workspace_id=workspace_id,
        )
        return [p for p in items if p.task_id == task_id]

    async def list_workspaces(self) -> list[Workspace]:
        return await self._snapshots.list_snapshots("workspace")

    async def list_tasks(
        self,
        *,
        workspace_id: str | None = None,
    ) -> list[Task]:
        return await self._snapshots.list_snapshots(
            "task", workspace_id=workspace_id,
        )

    async def list_approvals_for_task(
        self,
        task_id: str,
        *,
        workspace_id: str | None = None,
    ) -> list[Approval]:
        items = await self._snapshots.list_snapshots(
            "approval", workspace_id=workspace_id,
        )
        return [a for a in items if a.task_id == task_id]

    async def list_approvals_for_plan(
        self,
        plan_id: str,
        *,
        workspace_id: str | None = None,
    ) -> list[Approval]:
        items = await self._snapshots.list_snapshots(
            "approval", workspace_id=workspace_id,
        )
        return [a for a in items if a.target_id == plan_id]


__all__ = ["PilotRepository"]
