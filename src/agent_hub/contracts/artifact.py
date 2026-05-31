"""跨层产物引用契约。

定义：产物引用（ArtifactRef）和产物轻量描述（Artifact）。
这些类型是 core Pipeline、API 输出层与 Pilot artifact store 之间传递产物事实的公共格式。

注意：Pilot 领域内的完整 Artifact 业务实体（含状态机、版本、审批关联）
保留在 pilot.domain，本模块只定义跨层引用与轻量描述。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ArtifactRef(BaseModel):
    """产物的轻量引用（只含 ID 和类型，用于跨层传递）。"""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    type: str
    title: str


class Artifact(BaseModel):
    """轻量产物描述，供 core Pipeline 和 API 输出层使用。

    Pilot 领域内的完整产物实体保留在 pilot.domain.models；
    本模型只用于 core 层统一输出和跨层传递。
    """

    model_config = ConfigDict()

    artifact_id: str = Field(default_factory=lambda: f"art_{uuid.uuid4().hex[:16]}")
    type: str
    title: str
    status: str = "generated"
    uri: str | None = None
    storage_key: str | None = None
    mime_type: str | None = None
    checksum: str | None = None
    share_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


__all__ = [
    "Artifact",
    "ArtifactRef",
]
