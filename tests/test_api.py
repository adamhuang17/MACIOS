"""API 端点测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from agent_hub.core.models import TaskOutput

# ── 构造 mock pipeline 以避免真实 LLM 调用 ──────────


def _mock_pipeline_run_output(**overrides) -> TaskOutput:
    defaults = {
        "trace_id": "api-test-trace",
        "user_id": "test-user",
        "response": "Hello from mock pipeline",
        "agent_results": [],
        "memory_saved": [],
        "total_duration_ms": 42,
        "status": "success",
    }
    defaults.update(overrides)
    return TaskOutput(**defaults)


@pytest.fixture()
def client():
    """创建测试客户端，mock 掉 Pipeline 初始化和 run。"""
    # 先 patch AgentPipeline 构造函数，避免真实初始化
    with patch("agent_hub.api.routes.AgentPipeline") as MockPipeline:
        mock_instance = MockPipeline.return_value
        mock_instance.run = AsyncMock(return_value=_mock_pipeline_run_output())
        mock_instance._registry.list_tools_for_user.return_value = []

        from agent_hub.api.routes import app

        with TestClient(app) as c:
            yield c


# ── 测试 ─────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["pipeline"] is True


class TestChatEndpoint:
    def test_chat_success(self, client: TestClient) -> None:
        resp = client.post("/chat", json={
            "message": "你好",
            "user_id": "test-user",
            "role": "user",
            "channel": "test",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "trace_id" in data
        assert "response" in data
        assert data["status"] == "success"

    def test_chat_missing_fields(self, client: TestClient) -> None:
        """缺少必填字段应返回 422。"""
        resp = client.post("/chat", json={"message": "hi"})
        assert resp.status_code == 422

    def test_chat_with_custom_trace_id(self, client: TestClient) -> None:
        resp = client.post("/chat", json={
            "message": "hello",
            "user_id": "u1",
            "trace_id": "custom-trace",
        })
        assert resp.status_code == 200


class TestToolsEndpoint:
    def test_list_tools(self, client: TestClient) -> None:
        resp = client.get("/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert "tools" in data


class TestTraceEndpoint:
    def test_get_trace(self, client: TestClient) -> None:
        resp = client.get("/trace/abc-123")
        assert resp.status_code == 200
        assert resp.json()["trace_id"] == "abc-123"
