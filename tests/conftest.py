"""Pytest 全局 fixtures / 配置。"""

from __future__ import annotations

import os

# 禁止测试期间连接 OTLP collector（localhost:4317）
os.environ.setdefault("OTEL_SDK_DISABLED", "1")

if os.environ.get("AGENT_HUB_ALLOW_REAL_FEISHU_IN_TESTS") != "true":
    os.environ["FEISHU_ENABLED"] = "false"
    os.environ["FEISHU_USE_LONG_CONN"] = "false"
    os.environ["PILOT_USE_REAL_CHAIN"] = "false"
    os.environ["PILOT_USE_REAL_GATEWAY"] = "false"
