"""Pytest 全局 fixtures / 配置。"""

from __future__ import annotations

import os

# 禁止测试期间连接 OTLP collector（localhost:4317）
os.environ.setdefault("OTEL_SDK_DISABLED", "1")
