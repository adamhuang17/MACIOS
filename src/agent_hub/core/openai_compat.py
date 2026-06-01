"""Compatibility helpers for OpenAI-compatible providers."""

from __future__ import annotations

from typing import Any


def provider_extra_body(base_url: str | None, model: str | None) -> dict[str, Any]:
    """Return provider-specific request options for chat completions."""
    url = (base_url or "").lower()
    model_name = (model or "").lower()
    if "api.deepseek.com" in url and model_name.startswith("deepseek-v4"):
        return {"thinking": {"type": "disabled"}}
    return {}
