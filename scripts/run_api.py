"""Run the Agent-Hub API with local-platform startup fixes."""

from __future__ import annotations

import asyncio
import os
import sys

import uvicorn


def _ensure_windows_selector_event_loop_policy() -> None:
    if sys.platform != "win32":
        return
    selector_factory = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if selector_factory is None:
        return
    asyncio.set_event_loop_policy(selector_factory())


def main() -> None:
    _ensure_windows_selector_event_loop_policy()
    host = os.getenv("AGENT_HUB_API_HOST", "127.0.0.1")
    port = int(os.getenv("AGENT_HUB_API_PORT", "9000"))
    uvicorn.run(
        "agent_hub.api.routes:app",
        host=host,
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
