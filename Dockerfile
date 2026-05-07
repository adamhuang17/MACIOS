FROM python:3.12-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY pyproject.toml README.md ./
COPY src/ src/
COPY web/ web/
RUN pip install --no-cache-dir -e .

# Runtime data dirs
RUN mkdir -p /app/data/obsidian /app/data/pilot/artifacts

# Default env
ENV OBSIDIAN_VAULT_PATH=/app/data/obsidian
ENV CORS_ORIGINS=http://localhost:8080,http://127.0.0.1:8080
ENV DASHBOARD_STATIC_DIR=/app/web
ENV PUBLIC_BASE_URL=http://localhost:8080
ENV PILOT_ENABLED=true
ENV PILOT_DEMO_MODE=true
ENV PILOT_STORE_PATH=/app/data/pilot/pilot.sqlite3
ENV PILOT_ARTIFACT_DIR=/app/data/pilot/artifacts
ENV PILOT_USE_REAL_GATEWAY=false
ENV PILOT_USE_REAL_CHAIN=false
ENV FEISHU_ENABLED=false
ENV FEISHU_USE_LONG_CONN=false

EXPOSE 8080

CMD ["uvicorn", "agent_hub.api.routes:app", "--host", "0.0.0.0", "--port", "8080"]
