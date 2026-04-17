FROM python:3.12-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir -e .

# Default env
ENV OBSIDIAN_VAULT_PATH=/app/data/obsidian
ENV CORS_ORIGINS=*

EXPOSE 8080

CMD ["uvicorn", "agent_hub.api.routes:app", "--host", "0.0.0.0", "--port", "8080"]
