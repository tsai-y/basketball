FROM python:3.13-slim

ARG CODEX_VERSION=0.160.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROJECT_ROOT=/repo \
    BOT_DATA_DIR=/data \
    SOURCES_CONFIG=/app/config/project_sources.yaml \
    CODEX_HOME=/data/codex-home

RUN apt-get update \
    && apt-get install --yes --no-install-recommends nodejs npm ca-certificates bubblewrap \
    && npm install --global "@openai/codex@${CODEX_VERSION}" \
    && npm cache clean --force \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system bot && useradd --system --gid bot --home /app bot
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --requirement requirements.txt

COPY project_assistant_bot ./project_assistant_bot
COPY config ./config
RUN mkdir -p /data/codex-home && chown -R bot:bot /app /data

USER bot
VOLUME ["/data"]
CMD ["python", "-m", "project_assistant_bot"]
