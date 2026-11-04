# Single image for Cloud Run, Azure Container Apps, ECS/Fargate or a laptop.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY config ./config
COPY src ./src
RUN pip install ".[all]" \
 && useradd --system --uid 10001 --home /app app \
 && mkdir -p /data && chown app /data

USER app
ENV CONVO_HOST=0.0.0.0 \
    CONVO_MODELS_FILE=/app/config/models.yaml \
    CONVO_SQLITE_PATH=/data/conversations.db
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8080\")}/healthz', timeout=2)"

CMD ["convo", "serve"]
