FROM ghcr.io/astral-sh/uv:python3.11-alpine

RUN apk add --no-cache \
    bash \
    git \
    build-base \
    gcc \
    musl-dev

WORKDIR /app

RUN mkdir -p models data/processed data/monitoring

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md

RUN uv sync --frozen --no-install-project

COPY src src/

RUN uv sync --frozen

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["bash", "-lc", "dvc pull models/model.pth && uv run uvicorn src.mlops_group54_project.api:app --host 0.0.0.0 --port 8000"]

