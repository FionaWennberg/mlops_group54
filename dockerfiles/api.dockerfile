FROM ghcr.io/astral-sh/uv:python3.11-alpine

RUN apk add --no-cache \
    bash \
    git \
    build-base \
    gcc \
    musl-dev

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY LICENSE LICENSE
COPY README.md README.md
COPY .dvc/ .dvc/
COPY .dvcignore .dvcignore
COPY configs/ configs/
COPY models.dvc models.dvc
COPY data/processed.dvc data/processed.dvc


RUN uv sync --frozen --no-install-project

COPY src src/

RUN uv sync --frozen

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["bash", "-lc", "/app/.venv/bin/python -m dvc config core.no_scm True && /app/.venv/bin/python -m dvc pull models/model.pth && /app/.venv/bin/python -m dvc pull data/processed && /app/.venv/bin/python -m mlops_group54_project.api"]
