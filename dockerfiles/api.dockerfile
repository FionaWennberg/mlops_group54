FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim


RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git ca-certificates&& \
    apt clean && rm -rf /var/lib/apt/lists/*

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

COPY src src/

RUN uv sync --locked --no-cache

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["bash", "-lc", "/app/.venv/bin/python -m dvc config core.no_scm True && /app/.venv/bin/python -m dvc pull models/model.pth && /app/.venv/bin/python -m dvc pull data/processed && /app/.venv/bin/python -m mlops_group54_project.api"]
