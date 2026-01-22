FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim


RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git ca-certificates&& \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY LICENSE LICENSE
COPY src src/
COPY .dvc/ .dvc/
COPY .dvcignore .dvcignore
COPY configs/ configs/
COPY models.dvc models.dvc
COPY data/processed.dvc data/processed.dvc

ENV PYTHONUNBUFFERED=1


RUN uv sync --locked --no-cache


ENTRYPOINT ["bash", "-lc", "/app/.venv/bin/python -m dvc config core.no_scm True && /app/.venv/bin/python -m dvc pull models/model.pth && /app/.venv/bin/python -m dvc pull data/processed && /app/.venv/bin/python -m uvicorn src.mlops_group54_project.api:app --host 0.0.0.0 --port 8080"]
