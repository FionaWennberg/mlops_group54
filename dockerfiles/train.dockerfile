FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p models reports/figures data/processed


COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY LICENSE LICENSE
COPY src/ src/
COPY configs/ configs/
COPY data/ data/
COPY .dvc/ .dvc/
COPY .dvcignore .dvcignore
COPY data/processed.dvc data/processed.dvc


ENV PYTHONUNBUFFERED=1


RUN uv sync --locked --no-cache

ENTRYPOINT ["bash", "-lc", "/app/.venv/bin/python -m dvc config core.no_scm True && /app/.venv/bin/python -m dvc pull data/processed && /app/.venv/bin/python -m mlops_group54_project.train"]
