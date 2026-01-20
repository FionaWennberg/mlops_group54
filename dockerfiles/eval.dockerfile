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

ENV PYTHONUNBUFFERED=1

RUN uv sync --locked --no-cache --no-install-project

# Pull required artifacts and run evaluation
# We pull:
#  - data/processed (tensors)
#  - models/model.pth (checkpoint)  <-- only works if it's tracked by DVC
#
# Then run evaluate with Hydra overrides to ensure it uses the downloaded paths.
ENTRYPOINT ["bash", "-lc", "dvc pull data/processed && dvc pull models/model.pth && uv run python -m mlops_group54_project.evaluate data.processed_dir=data/processed eval.checkpoint_path=models/model.pth"]
