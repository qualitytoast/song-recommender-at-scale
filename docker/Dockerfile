# One image for every part of the sharded recommender: shard servers (recsys.shard) and the front desk
# (recsys.serve --shards ...); docker-compose.yml picks the command. The bundle (model files and shard
# slices) is mounted at /bundle, not baked in: a new model version doesn't need a new image.
FROM python:3.13-slim

WORKDIR /app
COPY docker/requirements-serve.txt .
# PyTorch's CPU-only build: the containers have no GPU (and on a Mac, Docker can't reach the Mac's).
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-serve.txt

COPY recsys ./recsys
ENV PYTHONUNBUFFERED=1
