# RoboCasa kitchen manipulation evaluation environment
#
# Pinned to the original RoboCasa release. The successor release is a
# separate, API-incompatible integration — see Dockerfile.robocasa365.

ARG BASE_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base:latest
ARG RUNTIME_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base-render:latest
FROM ${BASE_IMAGE} AS builder
ARG TORCH_BACKEND=cpu
ENV UV_TORCH_BACKEND=${TORCH_BACKEND}

ARG PYTHON_VERSION=3.11.13
RUN uv python install "${PYTHON_VERSION}" \
    && uv venv --seed --python "${PYTHON_VERSION}" /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# robosuite v1.5.0 is the release RoboCasa v0.2 was cut against.
# ARISE-Initiative/robosuite v1.5.0
COPY --from=sources /app/robosuite /app/robosuite
RUN mkdir -p /app/robosuite &&      cd /app/robosuite &&      uv pip install --no-cache-dir -e . &&      rm -rf /app/robosuite/.git

# robocasa/robocasa v0.2
# RoboCasa v0.2 pins numba==0.56.4, whose llvmlite has no cp311 wheel; 0.57.1 is the first numba with one.
COPY --from=sources /app/robocasa /app/robocasa
RUN cd /app/robocasa \
    && printf 'numba==0.57.1\n' > /tmp/numba-override.txt \
    && uv pip install --no-cache-dir --override /tmp/numba-override.txt -e . \
    && rm -rf /app/robocasa/.git /tmp/numba-override.txt

# Setup macros (non-interactive) and download kitchen assets (~10 GB)

RUN cd /app/robocasa \
    && python robocasa/scripts/setup_macros.py \
    && echo "y" | python robocasa/scripts/download_kitchen_assets.py
