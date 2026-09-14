# RoboCasa kitchen manipulation evaluation environment
#
# Pinned to the original RoboCasa release. The successor release is a
# separate, API-incompatible integration — see Dockerfile.robocasa365.

ARG BASE_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base:latest
ARG RUNTIME_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base-render:latest
FROM ${BASE_IMAGE} AS builder

# Python 3.10: RoboCasa v0.2 pins numba==0.56.4, whose llvmlite has no
# cp311 wheel.
ARG PYTHON_VERSION=3.10.18
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

COPY --from=sources /app/robocasa /app/robocasa
RUN mkdir -p /app/robocasa &&      cd /app/robocasa &&      uv pip install --no-cache-dir -e . &&      rm -rf /app/robocasa/.git

# Setup macros (non-interactive) and download kitchen assets (~10 GB)

RUN cd /app/robocasa \
    && python robocasa/scripts/setup_macros.py \
    && echo "y" | python robocasa/scripts/download_kitchen_assets.py

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src/ src/
ARG HARNESS_VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${HARNESS_VERSION}
# robocasa/__init__.py asserts numpy is 1.23.{2,3,5}, which the harness floor of
# >=1.24 would pull out from under it. Restore RoboCasa's pin after installing;
# the harness test suite passes on 1.23.3.
ARG ROBOCASA_NUMPY=1.23.3

ARG TORCH_BACKEND=cpu
ENV UV_TORCH_BACKEND=${TORCH_BACKEND}

RUN uv pip install --no-cache-dir -e . && \
    uv pip install --no-cache-dir "numpy==${ROBOCASA_NUMPY}"
COPY configs/ configs/
