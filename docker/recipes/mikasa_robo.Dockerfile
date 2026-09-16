# MIKASA-Robo evaluation environment
#
# ManiSkill3/SAPIEN-based memory-intensive manipulation benchmark.
# 32 tasks across categories: remember, shell-game, rotate, intercept, etc.

ARG BASE_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base:latest
ARG RUNTIME_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base-render:latest
FROM ${BASE_IMAGE} AS builder

ARG PYTHON_VERSION=3.10.18
RUN uv python install "${PYTHON_VERSION}" \
    && uv venv --seed --python "${PYTHON_VERSION}" /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG TORCH_BACKEND=cu121
ENV UV_TORCH_BACKEND=${TORCH_BACKEND}

RUN uv pip install --no-cache-dir torch==2.2.1 torchvision==0.17.1 \
        --torch-backend "${TORCH_BACKEND}"

RUN uv pip install --no-cache-dir --prerelease=allow mani_skill==3.0.0b15 gymnasium==0.29.1

# MIKASA-Robo's setup.py imports pkg_resources, which was removed from
# setuptools >=70 on conda-forge. Pin setuptools <70 to restore it.
# CognitiveAISystems/MIKASA-Robo v1.0.0
COPY --from=sources /opt/MIKASA-Robo /opt/MIKASA-Robo
RUN uv pip install --no-cache-dir "setuptools<70" wheel \
    && mkdir -p /opt/MIKASA-Robo \
    && cd /opt/MIKASA-Robo \
    && uv pip install --no-cache-dir --no-build-isolation . \
    && rm -rf /opt/MIKASA-Robo/.git
