# RoboCasa365 multi-task kitchen manipulation evaluation environment
#
# Both projects are pinned past their latest tag: the fixes this benchmark
# needs shipped without a new semantic version, so each install is checked
# against the version the pinned revision declares.

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

ARG ROBOSUITE_VERSION=1.5.2
COPY --from=sources /app/robosuite /app/robosuite
RUN mkdir -p /app/robosuite \
    && cd /app/robosuite \
    &&      uv pip install --no-cache-dir -e . \
    && test "$(python -c 'from importlib.metadata import version; print(version("robosuite"))')" = "${ROBOSUITE_VERSION}" \
    && rm -rf /app/robosuite/.git

ARG ROBOCASA365_VERSION=1.0.1
COPY --from=sources /app/robocasa365 /app/robocasa365
RUN mkdir -p /app/robocasa365 \
    && cd /app/robocasa365 \
    &&      uv pip install --no-cache-dir -e . \
    && test "$(python -c 'from importlib.metadata import version; print(version("robocasa"))')" = "${ROBOCASA365_VERSION}" \
    && rm -rf /app/robocasa365/.git

# Setup macros (non-interactive) and download kitchen assets (~10 GB)

RUN cd /app/robocasa365 \
    && python robocasa/scripts/setup_macros.py \
    && echo "y" | python robocasa/scripts/download_kitchen_assets.py
