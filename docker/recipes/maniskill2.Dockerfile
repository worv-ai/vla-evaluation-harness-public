# ManiSkill2 evaluation environment
# Assets (YCB ~25MB, EGAD ~320MB, pick_clutter ~16MB) are embedded in the image.

ARG BASE_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base:latest
ARG RUNTIME_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base-render:latest
ARG PREV_IMAGE=ghcr.io/allenai/vla-evaluation-harness/maniskill2:0.9.0
FROM ${PREV_IMAGE} AS prev

FROM ${BASE_IMAGE} AS builder

# ── uv environment ──────────────────────────────────────────────────
ARG PYTHON_VERSION=3.11.13
RUN uv python install "${PYTHON_VERSION}" \
    && uv venv --seed --python "${PYTHON_VERSION}" /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ── Software Vulkan for `--render cpu` ────────────────────────────────
# SAPIEN 2.2.2 needs VK_KHR_external_semaphore_fd, available in lavapipe >= 24.3.
# Keep it isolated so other benchmarks continue using the system Mesa.
RUN install-native-env lavapipe \
    && mkdir -p /opt/lavapipe \
    && cp /opt/pixi/.pixi/envs/lavapipe/share/vulkan/icd.d/lvp_icd.x86_64.json /opt/lavapipe/lvp_icd.json

# ── PyTorch with CUDA 12.1 ────────────────────────────────────────────
ARG TORCH_BACKEND=cu121
ENV UV_TORCH_BACKEND=${TORCH_BACKEND}

RUN uv pip install --no-cache-dir "torch==2.1.0" \
        --torch-backend "${TORCH_BACKEND}"

# ── ManiSkill2 + dependencies (gymnasium, sapien) ─────────────────────
# sapien imports pkg_resources, removed from conda-forge setuptools >=70.
RUN uv pip install --no-cache-dir "setuptools<70" \
    && uv pip install --no-cache-dir mani-skill2

# ── Download evaluation assets (YCB, EGAD, pick_clutter) ────────────
# Embedded in image so users don't need manual download + volume mount.
WORKDIR /workspace
# Their host (storage1.ucsd.edu) goes down at times; then reuse the previous image's copy.
# The asset lock rejects either source if its content differs.
RUN --mount=from=prev,source=/workspace/data,target=/tmp/prev-data \
    { python -m mani_skill2.utils.download_asset -y PickSingleYCB-v0 \
      && python -m mani_skill2.utils.download_asset -y PickSingleEGAD-v0 \
      && python -m mani_skill2.utils.download_asset -y PickClutterYCB-v0; } \
    || { rm -rf data && cp -a /tmp/prev-data data; }
