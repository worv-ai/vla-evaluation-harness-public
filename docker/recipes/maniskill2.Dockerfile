# ManiSkill2 evaluation environment
# Assets (YCB ~25MB, EGAD ~320MB, pick_clutter ~16MB) are embedded in the image.

ARG BASE_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base:latest
ARG RUNTIME_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base-render:latest
ARG MANISKILL_ASSET_IMAGE=scratch
FROM ${MANISKILL_ASSET_IMAGE} AS asset_source

FROM ${BASE_IMAGE} AS builder

# ── uv environment ──────────────────────────────────────────────────
ARG PYTHON_VERSION=3.10.18
RUN uv python install "${PYTHON_VERSION}" \
    && uv venv --seed --python "${PYTHON_VERSION}" /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

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
# Optional immutable asset image avoids depending on UCSD server availability.
ARG MANISKILL_ASSET_IMAGE
RUN --mount=from=asset_source,target=/asset-source \
    if [ "${MANISKILL_ASSET_IMAGE}" != scratch ]; then \
        cp -a /asset-source/workspace/data /workspace/data \
        && test -d data/mani_skill2_ycb && test -d data/mani_skill2_egad && test -d data/pick_clutter \
        && printf 'ManiSkill2 assets: %s\n' "${MANISKILL_ASSET_IMAGE}" >> /usr/local/share/vla-build/sources.txt; \
    else \
        python -m mani_skill2.utils.download_asset -y PickSingleYCB-v0 \
        && python -m mani_skill2.utils.download_asset -y PickSingleEGAD-v0 \
        && python -m mani_skill2.utils.download_asset -y PickClutterYCB-v0; \
    fi
