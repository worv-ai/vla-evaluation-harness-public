# RoboMME evaluation environment (ManiSkill3 fork + SAPIEN, Vulkan rendering)

ARG BASE_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base:latest
ARG RUNTIME_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base-render:latest
FROM ${BASE_IMAGE} AS builder

# ── uv environment (Python 3.11 per RoboMME requirement) ──────────────
ARG PYTHON_VERSION=3.11.13
RUN uv python install "${PYTHON_VERSION}" \
    && uv venv --seed --python "${PYTHON_VERSION}" /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ── PyTorch with CUDA (RoboMME envs use torch for scene randomization) ───
# Match the existing release: upstream pins with official CUDA 12.8 wheels.
ARG TORCH_BACKEND=cu128
ENV UV_TORCH_BACKEND=${TORCH_BACKEND}

RUN uv pip install --no-cache-dir "torch==2.9.1" "torchvision==0.24.1"

# ── setuptools (RoboMME pins 80.9.0) ────────────────────────────────────
RUN uv pip install --no-cache-dir "setuptools==80.9.0"

# ── Custom ManiSkill fork ────────────────────────────────────────────────
WORKDIR /app
COPY --from=sources /app/ManiSkill /app/ManiSkill
RUN mkdir -p /app/ManiSkill      && cd /app/ManiSkill && rm -rf .git

RUN cd /app/ManiSkill && uv pip install --no-cache-dir -e .

# ── RoboMME benchmark ───────────────────────────────────────────────────
COPY --from=sources /app/robomme_benchmark /app/robomme_benchmark
RUN mkdir -p /app/robomme_benchmark      && cd /app/robomme_benchmark && rm -rf .git

RUN cd /app/robomme_benchmark && uv pip install --no-cache-dir -e .

# ── Additional runtime deps (h5py imported by DemonstrationWrapper) ─────
RUN uv pip install --no-cache-dir "opencv-python>=4.11.0.86" h5py imageio
