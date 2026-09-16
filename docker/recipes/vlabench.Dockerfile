# VLABench evaluation environment (MuJoCo-based)

ARG BASE_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base:latest
ARG RUNTIME_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base-render:latest
FROM ${BASE_IMAGE} AS builder

ENV VLABENCH_ROOT=/app/VLABench/VLABench

ARG PYTHON_VERSION=3.10.18
RUN uv python install "${PYTHON_VERSION}" \
    && uv venv --seed --python "${PYTHON_VERSION}" /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install MuJoCo, dm_control, and numpy first

ARG TORCH_BACKEND=cpu
ENV UV_TORCH_BACKEND=${TORCH_BACKEND}

RUN uv pip install --no-cache-dir \
        mujoco==3.2.2 \
        dm_control==1.0.22 \
        numpy==1.25.0

RUN uv pip install --no-cache-dir \
        gym==0.26.2 \
        gymnasium==0.29.1 \
        opencv-python \
        scipy==1.14.0 \
        h5py==3.11.0 \
        mediapy==1.2.0 \
        imageio \
        gdown \
        colorlog \
        colorama \
        ipdb \
        Pillow \
        open3d==0.18.0 \
        scikit-learn \
        openai

COPY --from=sources /app/rrt-algorithms /app/rrt-algorithms
RUN mkdir -p /app/rrt-algorithms &&      cd /app/rrt-algorithms &&      uv pip install --no-cache-dir -e . &&      rm -rf /app/rrt-algorithms/.git

COPY --from=sources /app/VLABench /app/VLABench
RUN mkdir -p /app/VLABench &&      cd /app/VLABench &&      uv pip install --no-cache-dir -e . &&      rm -rf /app/VLABench/.git

# Download assets (obj + scene from Google Drive)

RUN cd /app/VLABench && \
    python scripts/download_assets.py
