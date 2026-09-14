ARG BASE_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base:latest
ARG RUNTIME_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base-render:latest
FROM ${BASE_IMAGE} AS builder

# ── Locked native environment ──────────────────────────────────────────────
RUN install-native-env py38
ENV CONDA_PREFIX=/opt/pixi/.pixi/envs/py38 \
    PATH=/opt/pixi/.pixi/envs/py38/bin:$PATH
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ── System deps for LIBERO-Plus rendering (ImageMagick/Wand textures) ──
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmagickwand-dev libfontconfig1-dev libexpat1 unzip \
    && rm -rf /var/lib/apt/lists/*

# ── CPU-only PyTorch ───────────────────────────────────────────────

ARG TORCH_BACKEND=cpu
ENV UV_TORCH_BACKEND=${TORCH_BACKEND}

RUN uv pip install --no-cache-dir "torch==2.1.0" \
        --torch-backend "${TORCH_BACKEND}"

# ── Python dependencies ───────────────────────────────────────────
# LIBERO-Plus adds scikit-image, wand (ImageMagick bindings), usd-core
# for the perturbation pipeline; gym is pinned to 0.25.2 to match
# robomimic==0.2.0 (pulled transitively by LIBERO-Plus).
RUN uv pip install --no-cache-dir \
        numpy==1.22.4 mujoco==3.2.3 robosuite==1.4.0 bddl==1.0.1 \
        Pillow scipy h5py pyyaml termcolor future easydict hydra-core \
        "gym>=0.21,<0.27" matplotlib opencv-python scikit-image wand \
    && rm -rf ~/.cache/pip ~/.cache/uv

# ── Clone LIBERO-Plus ──────────────────────────────────────────────
# The fork installs under the same `libero` package namespace as vanilla
# LIBERO and cannot coexist with it; this image intentionally does not
# install the upstream `Lifelong-Robot-Learning/LIBERO` repo.
COPY --from=sources /app/LIBERO-plus /app/LIBERO-plus
RUN mkdir -p /app/LIBERO-plus \
    && cd /app/LIBERO-plus \
    && uv pip install --no-cache-dir --config-setting editable_mode=compat -e . \
    && rm -rf /app/LIBERO-plus/.git ~/.cache/uv

# ── Download LIBERO-Plus MuJoCo assets (≈6.4 GB) ───────────────────
# assets.zip is stored with a deeply nested top-level path
# (`inspire/hdd/.../LIBERO-plus-0/assets`); we extract to a scratch dir
# and move `assets/` into the installed libero package root so BDDL
# files resolve scene/texture XMLs correctly.

ARG LIBERO_PLUS_ASSETS_REF=dd2bd61b7d9a6fef1abc52d606e983b41886a149
RUN wget --progress=dot:giga \
        "https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/${LIBERO_PLUS_ASSETS_REF}/assets.zip?download=true" \
        -O /tmp/assets.zip \
    && unzip -q /tmp/assets.zip -d /tmp/assets-extracted \
    && rm /tmp/assets.zip \
    && mv "$(find /tmp/assets-extracted -type d -name assets -print -quit)" \
          /app/LIBERO-plus/libero/libero/assets \
    && rm -rf /tmp/assets-extracted

# ── Pre-create LIBERO config (suppress interactive prompt) ─────────
RUN mkdir -p /root/.libero && printf '\
benchmark_root: /app/LIBERO-plus/libero/libero\n\
bddl_files: /app/LIBERO-plus/libero/libero/bddl_files\n\
init_states: /app/LIBERO-plus/libero/libero/init_files\n\
datasets: /app/LIBERO-plus/libero/datasets\n\
assets: /app/LIBERO-plus/libero/libero/assets\n' > /root/.libero/config.yaml

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src/ src/
ARG HARNESS_VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${HARNESS_VERSION}
RUN uv pip install --no-cache-dir -e .
COPY configs/ configs/
