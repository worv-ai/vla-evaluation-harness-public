ARG BASE_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base:latest
ARG RUNTIME_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base-render:latest
FROM ${BASE_IMAGE} AS builder

# ── Locked native environment ──────────────────────────────────────────────
RUN install-native-env py38
ENV CONDA_PREFIX=/opt/pixi/.pixi/envs/py38 \
    PATH=/opt/pixi/.pixi/envs/py38/bin:$PATH
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ── CPU-only PyTorch ───────────────────────────────────────────────

ARG TORCH_BACKEND=cpu
ENV UV_TORCH_BACKEND=${TORCH_BACKEND}

RUN uv pip install --no-cache-dir "torch==2.1.0" \
        --torch-backend "${TORCH_BACKEND}"

# ── Python dependencies ───────────────────────────────────────────
RUN uv pip install --no-cache-dir \
        numpy==1.22.4 mujoco==3.2.3 robosuite==1.4.0 bddl==1.0.1 \
        Pillow scipy h5py pyyaml termcolor future easydict hydra-core \
        "gym>=0.21,<0.27" matplotlib \
    && rm -rf ~/.cache/pip ~/.cache/uv

# ── Clone LIBERO ───────────────────────────────────────────────────
COPY --from=sources /app/libero /app/libero
RUN mkdir -p /app/libero \
    && cd /app/libero \
    && uv pip install --no-cache-dir --config-setting editable_mode=compat -e . \
    && rm -rf /app/libero/.git ~/.cache/uv

# ── Pre-create LIBERO config (suppress interactive prompt) ─────────

RUN mkdir -p /root/.libero && printf '\
benchmark_root: /app/libero/libero/libero\n\
bddl_files: /app/libero/libero/libero/bddl_files\n\
init_states: /app/libero/libero/libero/init_files\n\
datasets: /app/libero/libero/datasets\n\
assets: /app/libero/libero/libero/assets\n' > /root/.libero/config.yaml

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src/ src/
ARG HARNESS_VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${HARNESS_VERSION}
RUN uv pip install --no-cache-dir -e .
COPY configs/ configs/
