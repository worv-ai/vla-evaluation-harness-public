ARG BASE_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base:latest
ARG RUNTIME_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base-render:latest
FROM ${BASE_IMAGE} AS builder

# ── Locked native environment ──────────────────────────────────────────────
ARG PYTHON_VERSION=3.11.13
RUN uv python install "${PYTHON_VERSION}" \
    && uv venv --seed --python "${PYTHON_VERSION}" /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ── CPU-only PyTorch ───────────────────────────────────────────────

ARG TORCH_BACKEND=cpu
ENV UV_TORCH_BACKEND=${TORCH_BACKEND}

RUN uv pip install --no-cache-dir "torch==2.1.0" \
        --torch-backend "${TORCH_BACKEND}"

# ── Python dependencies ───────────────────────────────────────────
# numpy 1.24.4 (LIBERO pins 1.22.4, which has no cp311 wheel), numba and scipy are what the 3.8 image ran;
# with them the 3.11 LIBERO image reproduces the 3.8 one's observations and actions bit for bit.
RUN uv pip install --no-cache-dir \
        numpy==1.24.4 numba==0.58.1 scipy==1.10.1 mujoco==3.2.3 robosuite==1.4.0 bddl==1.0.1 \
        Pillow h5py pyyaml termcolor future easydict hydra-core \
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
