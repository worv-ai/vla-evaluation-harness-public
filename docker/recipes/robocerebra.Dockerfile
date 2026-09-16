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

# ── Clone RoboCerebra & install its custom LIBERO fork ─────────────
COPY --from=sources /tmp/RoboCerebra /tmp/RoboCerebra
RUN mkdir -p /tmp/RoboCerebra \
    && cd /tmp/RoboCerebra \
    && cd /tmp/RoboCerebra/LIBERO \
    && uv pip install --no-cache-dir --config-setting editable_mode=compat -e . \
    && uv pip install --no-cache-dir "bddl==1.0.1" h5py \
    && rm -rf /tmp/RoboCerebra/.git ~/.cache/uv

# ── Download RoboCerebra benchmark data ────────────────────────────

ARG ROBOCEREBRA_ASSETS_REF=4e386b9aa266f05b199739d7b58950252244ea21
RUN uv pip install --no-cache-dir huggingface_hub \
    && huggingface-cli download qiukingballball/RoboCerebraBench \
        --repo-type dataset --revision "${ROBOCEREBRA_ASSETS_REF}" \
        --local-dir /workspace/RoboCerebra_Bench \
    && rm -rf ~/.cache/pip ~/.cache/uv

# ── Pre-create LIBERO config (suppress interactive prompt) ─────────
RUN mkdir -p /root/.libero && printf '\
benchmark_root: /tmp/RoboCerebra/LIBERO/libero/libero\n\
bddl_files: /tmp/RoboCerebra/LIBERO/libero/libero/bddl_files\n\
init_states: /tmp/RoboCerebra/LIBERO/libero/libero/init_files\n\
datasets: /tmp/RoboCerebra/LIBERO/libero/datasets\n\
assets: /tmp/RoboCerebra/LIBERO/libero/libero/assets\n' > /root/.libero/config.yaml
