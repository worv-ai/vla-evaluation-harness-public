ARG BASE_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base:latest
ARG RUNTIME_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base-render:latest
FROM ${BASE_IMAGE} AS builder

RUN install-native-env py38
ENV CONDA_PREFIX=/opt/pixi/.pixi/envs/py38 \
    PATH=/opt/pixi/.pixi/envs/py38/bin:$PATH
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG TORCH_BACKEND=cpu
ENV UV_TORCH_BACKEND=${TORCH_BACKEND}

RUN uv pip install --no-cache-dir "torch==2.1.0" \
        --torch-backend "${TORCH_BACKEND}"

# robosuite/robomimic are installed from libero-mem's bundled thirdparty/
# forks (not PyPI) — they contain modifications for memory-dependent envs.
RUN uv pip install --no-cache-dir \
        numpy==1.22.4 mujoco==3.2.3 bddl==1.0.1 \
        Pillow scipy h5py pyyaml termcolor future easydict hydra-core \
        "gym>=0.21,<0.27" matplotlib \
    && rm -rf ~/.cache/pip ~/.cache/uv

COPY --from=sources /app/libero-mem /app/libero-mem
RUN mkdir -p /app/libero-mem \
    && cd /app/libero-mem \
    && cd /app/libero-mem/thirdparty/robomimic \
    && uv pip install --no-cache-dir -e . \
    && cd /app/libero-mem/thirdparty/robosuite \
    && uv pip install --no-cache-dir -e . \
    && cd /app/libero-mem \
    && uv pip install --no-cache-dir --config-setting editable_mode=compat -e . \
    && rm -rf /app/libero-mem/.git ~/.cache/uv

RUN mkdir -p /root/.libero && printf '\
benchmark_root: /app/libero-mem/libero/libero\n\
bddl_files: /app/libero-mem/libero/libero/bddl_files\n\
init_states: /app/libero-mem/libero/libero/init_files\n\
datasets: /app/libero-mem/libero/datasets\n\
assets: /app/libero-mem/libero/libero/assets\n' > /root/.libero/config.yaml

# Pre-generated init states (avoids ~43 min MuJoCo generation during build)
# To regenerate, run: docker/generate_libero_mem_inits.py
COPY docker/init_states/libero_mem/ /app/libero-mem/libero/libero/init_files/libero_mem/

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src/ src/
ARG HARNESS_VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${HARNESS_VERSION}
RUN uv pip install --no-cache-dir -e .
COPY configs/ configs/
