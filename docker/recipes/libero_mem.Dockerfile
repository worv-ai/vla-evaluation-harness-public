ARG BASE_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base:latest
ARG RUNTIME_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base-render:latest
FROM ${BASE_IMAGE} AS builder

ARG PYTHON_VERSION=3.11.13
RUN uv python install "${PYTHON_VERSION}" \
    && uv venv --seed --python "${PYTHON_VERSION}" /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG TORCH_BACKEND=cpu
ENV UV_TORCH_BACKEND=${TORCH_BACKEND}

RUN uv pip install --no-cache-dir "torch==2.1.0" \
        --torch-backend "${TORCH_BACKEND}"

# robosuite/robomimic are installed from libero-mem's bundled thirdparty/
# forks (not PyPI) — they contain modifications for memory-dependent envs.
# opencv-python is preinstalled so the fork's unpinned dep keeps numpy 1.x (opencv >=4.12 needs numpy 2).
# numpy 1.24.4 (LIBERO pins 1.22.4, which has no cp311 wheel), numba and scipy are what the 3.8 image ran;
# with them the 3.11 LIBERO image reproduces the 3.8 one's observations and actions bit for bit.
RUN uv pip install --no-cache-dir \
        numpy==1.24.4 numba==0.58.1 scipy==1.10.1 mujoco==3.2.3 bddl==1.0.1 opencv-python \
        Pillow h5py pyyaml termcolor future easydict hydra-core \
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
