# CALVIN evaluation environment
# Long-horizon language-conditioned manipulation benchmark
# https://github.com/mees/calvin

ARG BASE_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base:latest
ARG RUNTIME_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base-render:latest
FROM ${BASE_IMAGE} AS builder
ARG BUILD_JOBS=4
ENV MAX_JOBS=${BUILD_JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}

ARG PYTHON_VERSION=3.11.13
RUN uv python install "${PYTHON_VERSION}" \
    && uv venv --seed --python "${PYTHON_VERSION}" /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY --from=sources /app/calvin /app/calvin

ARG TORCH_BACKEND=cpu
ENV UV_TORCH_BACKEND=${TORCH_BACKEND}

# calvin_models pins torch==1.13.1, hydra-core==1.1.1 and pyhash, none of which install on 3.11, so it goes
# in --no-deps with what the eval path imports. torch 2.1.0 and numpy 1.24.4 are what the 3.8 image ran.
RUN uv pip install --no-cache-dir "numpy==1.24.4" "torch==2.1.0" "torchvision==0.16.0" --torch-backend "${TORCH_BACKEND}"
RUN cd /app/calvin/calvin_env && uv pip install --no-cache-dir "numpy==1.24.4" -e . \
    && cd /app/calvin/calvin_models && uv pip install --no-cache-dir --no-deps -e . \
    && uv pip install --no-cache-dir "numpy==1.24.4" "hydra-core==1.3.2" hydra-colorlog omegaconf matplotlib \
        opencv-python plotly "pytorch-lightning==1.8.6" lightning_lite gitpython scipy gym moviepy tqdm \
        termcolor fnvhash "setuptools<70"
# pyhash does not build on 3.11; calvin imports it at module level but the eval path never hashes with it.
RUN sed -i 's/^import pyhash$/import fnvhash/; s/^hasher = pyhash.fnv1_32()$/hasher = lambda s: fnvhash.fnv1_32(s.encode())/' \
    /app/calvin/calvin_models/calvin_agent/evaluation/utils.py /app/calvin/calvin_models/calvin_agent/datasets/base_dataset.py

# Patch: catch pybullet.error on double-disconnect during GC teardown
# (upstream only catches TypeError, not pybullet.error)
RUN sed -i 's/except TypeError:/except Exception:/' \
    /app/calvin/calvin_env/calvin_env/envs/play_table_env.py

# Remove .git history to shrink image, then re-init minimal repos
# (CALVIN calls git diff internally via get_git_commit_hash)
RUN find /app/calvin -name .git -exec rm -rf {} + 2>/dev/null || true \
    && git config --global user.email "x" && git config --global user.name "x" \
    && git -C /app/calvin/calvin_env init -q && git -C /app/calvin/calvin_env add -A && git -C /app/calvin/calvin_env commit -qm init \
    && git -C /app/calvin init -q && git -C /app/calvin add -A && git -C /app/calvin commit -qm init

# Embed CALVIN validation config (17KB) — no volume mount needed
COPY docker/calvin_validation_data/ /data/calvin/dataset/validation/

WORKDIR /workspace

# CALVIN otherwise compiles this GPU-discovery helper on first reset.
RUN cd /app/calvin/calvin_env/egl_check && bash build.sh && test -x EGL_options.o
