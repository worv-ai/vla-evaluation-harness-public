# CALVIN evaluation environment
# Long-horizon language-conditioned manipulation benchmark
# https://github.com/mees/calvin

ARG BASE_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base:latest
ARG RUNTIME_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base-render:latest
FROM ${BASE_IMAGE} AS builder
ARG BUILD_JOBS=4
ENV MAX_JOBS=${BUILD_JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}

# multicoretsne (CALVIN dep) needs FFTW and BLAS to compile
RUN apt-get update && apt-get install -y --no-install-recommends \
        libfftw3-dev libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN install-native-env calvin
ENV CONDA_PREFIX=/opt/pixi/.pixi/envs/calvin \
    PATH=/opt/pixi/.pixi/envs/calvin/bin:$PATH
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY --from=sources /app/calvin /app/calvin
# CALVIN editable installs need legacy setuptools/cmake and no build isolation.

ARG TORCH_BACKEND=cpu
ENV UV_TORCH_BACKEND=${TORCH_BACKEND}

RUN uv pip install --no-cache-dir "setuptools==57.5.0" wheel \
    && cd /app/calvin/calvin_env   && pip install --no-cache-dir --no-build-isolation -e . \
    && cd /app/calvin/calvin_models && pip install --no-cache-dir --no-build-isolation -e .

# Patch: catch pybullet.error on double-disconnect during GC teardown
# (upstream only catches TypeError, not pybullet.error)
RUN sed -i 's/except TypeError:/except Exception:/' \
    /app/calvin/calvin_env/calvin_env/envs/play_table_env.py

RUN uv pip install --no-cache-dir \
      "torch==2.1.0" --torch-backend "${TORCH_BACKEND}" \
    && uv pip install --no-cache-dir fnvhash

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
