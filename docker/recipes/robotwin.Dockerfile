# RoboTwin 2.0 — dual-arm manipulation (SAPIEN + CuRobo)
# https://github.com/RoboTwin-Platform/RoboTwin (main branch = 2.0)
#
# Includes SAPIEN 3 (GPU renderer), CuRobo (motion planner), mplib.

ARG BASE_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base-cuda:latest
ARG RUNTIME_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base-runtime:latest
FROM ${BASE_IMAGE} AS builder
ARG BUILD_JOBS=4
ENV MAX_JOBS=${BUILD_JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}

# ── CUDA build tools (nvcc required by CuRobo CUDA extensions) ──────
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvcc-12-1 cuda-cudart-dev-12-1 ninja-build \
    && rm -rf /var/lib/apt/lists/*

# ── uv environment (Python 3.10 required by SAPIEN 3) ────────────
ARG PYTHON_VERSION=3.10.18
RUN uv python install "${PYTHON_VERSION}" \
    && uv venv --seed --python "${PYTHON_VERSION}" /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ── PyTorch with CUDA 12.1 (required by CuRobo / SAPIEN GPU) ────────
ARG TORCH_BACKEND=cu121
ENV UV_TORCH_BACKEND=${TORCH_BACKEND}

RUN uv pip install --no-cache-dir \
        "torch==2.4.1" "torchvision==0.19.1" \
        --torch-backend "${TORCH_BACKEND}"

# ── SAPIEN 3, mplib, and core simulation dependencies ────────────────
# sapien imports pkg_resources, removed from conda-forge setuptools >=70.
RUN uv pip install --no-cache-dir "setuptools<70"
RUN uv pip install --no-cache-dir \
        "sapien==3.0.0b1" \
        "mplib==0.2.1" \
        "transforms3d==0.4.2" \
        "scipy==1.10.1" \
        "gymnasium==0.29.1" \
        "trimesh==4.4.3" \
        "imageio==2.34.2" \
        "open3d==0.18.0" \
        "pydantic" \
        "h5py" \
        "pyyaml" \
        "termcolor" \
        "matplotlib" \
        "Pillow" \
        "huggingface_hub==0.25.0"

# ── Fix SAPIEN urdf_loader.py encoding (upstream bug) ────────────────
RUN SAPIEN_LOC=$(python -c "import sapien; print(sapien.__path__[0])") && \
    sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' \
        "$SAPIEN_LOC/wrapper/urdf_loader.py"

# ── Fix mplib planner.py (remove `or collide` check) ────────────────
RUN MPLIB_LOC=$(python -c "import mplib; print(mplib.__path__[0])") && \
    sed -i -E \
        's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' \
        "$MPLIB_LOC/planner.py"

# ── Clone RoboTwin ───────────────────────────────────────────────────
COPY --from=sources /app/RoboTwin /app/RoboTwin
RUN mkdir -p /app/RoboTwin      && cd /app/RoboTwin && rm -rf /app/RoboTwin/.git

# ── Install CuRobo (motion planner, built from source) ──────────────
# No GPU visible during build → must specify arch explicitly.

ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
# NVlabs/curobo v0.7.8
ARG CUROBO_REF=d64c4b005459db10c5dd867d8b30a87d5bda9bdb
RUN uv pip install --no-cache-dir wheel \
    && cd /app/RoboTwin/envs \
    && mkdir -p curobo \
    && cd curobo \
    && git init -q \
    && git remote add origin https://github.com/NVlabs/curobo.git \
    && git fetch -q --depth 1 origin "${CUROBO_REF}" \
    && git checkout -q FETCH_HEAD \
    && { git remote get-url origin; git rev-parse HEAD; } >> /usr/local/share/vla-build/sources.txt \
    && uv pip install --no-cache-dir --no-build-isolation . \
    && rm -rf .git

# ── Download RoboTwin assets (objects, embodiments, textures) ────────
RUN cd /app/RoboTwin/assets \
    && python _download.py \
    && unzip -qo objects.zip && rm -f objects.zip \
    && unzip -qo embodiments.zip && rm -f embodiments.zip \
    && unzip -qo background_texture.zip && rm -f background_texture.zip

# ── Resolve embodiment config paths ──────────────────────────────────
RUN cd /app/RoboTwin && python script/update_embodiment_config_path.py

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src/ src/
ARG HARNESS_VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${HARNESS_VERSION}
RUN uv pip install --no-cache-dir -e .
COPY configs/ configs/
