ARG BASE_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base:latest
ARG RUNTIME_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base-render:latest
FROM ${BASE_IMAGE} AS builder
ARG BUILD_JOBS=4
ENV MAX_JOBS=${BUILD_JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}

# Adapted from robot-control-stack/docker/Dockerfile; sim eval only needs
# RCS core + DuoBench, not the real-robot hardware extensions.

# RCS compiles against the MuJoCo/urdfdom/pin stack below.
# RobotControlStack/robot-control-stack v0.7.1
# RobotControlStack/duobench v0.1.0

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── System build deps the shared base doesn't already provide ──────
# RCS links Poco; cmake/ninja/urdfdom/glfw come from pip/conda below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpoco-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Conda env with Python 3.11 (rcs + duobench requirement) ────────
# urdfdom pinned to 4.x: rcs's compiled C++ extension links against
# liburdfdom_sensor.so.4.0, and newer sonames break that ABI.
RUN install-native-env duobench
ENV CONDA_PREFIX=/opt/pixi/.pixi/envs/duobench \
    PATH=/opt/pixi/.pixi/envs/duobench/bin:$PATH
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# RCS dlopen()s conda libs; `conda run` does not set LD_LIBRARY_PATH like activation.
ENV LD_LIBRARY_PATH=/opt/pixi/.pixi/envs/duobench/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# ── Build / runtime deps ───────────────────────────────────────────
# RCS installs with --no-build-isolation, so build deps must be present here.
# mujoco 3.2.6 + pin 3.7.0 are the versions RCS upstream tests against.
# cmeel-urdfdom<5 keeps the pip-side urdfdom on the same soname-4 ABI.
RUN uv pip install --no-cache-dir \
        "pip>=25.1" \
        build wheel \
        "setuptools>=45" \
        "scikit-build-core>=0.3.3" \
        pybind11 \
        cmake \
        ninja \
        "mujoco==3.2.6" \
        "cmeel-urdfdom<5" \
        "pin==3.7.0"

# ── Clone + install Robot Control Stack (core only — no hardware ext) ──
# The build uses the deps installed above via --no-build-isolation.
COPY --from=sources /opt/rcs /opt/rcs
RUN mkdir -p /opt/rcs && cd /opt/rcs && uv pip install --no-cache-dir --no-build-isolation --editable .      && rm -rf /opt/rcs/.git

# ── Clone + install DuoBench ───────────────────────────────────────

COPY --from=sources /opt/duobench /opt/duobench
RUN mkdir -p /opt/duobench && cd /opt/duobench && uv pip install --no-cache-dir --editable .      && rm -rf /opt/duobench/.git

# ── Headless GL defaults (MuJoCo backend selection at sim init) ────
# benchmark.py also sets these at import time as a belt-and-braces, but
# baking them into the image means the right backend is picked even if the
# runtime doesn't forward the host environment.

ENV MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src/ src/
ARG HARNESS_VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${HARNESS_VERSION}
RUN uv pip install --no-cache-dir -e .
COPY configs/ configs/
