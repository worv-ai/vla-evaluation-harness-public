# RLBench benchmark (CoppeliaSim 4.1.0 + PyRep + RLBench)
# CoppeliaSim needs an X display even in headless mode → Xvfb wrapper
# https://github.com/stepjam/RLBench

ARG BASE_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base:latest
ARG RUNTIME_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base-render:latest
FROM ${BASE_IMAGE} AS builder

ARG ACCEPT_RLBENCH_LICENCE=
RUN if [ "$ACCEPT_RLBENCH_LICENCE" != "YES" ]; then \
        echo ""; \
        echo "============================================================"; \
        echo "Building rlbench requires accepting two licences:"; \
        echo "  1. RLBench (Imperial College London) — academic / non-commercial"; \
        echo "     https://github.com/stepjam/RLBench/blob/master/LICENSE"; \
        echo "  2. CoppeliaSim Edu — educational entities, non-commercial"; \
        echo "     https://manual.coppeliarobotics.com/en/licensing.htm"; \
        echo ""; \
        echo "Read the licences above, then re-run with:"; \
        echo "  docker build --build-arg ACCEPT_RLBENCH_LICENCE=YES ..."; \
        echo "  (or: docker/build.sh rlbench --accept-license rlbench)"; \
        echo "============================================================"; \
        exit 1; \
    fi

# ── Xvfb for headless CoppeliaSim ─────────────────────────────────
# libdbus-1-3: the bundled Qt xcb platform plugin links libdbus-1.so.3, its only
# unresolved dependency (everything else ships with the base image or CoppeliaSim).
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb libfontconfig1 tini libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

# ── CoppeliaSim 4.1.0 ─────────────────────────────────────────────
# PyRep/RLBench require exactly 4.1.0; newer versions cause segfaults.
ENV COPPELIASIM_ROOT=/opt/coppeliasim
# xcb, not offscreen: vision sensors need a Qt GL context or they render black.
# QT_PLUGIN_PATH lets Qt find the bundled xcbglintegrations/ GLX plugin.
ENV LD_LIBRARY_PATH=${COPPELIASIM_ROOT}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    QT_QPA_PLATFORM_PLUGIN_PATH=${COPPELIASIM_ROOT}/platforms \
    QT_PLUGIN_PATH=${COPPELIASIM_ROOT} \
    QT_QPA_PLATFORM=xcb
RUN wget -qO /tmp/coppeliasim.tar.xz \
        https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz \
    && mkdir -p ${COPPELIASIM_ROOT} \
    && tar -xf /tmp/coppeliasim.tar.xz -C ${COPPELIASIM_ROOT} --strip-components=1 \
    && rm /tmp/coppeliasim.tar.xz

# The OpenGL3 renderer plugin (once disabled here for segfaulting headless)
# works with the GLX integration above; cameras keep RLBench's default mode.

# ── Locked native environment (Python 3.8 required by PyRep/RLBench) ──────
RUN install-native-env py38
ENV CONDA_PREFIX=/opt/pixi/.pixi/envs/py38 \
    PATH=/opt/pixi/.pixi/envs/py38/bin:$PATH
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ── PyRep (from source) ───────────────────────────────────────────
# Master-commit pins, not tags: stepjam froze the version strings years ago, so
# the 4.1.0/1.1.0 tags are 2021 code predating the adapter's RLBench API.
# cffi + --no-build-isolation: PyRep imports cffi at setup time.
COPY --from=sources /tmp/PyRep /tmp/PyRep

ARG TORCH_BACKEND=cpu
ENV UV_TORCH_BACKEND=${TORCH_BACKEND}

RUN uv pip install --no-cache-dir cffi \
    && mkdir -p /tmp/PyRep \
    && cd /tmp/PyRep \
    && uv pip install --no-cache-dir --no-build-isolation --config-settings editable_mode=compat -e . \
    && rm -rf /tmp/PyRep/.git

# ── RLBench (from source) ─────────────────────────────────────────
# editable_mode=compat sidesteps uv's PEP 660 empty-finder trap (issue #92).
# --no-deps: RLBench declares an unpinned `pyrep @ git+...` that would replace
# the pinned PyRep above; its other requirements are installed explicitly.

COPY --from=sources /tmp/RLBench /tmp/RLBench
RUN uv pip install --no-cache-dir natsort numpy Pillow pyquaternion scipy \
    && mkdir -p /tmp/RLBench \
    && cd /tmp/RLBench \
    && uv pip install --no-cache-dir --no-deps --config-settings editable_mode=compat -e . \
    && rm -rf /tmp/RLBench/.git

WORKDIR /workspace

# ── Entrypoint: start Xvfb then run vla-eval ──────────────────────
COPY docker/rlbench_entrypoint.sh /rlbench_entrypoint.sh
RUN chmod +x /rlbench_entrypoint.sh

# tini reaps orphaned CoppeliaSim processes so the container exits cleanly
