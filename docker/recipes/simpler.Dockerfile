# SimplerEnv evaluation environment
# Includes ManiSkill2_real2sim + SimplerEnv.

ARG BASE_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base:latest
ARG RUNTIME_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base-render:latest
FROM ${BASE_IMAGE} AS builder

# ── uv environment ──────────────────────────────────────────────────
ARG PYTHON_VERSION=3.10.18
RUN uv python install "${PYTHON_VERSION}" \
    && uv venv --seed --python "${PYTHON_VERSION}" /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ── ManiSkill2_real2sim + SimplerEnv source ───────────────────────────
WORKDIR /app
COPY --from=sources /app/ManiSkill2_real2sim /app/ManiSkill2_real2sim
RUN mkdir -p /app/ManiSkill2_real2sim      && cd /app/ManiSkill2_real2sim && rm -rf .git

COPY --from=sources /app/simpler_env_src /app/simpler_env_src
RUN mkdir -p /app/simpler_env_src      && cd /app/simpler_env_src && rm -rf .git

# ── Install simulation stack ──────────────────────────────────────────
# sapien imports pkg_resources, removed from conda-forge setuptools >=70.

RUN uv pip install --no-cache-dir "setuptools<70"
# ruckig 0.17.3's pyproject uses cmake.targets, removed in scikit-build-core >= 0.10.
RUN printf 'scikit-build-core<0.10\n' > /tmp/build-constraints.txt \
    && cd /app/ManiSkill2_real2sim \
    && uv pip install --no-cache-dir --build-constraints /tmp/build-constraints.txt -e .
RUN cd /app/simpler_env_src && uv pip install --no-cache-dir -e .

# opencv-python 4.12 pulls numpy>=2, but SimplerEnv needs numpy 1.24.
# Install opencv first, then force-downgrade numpy.
RUN uv pip install --no-cache-dir \
        matplotlib mediapy omegaconf hydra-core \
        opencv-python==4.12.0.88
RUN uv pip install --no-cache-dir numpy==1.24.4

# ── Symlink for rgb_overlay_path config compatibility ─────────────────
RUN mkdir -p /app/simpler \
    && ln -sfn /app/ManiSkill2_real2sim /app/simpler/ManiSkill2_real2sim

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src/ src/
ARG HARNESS_VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${HARNESS_VERSION}
RUN uv pip install --no-cache-dir -e .
COPY configs/ configs/
