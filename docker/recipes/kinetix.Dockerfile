ARG BASE_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base:latest
ARG RUNTIME_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base-render:latest
FROM ${BASE_IMAGE} AS builder

# ── uv environment ──────────────────────────────────────────────
ARG PYTHON_VERSION=3.11.13
RUN uv python install "${PYTHON_VERSION}" \
    && uv venv --seed --python "${PYTHON_VERSION}" /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ── Kinetix ───────────────────────────────────────────────────────
# FLAIROx/Kinetix v3.0.0
COPY --from=sources /opt/kinetix /opt/kinetix
RUN mkdir -p /opt/kinetix      && cd /opt/kinetix && uv pip install --no-cache-dir -e . numpy Pillow

# ── JAX with CUDA support (after kinetix to override its JAX pin) ─

ARG JAX_EXTRAS=cuda12
RUN uv pip install --no-cache-dir --upgrade "jax${JAX_EXTRAS:+[${JAX_EXTRAS}]}"

# ── RTC level files ───────────────────────────────────────────────
COPY --from=sources /app/rtc /app/rtc
RUN mkdir -p /app/rtc      && cd /app/rtc && rm -rf /app/rtc/.git
