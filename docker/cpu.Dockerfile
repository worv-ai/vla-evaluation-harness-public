# The benchmark's runtime section is appended by generate_cpu_dockerfile.sh.
ARG GPU_IMAGE=scratch
ARG RUNTIME_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base-cpu:latest
FROM ghcr.io/astral-sh/uv:0.11.25@sha256:1e3808aa9023d0980e7c15b1fa7c1ac16ff35925780cf5c459858b2d693f01a9 AS uv
FROM ${GPU_IMAGE} AS gpu
FROM gpu AS builder
COPY --from=uv /uv /usr/local/bin/uv
COPY docker/prepare_cpu.py docker/optimize_libero.py docker/prune_cpu.py docker/export_runtime.py /usr/local/lib/vla/
ARG GPU_IMAGE=scratch
RUN printf '%s\n' "$GPU_IMAGE" > /usr/local/share/vla-build/gpu-source-image.txt
RUN python /usr/local/lib/vla/optimize_libero.py && python /usr/local/lib/vla/prepare_cpu.py && python /usr/local/lib/vla/prune_cpu.py
