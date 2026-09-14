#!/usr/bin/env bash
# Build Docker images locally with BuildKit.
# Usage:
#   docker/build.sh [benchmark] [--tag VERSION] [--profile gpu|cpu|all] [--dry-run]
#   docker/build.sh behavior1k --accept-license behavior1k
#   docker/build.sh robodojo --base-image robodojo:cuda12.8 --accept-license robodojo
#   docker/build.sh libero --base-image BUILDER@sha256:... --runtime-image RUNTIME@sha256:...
#   docker/build.sh libero --build-arg LIBERO_REF=COMMIT
#   docker/build.sh libero --profile cpu --gpu-image GPU_IMAGE@sha256:...
# Base targets: base/base-cuda (builders), base-render/base-runtime (GPU), base-cpu (CPU).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

source docker/images.sh
PROFILE=gpu
TAG=latest
GPU_IMAGE=""
BASE_IMAGE=""
RUNTIME_IMAGE=""
TARGET=""
DRY_RUN=false
ACCEPTED_LICENSES=()
EXTRA_BUILD_ARGS=()
REGISTRY=ghcr.io/allenai/vla-evaluation-harness
declare -A EULA_GATED=(
  [rlbench]="ACCEPT_RLBENCH_LICENCE https://github.com/stepjam/RLBench/blob/master/LICENSE"
  [behavior1k]="ACCEPT_NVIDIA_EULA https://docs.omniverse.nvidia.com/eula/"
  [robodojo]="ACCEPT_NVIDIA_EULA https://docs.omniverse.nvidia.com/eula/"
)

contains() {
  local needle="$1" item
  shift
  for item in "$@"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile|--tag|--gpu-image|--base-image|--runtime-image|--accept-license|--build-arg)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "Missing value for $1" >&2; exit 1; }
      case "$1" in
        --gpu-image) GPU_IMAGE="$2" ;;
        --profile) PROFILE="$2" ;;
        --tag) TAG="$2" ;;
        --base-image) BASE_IMAGE="$2" ;;
        --runtime-image) RUNTIME_IMAGE="$2" ;;
        --accept-license) ACCEPTED_LICENSES+=("$2") ;;
        --build-arg) EXTRA_BUILD_ARGS+=(--build-arg "$2") ;;
      esac
      shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) sed -n '2,/^[^#]/{ s/^# \?//p; }' "$0"; exit 0 ;;
    -*) echo "Unknown flag: $1" >&2; exit 1 ;;
    *) [[ -z "$TARGET" ]] || { echo "Specify one target" >&2; exit 1; }; TARGET="$1"; shift ;;
  esac
done

if [[ -n "$TARGET" ]] && ! contains "$TARGET" "${BASE_IMAGES[@]}" "${BENCHMARKS[@]}" "${DERIVED_BENCHMARKS[@]}"; then
  echo "Unknown image: $TARGET" >&2
  exit 1
fi
contains "$PROFILE" gpu cpu all || { echo "Unknown profile: $PROFILE" >&2; exit 1; }
if [[ "$PROFILE" == cpu && -n "$TARGET" ]] && ! contains "$TARGET" "${CPU_BENCHMARKS[@]}" "${BASE_IMAGES[@]}"; then
  echo "$TARGET requires a GPU; no CPU image is supported" >&2
  exit 1
fi
if [[ -n "$GPU_IMAGE" && ( "$PROFILE" != cpu || -z "$TARGET" ) ]]; then
  echo "--gpu-image requires one benchmark and --profile cpu" >&2
  exit 1
fi
for license in "${ACCEPTED_LICENSES[@]}"; do
  [[ -n "${EULA_GATED[$license]:-}" ]] || { echo "Unknown license: $license" >&2; exit 1; }
done
# An upstream RoboDojo base cannot also serve as the common benchmark builder.
if [[ -z "$TARGET" && -n "$BASE_IMAGE" ]] && contains robodojo "${ACCEPTED_LICENSES[@]}"; then
  echo "Build robodojo separately when overriding --base-image" >&2
  exit 1
fi

if $DRY_RUN; then
  HARNESS_VERSION="${HARNESS_VERSION:-0.0.0}"
else
  HARNESS_VERSION="${HARNESS_VERSION:-$(NO_COLOR=1 uvx -q hatch -q version 2>/dev/null || echo 0.0.0)}"
fi

run_build() {
  if $DRY_RUN; then
    printf '%q ' docker build "$@"
    printf '\n'
  else
    DOCKER_BUILDKIT=1 docker build "$@"
  fi
}

CPU_RECIPES=()
trap 'rm -f "${CPU_RECIPES[@]}"' EXIT
declare -A BUILT=()
build_image() {
  local name="$1" profile="${2:-gpu}" runtime_name=base-runtime builder_name=base-cuda
  local image_tag="$TAG" key="${1}-${2:-gpu}"
  contains "$name" "${BASE_IMAGES[@]}" || image_tag="${TAG}-${profile}"
  local dockerfile="docker/Dockerfile.${name}" arg_name url
  local build_args=() tag_args=()
  [[ -z "${BUILT[$key]:-}" ]] || return 0
  if [[ -n "${EULA_GATED[$name]:-}" ]]; then
    read -r arg_name url <<< "${EULA_GATED[$name]}"
    if ! contains "$name" "${ACCEPTED_LICENSES[@]}"; then
      echo "Skipping $name: pass --accept-license $name ($url)"
      return 0
    fi
    build_args+=(--build-arg "${arg_name}=YES")
  fi
  if [[ "$profile" == cpu ]]; then
    local cpu_runtime="${RUNTIME_IMAGE:-${REGISTRY}/base-cpu:${TAG}}"
    if [[ -z "$GPU_IMAGE" ]]; then
      # A custom CPU runtime must not become the GPU dependency's runtime.
      local RUNTIME_IMAGE=""
      build_image "$name" gpu
    fi
    [[ "$cpu_runtime" != "${REGISTRY}/base-cpu:${TAG}" ]] || build_image base-cpu
    dockerfile=$(mktemp /tmp/vla-cpu-XXXXXX.Dockerfile)
    CPU_RECIPES+=("$dockerfile")
    if $DRY_RUN; then
      printf '%q ' bash docker/generate_cpu_dockerfile.sh "docker/Dockerfile.${name}"
      printf '> %q\n' "$dockerfile"
    fi
    bash docker/generate_cpu_dockerfile.sh "docker/Dockerfile.${name}" > "$dockerfile"
    build_args+=(--build-arg "GPU_IMAGE=${GPU_IMAGE:-${REGISTRY}/${name//_/-}:${TAG}-gpu}")
    build_args+=(--build-arg "RUNTIME_IMAGE=${cpu_runtime}")
  else
  case "$name" in
    base|base-cuda|base-runtime|base-render|base-cpu)
      dockerfile=docker/Dockerfile.base
      if [[ "$name" == base || "$name" == base-cuda ]]; then
        build_args+=(--target builder)
      elif [[ "$name" == base-cpu ]]; then
        build_args+=(--target runtime-cpu)
      else
        build_args+=(--target runtime)
      fi
      if [[ "$name" == base-cuda || "$name" == base-runtime ]]; then
        build_args+=(--build-arg OS_IMAGE=nvidia/cuda:12.1.1-runtime-ubuntu22.04@sha256:8bbc6e304b193e84327fa30d93eea70ec0213b808239a46602a919a479a73b12)
      fi
      ;;
    robodojo)
      build_args+=(--build-arg "BASE_IMAGE=${BASE_IMAGE:-robodojo:cuda12.8}")
      ;;
    simpler_groot|simpler_xvla)
      build_image simpler
      build_args+=(--build-arg "BASE_IMAGE=${REGISTRY}/simpler:${TAG}-gpu")
      build_args+=(--build-arg "BUILD_IMAGE=${BASE_IMAGE:-${REGISTRY}/base:${TAG}}")
      ;;
    *)
      if contains "$name" "${NO_SYSTEM_CUDA_BENCHMARKS[@]}"; then
        runtime_name=base-render
        builder_name=base
      fi
      [[ "$profile" != cpu ]] || runtime_name=base-cpu
      [[ -n "$BASE_IMAGE" ]] || build_image "$builder_name"
      [[ -n "$RUNTIME_IMAGE" ]] || build_image "$runtime_name"
      build_args+=(--build-arg "BASE_IMAGE=${BASE_IMAGE:-${REGISTRY}/${builder_name}:${TAG}}")
      build_args+=(--build-arg "RUNTIME_IMAGE=${RUNTIME_IMAGE:-${REGISTRY}/${runtime_name}:${TAG}}")
      ;;
  esac
  fi
  build_args+=(--build-arg "HARNESS_VERSION=${HARNESS_VERSION}" --build-arg "IMAGE_PROFILE=${profile}")
  if [[ "$profile" == gpu ]] && ! contains "$name" "${BASE_IMAGES[@]}"; then
    tag_args+=(-t "${REGISTRY}/${name//_/-}:${TAG}")
  fi
  run_build -t "${REGISTRY}/${name//_/-}:${image_tag}" "${tag_args[@]}" -f "$dockerfile" \
    "${build_args[@]}" "${EXTRA_BUILD_ARGS[@]}" .
  BUILT[$key]=1
}

build_profiles() {
  local name="$1"
  if contains "$name" "${BASE_IMAGES[@]}"; then
    build_image "$name"
    return
  fi
  [[ "$PROFILE" == cpu ]] || build_image "$name" gpu
  if [[ "$PROFILE" != gpu ]] && contains "$name" "${CPU_BENCHMARKS[@]}"; then
    build_image "$name" cpu
  fi
}

if [[ -n "$TARGET" ]]; then
  build_profiles "$TARGET"
else
  for name in "${BENCHMARKS[@]}" "${DERIVED_BENCHMARKS[@]}"; do
    build_profiles "$name"
  done
fi
