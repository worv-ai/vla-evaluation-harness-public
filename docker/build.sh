#!/usr/bin/env bash
# Build Docker images locally with BuildKit.
# Usage:
#   docker/build.sh [benchmark] [--tag VERSION] [--dry-run]
#   docker/build.sh behavior1k --accept-license behavior1k
#   docker/build.sh robodojo --base-image robodojo:cuda12.8 --accept-license robodojo
#   docker/build.sh libero --base-image BUILDER@sha256:... --runtime-image RUNTIME@sha256:...
#   docker/build.sh libero --build-arg LIBERO_REF=COMMIT
# Base targets: base (builder), base-runtime (CUDA), base-cpu (CPU compute + GPU rendering).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TAG=latest
BASE_IMAGE=""
RUNTIME_IMAGE=""
TARGET=""
DRY_RUN=false
ACCEPTED_LICENSES=()
EXTRA_BUILD_ARGS=()
REGISTRY=ghcr.io/allenai/vla-evaluation-harness
BENCHMARKS=(simpler libero libero_pro libero_plus libero_mem robocerebra maniskill2 calvin mikasa_robo vlabench rlbench robotwin robocasa robocasa365 kinetix robomme molmospaces behavior1k duobench robodojo)
DERIVED_BENCHMARKS=(simpler_groot simpler_xvla)
NO_SYSTEM_CUDA_BENCHMARKS=(maniskill2 mikasa_robo robomme molmospaces kinetix duobench simpler libero libero_pro libero_plus libero_mem robocerebra calvin rlbench robocasa robocasa365 vlabench)
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
    --tag|--base-image|--runtime-image|--accept-license|--build-arg)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "Missing value for $1" >&2; exit 1; }
      case "$1" in
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

if [[ -n "$TARGET" ]] && ! contains "$TARGET" base base-runtime base-cpu "${BENCHMARKS[@]}" "${DERIVED_BENCHMARKS[@]}"; then
  echo "Unknown image: $TARGET" >&2
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

declare -A BUILT=()
build_image() {
  local name="$1" runtime_name=base-runtime
  local dockerfile="docker/Dockerfile.${name}" arg_name url
  local build_args=()
  [[ -z "${BUILT[$name]:-}" ]] || return 0
  if [[ -n "${EULA_GATED[$name]:-}" ]]; then
    read -r arg_name url <<< "${EULA_GATED[$name]}"
    if ! contains "$name" "${ACCEPTED_LICENSES[@]}"; then
      echo "Skipping $name: pass --accept-license $name ($url)"
      return 0
    fi
    build_args+=(--build-arg "${arg_name}=YES")
  fi
  case "$name" in
    base|base-runtime|base-cpu)
      dockerfile=docker/Dockerfile.base
      if [[ "$name" == base ]]; then
        build_args+=(--target builder)
      else
        build_args+=(--target runtime)
      fi
      [[ "$name" != base-cpu ]] || build_args+=(--build-arg CUDA_IMAGE=ubuntu:22.04@sha256:829f6df217bcbae2b371026e81711d1a787c61b2967ad09d015063663ebafbf7)
      ;;
    robodojo)
      build_args+=(--build-arg "BASE_IMAGE=${BASE_IMAGE:-robodojo:cuda12.8}")
      ;;
    simpler_groot|simpler_xvla)
      build_image simpler
      build_args+=(--build-arg "BASE_IMAGE=${REGISTRY}/simpler:${TAG}")
      build_args+=(--build-arg "BUILD_IMAGE=${BASE_IMAGE:-${REGISTRY}/base:${TAG}}")
      ;;
    *)
      [[ -n "$BASE_IMAGE" ]] || build_image base
      contains "$name" "${NO_SYSTEM_CUDA_BENCHMARKS[@]}" && runtime_name=base-cpu
      [[ -n "$RUNTIME_IMAGE" ]] || build_image "$runtime_name"
      build_args+=(--build-arg "BASE_IMAGE=${BASE_IMAGE:-${REGISTRY}/base:${TAG}}")
      build_args+=(--build-arg "RUNTIME_IMAGE=${RUNTIME_IMAGE:-${REGISTRY}/${runtime_name}:${TAG}}")
      ;;
  esac
  build_args+=(--build-arg "HARNESS_VERSION=${HARNESS_VERSION}")
  run_build -t "${REGISTRY}/${name//_/-}:${TAG}" -f "$dockerfile" \
    "${build_args[@]}" "${EXTRA_BUILD_ARGS[@]}" .
  BUILT[$name]=1
}

if [[ -n "$TARGET" ]]; then
  build_image "$TARGET"
else
  for name in "${BENCHMARKS[@]}" "${DERIVED_BENCHMARKS[@]}"; do
    build_image "$name"
  done
fi
