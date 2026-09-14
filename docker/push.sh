#!/usr/bin/env bash
# Push Docker images to ghcr.io.
# Requires: docker login ghcr.io
# Usage:
#   docker/push.sh --tag 0.1.0 --profile all  # push CPU/GPU tags and update latest-cpu/latest-gpu
#   docker/push.sh --tag 0.1.0 libero   # push a single image
#   docker/push.sh --tag 0.1.0 --no-latest  # push version tag only
#   docker/push.sh --tag 0.1.0 --registry ghcr.io/worv-ai/vla-evaluation-public
#                                       # mirror to another registry (re-tags the local build)
#   docker/push.sh                      # push :latest only (with confirmation)
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/images.sh"
PROFILE=gpu
LAYOUT=full
TAG="latest"
TARGET=""
SOURCE_REGISTRY="ghcr.io/allenai/vla-evaluation-harness"  # where build.sh tags images locally
REGISTRY="$SOURCE_REGISTRY"                               # push target; override with --registry
FORCE=false
UPDATE_LATEST=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --layout)    LAYOUT="$2"; shift 2 ;;
    --profile)   PROFILE="$2"; shift 2 ;;
    --tag)       TAG="$2"; shift 2 ;;
    --registry)  REGISTRY="$2"; shift 2 ;;
    --no-latest) UPDATE_LATEST=false; shift ;;
    -y)          FORCE=true; shift ;;
    -h|--help)
      sed -n '2,/^[^#]/{ s/^# \?//p; }' "$0"
      exit 0 ;;
    -*)          echo "Unknown flag: $1"; exit 1 ;;
    *)           TARGET="$1"; shift ;;
  esac
done

if [[ "$TAG" == "latest" && "$FORCE" != true ]]; then
  read -rp "WARNING: Pushing ':latest' without a version tag. Continue? [y/N] " ans
  [[ "$ans" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
  UPDATE_LATEST=false  # already pushing as latest, no need to double-tag
fi

IMAGES=("${BASE_IMAGES[@]}" "${BENCHMARKS[@]}" "${DERIVED_BENCHMARKS[@]}")
# Images excluded from registry pushes — build locally only.

is_no_redist() {
  local n="$1"
  for g in "${NO_REDIST[@]}"; do
    [[ "$g" == "$n" ]] && return 0
  done
  return 1
}

push_image() {
  local name="$1" profile="${2:-gpu}" suffix="-${2:-gpu}"
  [[ "$name" != base* ]] || suffix=""
  local image_name="${name//_/-}"
  suffix="${suffix}${3:-}"
  local source="${SOURCE_REGISTRY}/${image_name}:${TAG}${suffix}"
  local versioned="${REGISTRY}/${image_name}:${TAG}${suffix}"

  if is_no_redist "$name"; then
    echo "Refusing to push ${versioned}: image bundles proprietary-licensed"
    echo "binaries that may not be redistributed to a public registry."
    echo "(See docs/reproductions/${name}.md for the license rationale.)"
    return 0
  fi

  # build.sh always tags under SOURCE_REGISTRY; re-tag when mirroring to another registry.
  if [[ "$versioned" != "$source" ]]; then
    echo "Tagging: ${source} -> ${versioned}"
    docker tag "${source}" "${versioned}"
  fi

  echo "Pushing: ${versioned}"
  if ! docker push "${versioned}"; then
    echo "ERROR: Push failed. Make sure you are logged in:"
    echo "  docker login ghcr.io"
    exit 1
  fi

  if [[ "$suffix" == -gpu ]]; then
    docker tag "$versioned" "${REGISTRY}/${image_name}:${TAG}"
    docker push "${REGISTRY}/${image_name}:${TAG}"
  fi
  if [[ "$UPDATE_LATEST" == true ]]; then
    local latest="${REGISTRY}/${image_name}:latest${suffix}"
    echo "Tagging: ${versioned} -> ${latest}"
    docker tag "${versioned}" "${latest}"
    docker push "${latest}"
    if [[ "$suffix" == -gpu ]]; then
      docker tag "$versioned" "${REGISTRY}/${image_name}:latest"
      docker push "${REGISTRY}/${image_name}:latest"
    fi
  fi
}

contains() {
  local needle="$1" item
  shift
  for item in "$@"; do [[ "$item" != "$needle" ]] || return 0; done
  return 1
}
contains "$LAYOUT" full split all || { echo "Unknown layout: $LAYOUT" >&2; exit 1; }
contains "$PROFILE" gpu cpu all || { echo "Unknown profile: $PROFILE" >&2; exit 1; }
if [[ -n "$TARGET" ]]; then
  contains "$TARGET" "${IMAGES[@]}" || { echo "Unknown image: $TARGET" >&2; exit 1; }
  if [[ "$PROFILE" == cpu ]] && ! contains "$TARGET" "${CPU_BENCHMARKS[@]}" "${BASE_IMAGES[@]}"; then
    echo "$TARGET requires a GPU" >&2
    exit 1
  fi
fi
push_layouts() {
  local name="$1" profile="$2"
  if [[ "$LAYOUT" != full ]] && contains "$name" "${SPLIT_BENCHMARKS[@]}"; then
    push_image "$name" "$profile" -runtime
    push_image "$name" "$profile" -assets
  fi
  [[ "$LAYOUT" == split ]] || push_image "$name" "$profile"
}
push_profiles() {
  local name="$1"
  if contains "$name" "${BASE_IMAGES[@]}"; then push_image "$name"; return; fi
  [[ "$PROFILE" == cpu ]] || push_layouts "$name" gpu
  if [[ "$PROFILE" != gpu ]] && contains "$name" "${CPU_BENCHMARKS[@]}"; then push_layouts "$name" cpu; fi
}
if [[ -n "$TARGET" ]]; then
  push_profiles "$TARGET"
else
  for img in "${IMAGES[@]}"; do
    push_profiles "$img"
  done
fi

