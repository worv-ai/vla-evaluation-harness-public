#!/bin/bash
# One-off: export benchmark images into vla-eval's Charliecloud cache as SquashFS (what `vla-eval run` looks for).
# Lazy export on CephFS takes >30 min per image; this pulls into node-local storage and converts on /tmp.
#   IMAGES="ghcr.io/allenai/vla-evaluation-harness/robocasa365:0.7.0" MODES="gpu cpu" sbatch scripts/profile/prep_image.sbatch
#SBATCH --job-name=vlaeval-prep
#SBATCH --partition=h100
#SBATCH --exclude=A100-Zebra,DGX-H100-9,DGX-H100-10
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/profile/env.sh
DRV=$(cat /sys/module/nvidia/version)
tmp=$(mktemp -d /tmp/vlaeval-prep.XXXX); trap 'rm -rf "$tmp"' EXIT
# SHARED_STORAGE=1: convert an image already pulled into $CH_IMAGE_STORAGE (e.g. from the login node when a compute
# node's download keeps breaking); default: node-local storage, layers re-downloaded (a shared dlcache races between jobs)
[ "${SHARED_STORAGE:-0}" = 1 ] || export CH_IMAGE_STORAGE=$tmp/storage
ch-image list >/dev/null
for src in ${IMAGES:?}; do
  pulled=""
  for mode in ${MODES:-gpu cpu}; do
    key=$(echo "$src" | tr '/:' '%+'); [ "$mode" = gpu ] && key="$key+nvidia-$DRV"
    dst=$VLA_EVAL_HOME/charliecloud/.squashfs/$key.sqfs
    [ -f "$dst" ] && { echo "have $dst"; continue; }
    echo "=== $src ($mode) -> $dst $(date)"
    if [ -z "$pulled" ] && [ "${SHARED_STORAGE:-0}" = 1 ]; then
      pulled=$(ch-image list | grep -E "^(${src}|${src/ghcr.io\/allenai\/vla-evaluation-harness/ghcr.io\/worv-ai\/vla-evaluation-harness-public})$" | head -1)
      [ -n "$pulled" ] || { echo "not in shared storage: $src"; exit 1; }
    fi
    if [ -z "$pulled" ]; then
      mirror=${src/ghcr.io\/allenai\/vla-evaluation-harness/ghcr.io\/worv-ai\/vla-evaluation-harness-public}
      for ref in "$src" "$mirror" "$mirror" "$mirror"; do  # multi-GB layers: retry broken downloads
        if ch-image pull "$ref" > "$tmp/pull.log" 2>&1; then pulled=$ref; break; fi
        echo "pull of $ref failed: $(tail -1 "$tmp/pull.log" | cut -c1-120)"
      done
      [ -n "$pulled" ] || exit 1
    fi
    ch-convert -i ch-image -o dir "$pulled" "$tmp/img"
    [ "$mode" = gpu ] && ch-fromhost --nvidia "$tmp/img"
    mkdir -p "$(dirname "$dst")"
    TMPDIR=$tmp ch-convert -i dir -o squash "$tmp/img" "$dst.tmp" && mv "$dst.tmp" "$dst"
    rm -rf "$tmp/img"
    echo "done $(date)"; ls -la "$dst"
  done
  [ -n "$pulled" ] && [ "${SHARED_STORAGE:-0}" != 1 ] && ch-image delete "$pulled" || true
done
