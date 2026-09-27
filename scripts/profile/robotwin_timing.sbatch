#!/bin/bash
# RoboTwin per-step timing per render variant, one ch-run process per variant, on one GPU.
#   VARIANTS="shipped rt8 rt4 raster" TASK=beat_block_hammer sbatch scripts/profile/robotwin_timing.sbatch
#SBATCH --job-name=vlaeval-rt-timing
#SBATCH --partition=h100
#SBATCH --exclude=A100-Zebra,DGX-H100-9,DGX-H100-10
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:1
#SBATCH --time=01:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/profile/env.sh
OUT=${OUT:-/mnt/cepheid/users/claude/scratch/vla-profile/rt-timing/$SLURM_JOB_ID}; mkdir -p "$OUT"
SQFS=$VLA_EVAL_HOME/charliecloud/.squashfs/ghcr.io%allenai%vla-evaluation-harness%robotwin+0.7.0+nvidia-$(cat /sys/module/nvidia/version).sqfs
IMG=$(mktemp -d); squashfuse_ll -o ro,uid=$(id -u),gid=$(id -g) "$SQFS" "$IMG"; trap 'fusermount -u "$IMG"' EXIT
for v in ${VARIANTS:-shipped raster}; do
  echo "=== variant $v $(date -Is)"
  ch-run --write-fake --unset-env='*' --set-env --set-env=HOME=/root --set-env=OMP_NUM_THREADS=1 --set-env=MKL_NUM_THREADS=1 \
    --set-env=CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" --cd /workspace \
    -b "$PWD/src:/workspace/src" -b "$PWD/scripts/profile:/workspace/profile" -b "$OUT:/workspace/results" "$IMG" -- \
    conda run --no-capture-output -n robotwin python /workspace/profile/robotwin_step_timing.py \
      --task "${TASK:-beat_block_hammer}" --steps "${STEPS:-30}" --variant "$v" 2>&1 | grep -v "^step:" | tail -40
done
