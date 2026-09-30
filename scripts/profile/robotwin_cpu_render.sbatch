#!/bin/bash
# Experiment: can RoboTwin (SAPIEN 3.0.0b1, ray-traced cameras) render on the CPU with a newer Mesa lavapipe?
# Installs mesa-lavapipe from conda-forge into the container's tmpfs overlay, then runs the step timing with no GPU.
#   VARIANTS="raster rt4" sbatch scripts/profile/robotwin_cpu_render.sbatch
#SBATCH --job-name=vlaeval-rt-cpu
#SBATCH --partition=h100
#SBATCH --exclude=A100-Zebra,DGX-H100-9,DGX-H100-10
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH --time=01:30:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/profile/env.sh
OUT=${OUT:-/mnt/cepheid/users/claude/scratch/vla-profile/rt-cpu/$SLURM_JOB_ID}; mkdir -p "$OUT"
# Only the GPU export exists; VK_ICD_FILENAMES pins lavapipe, so the injected driver is unused.
SQFS=$VLA_EVAL_HOME/charliecloud/.squashfs/ghcr.io%allenai%vla-evaluation-harness%robotwin+0.7.0+nvidia-$(cat /sys/module/nvidia/version).sqfs
IMG=$(mktemp -d); squashfuse_ll -o ro,uid=$(id -u),gid=$(id -g) "$SQFS" "$IMG"; trap 'fusermount -u "$IMG"' EXIT
LVP=$OUT/lavapipe-env
if [ ! -f "$LVP/share/vulkan/icd.d/lvp_icd.x86_64.json" ]; then
  echo "=== installing mesa-lavapipe $(date -Is)"
  ch-run --write-fake --unset-env='*' --set-env --set-env=HOME=/root -b "$OUT:/out" "$IMG" -- \
    conda create -y -p /out/lavapipe-env -c conda-forge "mesa-lavapipe==${MESA:-26.2.1}" 2>&1 | tail -3
fi
sed "s#\"library_path\" *: *\"[^\"]*\"#\"library_path\": \"/opt/lavapipe-env/lib/libvulkan_lvp.so\"#" \
  "$LVP/share/vulkan/icd.d/lvp_icd.x86_64.json" > "$OUT/lvp_icd.json"; cat "$OUT/lvp_icd.json"
for v in ${VARIANTS:-raster}; do
  echo "=== variant $v $(date -Is)"
  ch-run --write-fake --unset-env='*' --set-env --set-env=HOME=/root --set-env=OMP_NUM_THREADS=1 --set-env=MKL_NUM_THREADS=1 \
    --set-env=LP_NUM_THREADS="${LP_NUM_THREADS:-8}" --set-env=CUDA_VISIBLE_DEVICES= \
    --set-env=VK_ICD_FILENAMES=/out/lvp_icd.json --set-env=VK_DRIVER_FILES=/out/lvp_icd.json --cd /workspace \
    -b "$PWD/src:/workspace/src" -b "$PWD/scripts/profile:/workspace/profile" -b "$OUT:/out" -b "$OUT:/workspace/results" -b "$LVP:/opt/lavapipe-env" "$IMG" -- \
    conda run --no-capture-output -n robotwin python /workspace/profile/robotwin_step_timing.py \
      --task "${TASK:-beat_block_hammer}" --steps "${STEPS:-10}" --episodes 1 --variant "$v" 2>&1 | grep -v "^step:" | tail -30 || true
done
