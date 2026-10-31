#!/bin/bash
# Profiling runs in one Slurm job. Resources come from the sbatch line; everything else from env (run_profile.sh).
# SHARDS may list several counts ("16 32 64"): each runs in turn into $OUT/n<N>.
#   CONFIG=... OUT=... SHARDS="16 32" RENDER=cpu sbatch -c 64 --mem=200G scripts/profile/profile.sbatch
#SBATCH --job-name=vlaeval-profile
#SBATCH --partition=h100
#SBATCH --exclude=A100-Zebra,DGX-H100-9,DGX-H100-10
#SBATCH --nodes=1 --ntasks=1
#SBATCH --time=01:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
# bash reads scripts incrementally: run a snapshot so editing scripts/profile during a run cannot break it
mkdir -p "${OUT:?}"; cp -r scripts/profile "$OUT/.profile-scripts"
for n in ${SHARDS:?}; do
  PROJ=$PWD EVAL_ID=$(basename "${CONFIG:?}" .yaml)-$SLURM_JOB_ID-n$n "$OUT/.profile-scripts/run_profile.sh" "$CONFIG" "$OUT/n$n" "$n" || echo "run n=$n failed"
done
