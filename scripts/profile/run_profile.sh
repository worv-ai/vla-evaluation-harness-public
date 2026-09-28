#!/bin/bash
# Profile one benchmark config: instant model server + N vla-eval shards (Charliecloud) on one node, then a summary.
#   usage: run_profile.sh <config.yaml> <out_dir> <shards>
# env:
#   RENDER       gpu|cpu (default: the config's). cpu attaches no GPU to the shards.
#   RENDER_GPUS  GPU ids handed to shards round-robin (default: all of CUDA_VISIBLE_DEVICES)
#   ACTION_DIM   dummy server action dim (default: the config's first entry's action_dim, else 7)
#   HOLD_KEY     observation key echoed as the action (RoboTwin: joint_state)
#   OBS_PARAMS   JSON for the server's get_observation_params()
#   CHUNK_SIZE   server chunk size (default 1: every env step asks the server)
#   OPEN_LOOP=1  the server sends whole chunks; the client executes them without observing (VLA_EVAL_OPEN_LOOP_CHUNKS)
#   RUN_ARGS     extra `vla-eval run` args, e.g. "--benchmark-field episodes_per_task=2 --benchmark-field max_tasks=4"
#   VLA          vla-eval executable (default: this checkout's .venv); DEV=1 mounts this checkout's src (--dev)
#   SERVER_URL   use an already running server instead of the dummy one
#   LOCAL_IMAGE=1  stage the image's SquashFS export on node-local /tmp first: shards read the image through
#                squashfuse, and CephFS traffic from other jobs on the node (an image export, say) stalls them
set -euo pipefail
CONFIG=$(realpath "$1"); OUT=$2; SHARDS=$3
HERE=$(cd "$(dirname "$0")" && pwd); PROJ=${PROJ:-$(cd "$HERE/../.." && pwd)}
source "$HERE/env.sh"
JOB=${SLURM_JOB_ID:-local$$}; EVAL_ID=${EVAL_ID:-$(basename "$CONFIG" .yaml)-$JOB}
PORT=${PORT:-$((20000 + $$ % 20000))}
VLA=${VLA:-$PROJ/.venv/bin/vla-eval}; PY=$(dirname "$VLA")/python
mkdir -p "$OUT/logs"; OUT=$(realpath "$OUT")
GPUS=(${RENDER_GPUS:-${CUDA_VISIBLE_DEVICES:-}}); GPUS=(${GPUS[@]//,/ })
[ "${RENDER:-}" = cpu ] && GPUS=()
ACTION_DIM=${ACTION_DIM:-$("$PY" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['benchmarks'][0].get('action_dim', 7))" "$CONFIG")}

if [ "${LOCAL_IMAGE:-0}" = 1 ]; then
  IMG=$("$PY" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['docker']['image'])" "$CONFIG")
  MODE=${RENDER:-$("$PY" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1])).get('render') or 'gpu')" "$CONFIG")}
  KEY=$(echo "$IMG" | tr '/:' '%+'); [ "$MODE" = gpu ] && KEY="$KEY+nvidia-$(cat /sys/module/nvidia/version)"
  LOCAL_HOME=/tmp/vla-eval-$USER; mkdir -p "$LOCAL_HOME/charliecloud/.squashfs"
  if [ ! -f "$LOCAL_HOME/charliecloud/.squashfs/$KEY.sqfs" ]; then
    echo "staging $KEY.sqfs on $LOCAL_HOME $(date -Is)"
    cp "$VLA_EVAL_HOME/charliecloud/.squashfs/$KEY.sqfs" "$LOCAL_HOME/charliecloud/.squashfs/$KEY.sqfs.tmp" \
      && mv "$LOCAL_HOME/charliecloud/.squashfs/$KEY.sqfs.tmp" "$LOCAL_HOME/charliecloud/.squashfs/$KEY.sqfs"
  fi
  export VLA_EVAL_HOME=$LOCAL_HOME
fi

SPID=; MON=
cleanup() { kill -TERM $(jobs -p) 2>/dev/null; [ -n "$SPID" ] && kill -TERM -- -$SPID 2>/dev/null; true; }
trap cleanup EXIT INT TERM
echo "=== $EVAL_ID config=$CONFIG shards=$SHARDS chunk=${CHUNK_SIZE:-1} open_loop=${OPEN_LOOP:-0} render=${RENDER:-config} gpus=[${GPUS[*]:-}] node=$(hostname) cpus=${SLURM_CPUS_ON_NODE:-?} $(date -Is)"

if [ -z "${SERVER_URL:-}" ]; then
  SERVER_URL=ws://127.0.0.1:$PORT
  VLA_EVAL_OPEN_LOOP_CHUNKS=${OPEN_LOOP:-0} setsid "$PY" "$HERE/dummy_server.py" --action-dim "$ACTION_DIM" --port "$PORT" --chunk-size "${CHUNK_SIZE:-1}" \
    ${HOLD_KEY:+--hold-key "$HOLD_KEY"} --obs-params "${OBS_PARAMS:-{\}}" > "$OUT/logs/server.log" 2>&1 &
  SPID=$!
  for _ in $(seq 60); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; kill -0 $SPID 2>/dev/null || break; sleep 1; done
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "server failed"; tail -20 "$OUT/logs/server.log"; exit 1; }
fi

# job load: epoch, cumulative CPU seconds and current/peak memory of this job's cgroup, GPU util/mem
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1)
(while :; do echo "$(date +%s) cpu_s=$(awk '/usage_usec/{print int($2/1e6)}' $CG/cpu.stat 2>/dev/null)" \
  "mem_gb=$(( $(cat $CG/memory.current 2>/dev/null || echo 0) >> 30 )) peak_gb=$(( $(cat $CG/memory.peak 2>/dev/null || echo 0) >> 30 ))" \
  "io_some10=$(awk '/^some/{split($2,a,"=");print a[2]}' /proc/pressure/io 2>/dev/null)" \
  "gpu=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr '\n' ';')"
  sleep 10; done) > "$OUT/logs/load.log" 2>&1 &
MON=$!

T0=$(date +%s); FAILED=0; pids=()
for i in $(seq 0 $((SHARDS - 1))); do
  genv="CUDA_VISIBLE_DEVICES="; [ ${#GPUS[@]} -gt 0 ] && genv="CUDA_VISIBLE_DEVICES=${GPUS[$((i % ${#GPUS[@]}))]}"
  env $genv "$VLA" run --yes --server-url "$SERVER_URL" --output-dir "$OUT" --eval-id "$EVAL_ID" ${RENDER:+--render $RENDER} \
    ${DEV:+--dev} ${RUN_ARGS:-} -c "$CONFIG" --shard-id "$i" --num-shards "$SHARDS" > "$OUT/logs/shard$i.log" 2>&1 &
  pids+=($!); sleep 0.3
done
for p in "${pids[@]}"; do wait "$p" || FAILED=$((FAILED + 1)); done
T1=$(date +%s)
echo "shards done in $((T1 - T0)) s, failed processes: $FAILED  $(date -Is)"
grep -h "^profile " "$OUT/logs/server.log" 2>/dev/null | tail -1 || true
"$PY" "$HERE/summarize.py" "$OUT" "$EVAL_ID" --shards "$SHARDS" --wall $((T1 - T0)) --render "${RENDER:-config}" --gpus "${#GPUS[@]}" --chunk "${CHUNK_SIZE:-1}" --open-loop "${OPEN_LOOP:-0}" | tee "$OUT/summary.txt"
[ "$FAILED" = 0 ]
