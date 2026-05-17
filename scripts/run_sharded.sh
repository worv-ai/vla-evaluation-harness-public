#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") -c <config> [-n <num_shards>] [-o <output>] [-d <daemon_url>] [--no-daemon]

Default flow (daemon mode):
  Spawn a recording-daemon sidecar, run N shards in parallel, and let the
  daemon write per-episode jsonl/mp4 + per-eval aggregate JSON. No merge step.

  Options:
    -c <config>          Config YAML file (required)
    -n <num_shards>      Number of shards (default: 50)
    -o <output>          Daemon output dir (default: results/recording-<config_name>/)
    -d <daemon_url>      WS URL for the daemon (default: ws://127.0.0.1:9001)
    --no-daemon          Legacy flow: shards write per-shard JSON; merge after.
                         -o then names the merged result file (default:
                         results/<config_name>.json).
    -h                   Show this help
EOF
  exit "${1:-0}"
}

CONFIG=""
NUM_SHARDS=50
OUTPUT=""
DAEMON_URL="ws://127.0.0.1:9001"
USE_DAEMON=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c) CONFIG="$2"; shift 2 ;;
    -n) NUM_SHARDS="$2"; shift 2 ;;
    -o) OUTPUT="$2"; shift 2 ;;
    -d) DAEMON_URL="$2"; shift 2 ;;
    --no-daemon) USE_DAEMON=0; shift ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown option: $1" >&2; usage 1 ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "Error: -c <config> is required." >&2
  usage 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Error: config file not found: $CONFIG" >&2
  exit 1
fi

config_name="$(basename "$CONFIG" .yaml)"
config_name="$(basename "$config_name" .yml)"

if [[ "$USE_DAEMON" -eq 1 ]]; then
  # ---- Daemon flow ----
  if [[ -z "$OUTPUT" ]]; then
    OUTPUT="results/recording-${config_name}/"
  fi
  mkdir -p "$OUTPUT"

  EVAL_ID="$(uuidgen 2>/dev/null || python3 -c 'import uuid; print(uuid.uuid4())')"
  DAEMON_PID=""

  cleanup() {
    echo "Cleaning up background processes..."
    if [[ -n "$DAEMON_PID" ]] && kill -0 "$DAEMON_PID" 2>/dev/null; then
      kill -TERM "$DAEMON_PID" 2>/dev/null || true
      wait "$DAEMON_PID" 2>/dev/null || true
    fi
    kill -- -$$ 2>/dev/null || true
  }
  trap cleanup EXIT

  echo "Config:     $CONFIG"
  echo "Shards:     $NUM_SHARDS"
  echo "Output:     $OUTPUT"
  echo "Eval ID:    $EVAL_ID"
  echo "Daemon URL: $DAEMON_URL"
  echo ""

  echo "Starting recording daemon..."
  vla-eval recording-daemon --bind "$DAEMON_URL" --out-dir "$OUTPUT" &
  DAEMON_PID=$!
  # Give the daemon a moment to bind before shards start dialing it.
  sleep 1

  echo "Launching ${NUM_SHARDS} shards..."
  pids=()
  for i in $(seq 0 $((NUM_SHARDS - 1))); do
    vla-eval run -c "$CONFIG" \
      --shard-id "$i" --num-shards "$NUM_SHARDS" \
      --recording-daemon-url "$DAEMON_URL" --eval-id "$EVAL_ID" &
    pids+=($!)
  done

  echo "Waiting for all shards to finish..."
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=$((failed + 1))
    fi
  done

  if [[ "$failed" -gt 0 ]]; then
    echo "ERROR: $failed of $NUM_SHARDS shards failed." >&2
  fi

  echo "Done. Per-episode artefacts and aggregate JSON in $OUTPUT"

  if [[ "$failed" -gt 0 ]]; then
    exit 1
  fi
  exit 0
fi

# ---- Legacy (no-daemon) flow ----
if [[ -z "$OUTPUT" ]]; then
  OUTPUT="results/${config_name}.json"
fi

cleanup_nodaemon() {
  echo "Cleaning up background processes..."
  kill -- -$$ 2>/dev/null || true
}
trap cleanup_nodaemon EXIT

echo "Config:  $CONFIG"
echo "Shards:  $NUM_SHARDS"
echo "Output:  $OUTPUT"
echo ""

existing=$(CONFIG="$CONFIG" NUM_SHARDS="$NUM_SHARDS" python3 -c "
import os, yaml, re
from pathlib import Path
with open(os.environ['CONFIG']) as f:
    cfg = yaml.safe_load(f)
num_shards = os.environ['NUM_SHARDS']
output_dir = Path(cfg.get('output_dir', './results'))
found = []
seen = set()
for b in cfg.get('benchmarks', []):
    name = b.get('name') or b['benchmark'].rsplit(':', 1)[-1]
    sub = b.get('subname')
    if sub:
        name = f'{name}_{sub}'
    safe = re.sub(r'[^\w\-.]', '_', name)
    if safe in seen:
        continue
    seen.add(safe)
    found.extend(output_dir.glob(f'{safe}_shard*of{num_shards}.json'))
if found:
    print(f'{len(found)} existing shard file(s) found, e.g.: {found[0]}')
")
if [[ -n "$existing" ]]; then
  echo "Error: $existing" >&2
  echo "Remove existing results or use a different output_dir." >&2
  exit 1
fi

echo "Launching ${NUM_SHARDS} shards..."
pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  vla-eval run -c "$CONFIG" --shard-id "$i" --num-shards "$NUM_SHARDS" &
  pids+=($!)
done

echo "Waiting for all shards to finish..."
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=$((failed + 1))
  fi
done

if [[ "$failed" -gt 0 ]]; then
  echo "ERROR: $failed of $NUM_SHARDS shards failed." >&2
  exit 1
fi

echo "Merging results..."
vla-eval merge -c "$CONFIG" -o "$OUTPUT"

echo "Done. Results saved to $OUTPUT"
