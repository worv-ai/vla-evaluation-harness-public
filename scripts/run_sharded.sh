#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") -c <config> [-n <num_shards>] [-o <output>] [-d <daemon_url>] [--no-daemon]

Run a benchmark in parallel shards. Spawns a recording-daemon sidecar
that collects per-episode artefacts and writes the aggregate result.

Options:
  -c <config>          Config YAML file (required)
  -n <num_shards>      Number of shards (default: 50)
  -o <output>          Output directory for recording-daemon (default: results/recording-<config_name>/)
  -d <daemon_url>      WS URL for daemon (default: ws://127.0.0.1:9001)
  --no-daemon          Don't spawn the recording daemon (skip daemon-based collection)
  -h                   Show this help
EOF
  exit "${1:-0}"
}

CONFIG=""
NUM_SHARDS=50
OUTPUT=""
DAEMON_URL="ws://127.0.0.1:9001"
NO_DAEMON=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c) CONFIG="$2"; shift 2 ;;
    -n) NUM_SHARDS="$2"; shift 2 ;;
    -o) OUTPUT="$2"; shift 2 ;;
    -d) DAEMON_URL="$2"; shift 2 ;;
    --no-daemon) NO_DAEMON=1; shift ;;
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

if [[ -z "$OUTPUT" ]]; then
  config_name="$(basename "$CONFIG" .yaml)"
  config_name="$(basename "$config_name" .yml)"
  OUTPUT="results/recording-${config_name}/"
fi

mkdir -p "$OUTPUT"

EVAL_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"
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
echo "Daemon URL: $DAEMON_URL (no-daemon=$NO_DAEMON)"
echo ""

if [[ "$NO_DAEMON" -eq 0 ]]; then
  echo "Starting recording daemon..."
  vla-eval recording-daemon --bind "$DAEMON_URL" --out-dir "$OUTPUT" &
  DAEMON_PID=$!
  # Give the daemon a moment to bind before shards start dialing it.
  sleep 1
fi

echo "Launching ${NUM_SHARDS} shards..."

DAEMON_ARGS=()
if [[ "$NO_DAEMON" -eq 0 ]]; then
  DAEMON_ARGS=(--recording-daemon-url "$DAEMON_URL" --eval-id "$EVAL_ID")
fi

pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  vla-eval run -c "$CONFIG" --shard-id "$i" --num-shards "$NUM_SHARDS" "${DAEMON_ARGS[@]}" &
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

if [[ "$NO_DAEMON" -eq 0 ]]; then
  echo "Sending EVAL_END to recording daemon..."
  python3 -c "
from vla_eval.recording_daemon.client import RecordingClient
import time
e = RecordingClient('$DAEMON_URL')
e.end_eval('$EVAL_ID')
time.sleep(1.0)  # let the queue drain over WS
e.close()
"
  echo "Done. Per-episode artefacts and aggregate JSON in $OUTPUT"
fi

if [[ "$failed" -gt 0 ]]; then
  exit 1
fi
