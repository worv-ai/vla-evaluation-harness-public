# CLAUDE.md

This file provides context for AI coding assistants working on this repository.

## Project Overview

**vla-evaluation-harness** (`vla-eval`) is a unified evaluation framework for Vision-Language-Action (VLA) models across 11+ robot simulation benchmarks. Models integrate once, benchmarks integrate once, and the full cross-evaluation matrix works automatically.

Core design: Model server communicates with benchmark (Docker container, with optional GPU access for rendering) via WebSocket + msgpack binary protocol. This decouples model dependencies from benchmark dependencies entirely.

## Commands

```bash
# Setup
uv sync --python 3.11 --all-extras --dev

# Quality (CI runs these on every PR)
make lint          # ruff check --fix + ruff format
make check         # ruff check + ruff format --check + ty check (no auto-fix)
make test          # uv run pytest

# Single test
uv run pytest tests/test_protocol.py -v
uv run pytest tests/test_protocol.py::test_name -v

# Smoke tests (model servers, benchmarks, config validation)
vla-eval test                                       # validate configs only (fast, default)
vla-eval test --all                                 # run all categories (validate + server + benchmark)
vla-eval test --list                                # show available tests + prerequisites
vla-eval test --server                              # smoke-test all model servers
vla-eval test --benchmark                           # smoke-test all benchmarks
vla-eval test -c configs/model_servers/cogact.yaml  # smoke-test a specific config
make smoke                                          # shortcut for vla-eval test --all
```

Line length is **119** (configured in pyproject.toml for ruff and ty).

## Architecture

```
CLI (cli/main.py)
 └─ Orchestrator (orchestrator.py)
     ├─ Benchmark (benchmarks/base.py)  ── runs inside Docker container
     │   └─ EpisodeRunner (runners/)    ── sync or async (Sim2Live)
     │       └─ Connection (connection.py) ←─ WebSocket/msgpack ─→ ModelServer (model_servers/base.py)
     └─ ResultCollector (results/collector.py)  ── in-memory summary

recording_daemon/  ── optional sidecar (`vla-eval recording-daemon`); sole writer
                     of per-episode jsonl + per-eval aggregate JSON when enabled.
                     Three-message wire protocol (STEP / RESULT / EVAL_END).
                     Video lives entirely on the harness side via
                     vla_eval.benchmarks.video.EpisodeVideoRecorder — binary
                     never crosses the wire.
```

### Key design decisions

- **Episode-level error isolation**: One episode failure never aborts the entire evaluation.
- **anyio-based async**: asyncio-compatible, not trio. Use anyio primitives for new async code.
- **Parallel evaluation**: Environment parallelism via episode sharding + inference parallelism via batch forward passes.
- **Recording is opt-in via daemon URL**: without `--recording-daemon-url`, the orchestrator writes per-shard JSON itself (use `vla-eval merge` to combine). With a daemon URL, the daemon is the sole disk writer for jsonl + aggregate and there is no merge step.

### Recording daemon (opt-in)

`vla-eval recording-daemon --bind ws://0.0.0.0:9001` runs as a sidecar.
The orchestrator and (optionally) model-server-side code push three message
types to it:

- `STEP {sid, eid, step_id, fields}` — accumulates per-episode rows.
- `RESULT {eval_id, sid, eid, status, metrics, context, jsonl_path,
  aggregate_dir, aggregate_filename}` — orchestrator declares the episode
  done; daemon writes the jsonl at the supplied path and appends to the
  eval aggregate.
- `EVAL_END {eval_id}` — orchestrator declares the run done; daemon
  flushes the aggregate JSON.

Buckets and evals are created implicitly on first arrival — there is no
`EPISODE_START` or `EVAL_START`. Filename rendering is done by the
orchestrator before pushing `RESULT`, so the daemon stays
benchmark-agnostic.

Benchmarks talk to the daemon indirectly via the
:class:`vla_eval.recording.EpisodeRecorder` the orchestrator passes to
``Benchmark.start_episode(task, recorder=...)``. The recorder owns the
local video file (one mp4 per episode, written directly to its final
path) and forwards `record_step(...)` calls to the daemon.

Use `scripts/run_sharded.sh` to spawn the daemon + N shards in one go.

Read `CONTRIBUTING.md` before any integration work (adding benchmarks/model servers, recording, PR workflow).
