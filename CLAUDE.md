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
     └─ ResultCollector (results/collector.py)

recording_daemon/  ── optional sidecar; collects per-episode jsonl + mp4 + per-eval aggregate JSON over its own WS.
```

### Key design decisions

- **Episode-level error isolation**: One episode failure never aborts the entire evaluation.
- **anyio-based async**: asyncio-compatible, not trio. Use anyio primitives for new async code.
- **Parallel evaluation**: Environment parallelism via episode sharding + inference parallelism via batch forward passes.

### Recording daemon (optional)

`vla-eval recording-daemon` runs as a sidecar on the eval host. Model server (`--recording-daemon-url ws://...`) and harness shards (`--recording-daemon-url`, `--eval-id`) push per-step / per-episode artefacts to it; the daemon writes jsonl + mp4 + aggregate JSON.

Use `scripts/run_sharded.sh` to spawn the daemon, run N shards, and push EVAL_END automatically. `scripts/run_sharded.sh --no-daemon` disables the daemon flow if you only need per-shard JSON (legacy).

The daemon owns disk writes — local-mode `EpisodeRecorder` still works when no emitter is installed; daemon-mode kicks in once the orchestrator installs an emitter and calls `Benchmark.set_recording_target(sid, eid)`.

Read `CONTRIBUTING.md` before any integration work (adding benchmarks/model servers, PR workflow).
