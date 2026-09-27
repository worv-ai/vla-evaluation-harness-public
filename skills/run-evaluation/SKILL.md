---
name: run-evaluation
description: Run or debug VLA model evaluations on simulation benchmarks with vla-evaluation-harness, including serving, sharding, and result export.
---

# Run an evaluation

Work from the target harness checkout. Read `AGENTS.md`, `configs/README.md`, and
`docs/reproductions/running-guide.md`, plus the selected model/benchmark notes.
Discover actual paths with `rg --files configs/model_servers configs/benchmarks`.
Select a compatible checkpoint, action format, and benchmark protocol.

## Launch

Check uv, weights/cache capacity, allocated GPUs, and permitted container runtime.
Follow the target machine's resource and container policies. Obtain any required
authorization before launching jobs or containers. Keep simulator/model dependencies
in isolated environments, not system Python.

Run the server and benchmark concurrently in managed processes/jobs:

```bash
uv run vla-eval serve -c configs/model_servers/<model>/<config>.yaml
# Wait for HTTP 200 from http://<server-host>:8000/health.
uv run vla-eval run -c configs/benchmarks/<benchmark>/<config>.yaml \
  --server-url ws://<server-host>:8000 --output-dir results/<run-name>
```

Use `--dev` for local source changes or rebuild the benchmark image before running.
For software rendering, consult `docs/render-backends.md`. Inspect the CLI help
for current options. Use separate output directories for independent evaluations.

## Parallel runs and results

Read `docs/tuning-guide.md` before choosing shard and batch counts. Prefer
`uv run bash scripts/run_sharded.sh -c <benchmark-config> -n <count> -o <output>`;
configure the server URL in that config. The helper shares an eval ID and exports
after shards finish. Manual shards must share `--eval-id` and output directory.

The authoritative recording is `recording-<eval-id>.sqlite`, shared by shards on
storage supporting SQLite locks and sync. Single-shard runs export automatically;
use `uv run vla-eval export <recording.sqlite> [-o <directory>]` to export again.
An export alone does not prove all episodes completed: check process exit status,
expected episode counts, and episode errors before reporting scores. Include config,
checkpoint, seed, protocol, success metrics, and failure counts in the result summary.
Stop only the processes/jobs launched for this run.

These are reproduced evaluation results. The separate `leaderboard/` site records
paper-reported scores; do not insert runtime results there.
