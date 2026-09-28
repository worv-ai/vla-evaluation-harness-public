---
name: run-evaluation
description: Run, debug, and scale VLA evaluations in vla-eval while preserving evaluation conditions and verifying complete results.
---

# Run an evaluation

Work from the target harness checkout. Read `AGENTS.md`, `configs/README.md`,
and the selected model/benchmark notes. Discover current config paths with
`rg --files configs/model_servers configs/benchmarks`. Use the sections relevant
to the requested run; an existing validated setup need not be re-integrated.

## 1. Evaluation plan

Identify the checkpoint/revision, benchmark version, task set, variations, episodes,
seeds, and initialization procedure. Check observation inputs, controller/action
format, horizon, and replanning settings against the intended comparison.

Record deliberate differences from the reference evaluation. Define the expected
workload, recording needs, and output directory before launching. A smoke run and
a full reproduction have different completion criteria.

Consult `docs/reproductions/running-guide.md` for examples and prior measurements;
its hardware allocations and benchmark-specific verdict thresholds are not universal.

## 2. Preflight

Verify the requested checkpoint and auxiliary files, benchmark assets, image, and
actual source code used by the run. Use `--dev` for local benchmark changes or
rebuild the image. Keep model dependencies in their isolated uv environment.

Inspect `src/vla_eval/dirs.py` and config mounts to resolve asset locations. Verify
paths on the machine that consumes them: the server and benchmark may not share a
filesystem. Reuse caches, arrange writable scratch/output locations, and complete
or coordinate acquisition before parallel launch. Check disk capacity, including
video/step recordings when enabled.

Use the machine's allocated CPUs/GPUs and permitted runtime; do not assume exclusive
host access. Read `docs/runtimes.md` or `docs/render-backends.md` when changing runtime
or renderer. Follow local authorization and scheduler policies.

Start the configured server and wait for HTTP 200 from `/health`. Check connectivity
from the benchmark runtime. Model loading and first-inference compilation are
separate phases; allow justified warm-up without hiding a stalled process.

## 3. Baseline run

Run a small representative workload before an unfamiliar full evaluation:

```bash
uv run vla-eval serve -c <model-config>
uv run vla-eval run -c <benchmark-config> \
  --server-url ws://<server-host>:<port> --output-dir <run-directory>
```

Select actual tasks/episode counts through supported config or CLI options. Inspect
images, state, actions, reset, termination, and recordings. Include multiple task
paths and consecutive episodes when those behaviors are unverified.

Diagnose failures in order: asset/environment setup → observation generation →
server preprocessing/inference → action application → success/termination → recording.
Compare with upstream at the first divergent boundary. A process exiting normally,
a low score, or matching tensor shapes does not by itself establish correctness.

For matched comparisons, inspect representative initial observations/states rather
than assuming equal seeds guarantee equal starts. Record known nondeterminism.

## 4. Throughput tuning

Optimize completed valid episodes per time/resource under the same evaluation
conditions. Read `docs/tuning-guide.md`, then inspect the current implementation
and CLI for supported controls.

### Measure the bottleneck

Separate cold start from steady state. Measure environment demand and server
capacity with representative inputs; `experiments/bench_demand.py` and
`experiments/bench_supply.py` provide dedicated probes. Observe rendering, CPU
contention, GPU memory, request latency, and recording I/O.

Distinguish observation requests from actual model forward passes when chunks are
cached. Account for environment setup and uneven task durations. Failed episodes
may run to the horizon, so a changed success mix can distort timing comparisons.

### Adjust concurrency

Increase independent environment shards and supported inference batch sizes
incrementally. Bound CPUs, GPUs, and thread counts to allocated resources; tune
batch waiting against latency. Leave capacity headroom and stop increasing workers
when valid-episode throughput plateaus or latency/errors grow.

Measure total wall time and worker imbalance, not just model throughput or fewer
environment builds. Inspect the checkout's work-distribution behavior; do not assume
shards have fixed episode counts or copy a previous machine's optimal settings.

### Preserve evaluation conditions

Check episode identity, reset, and session isolation under concurrency. Keep camera
inputs, precision, inference steps, action horizon, control frequency, and episode
limits unchanged unless the user requested a different evaluation.

For live or latency-sensitive evaluation, batching/serving delays are part of the
conditions. Verify the runner's timing and hold-action semantics before treating a
concurrency change as an equivalent faster run.

## 5. Full run and recovery

Prefer `scripts/run_sharded.sh` when it fits; inspect its current options. Shards
of one evaluation share an evaluation ID and recording destination. Independent
evaluations use distinct destinations. Shared recording requires a filesystem with
working SQLite locks and synchronization.

Monitor episode errors, worker exit statuses, and completion against the expected
workload. An episode-level error may leave the overall process running normally.

Before retrying, inspect completed/unfinished work and the current recording/queue
semantics. Establish how the retry selects work and handles existing records; do
not blindly rerun everything or combine potentially duplicated outputs. Preserve
logs needed to diagnose the failure. Stop only jobs/processes belonging to this run.

## 6. Results and reporting

The authoritative recording is `recording-<eval-id>.sqlite`. Single-shard runs
export automatically; sharded runs use the helper's export or:

```bash
uv run vla-eval export <recording.sqlite> -o <export-directory>
```

Before reporting scores, check expected versus recorded task/episode identities,
missing/duplicate/errored episodes, worker completion, and metric aggregation.
An export can be a valid snapshot of an incomplete run. Distinguish task failures
from infrastructure errors and disclose their treatment in the score denominator.

Report scores, episode/error counts, protocol deviations, checkpoint and code/image
provenance, resource use, and recording/log locations. Distinguish smoke validation
from reproduction evidence. These are executed evaluation results; the separate
paper-reported leaderboard is not their output destination.
