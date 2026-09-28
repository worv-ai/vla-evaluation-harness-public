---
name: add-benchmark
description: Add or improve a simulation benchmark in vla-eval, including evaluation protocol, assets, adapter lifecycle, and runtime validation.
---

# Add a benchmark

Work from the target harness checkout, not the installed skill directory. Read
`AGENTS.md`, `CONTRIBUTING.md`, and the nearest benchmark's implementation and
config. Paths below refer to that checkout. For an existing integration, focus
on the affected sections and preserve unrelated behavior.

## 1. Evaluation protocol

Trace the upstream evaluator from task selection through initialization, rollout,
success checking, and cleanup. Establish:

- Version, task set, variations, initial-state selection, and seed handling.
- Robot/controller, action representation, control frequency, and observations.
- Warm-up, task-specific horizons, termination, success criteria, and aggregation.

Inspect helper calls for side effects: expert rollouts may both select solvable
initial states and initialize fields later read by success checks. Preserve the
required behavior without blindly repeating expensive work. Bound setup retries
and expose failures instead of silently changing the evaluated population.

Keep upstream defaults and intentional deviations distinguishable in configs and
documentation. Do not substitute a convenient task set or success condition.

## 2. Runtime and assets

Separate simulator dependencies, reusable assets, and generated working files.
Put simulator dependencies in `docker/Dockerfile.<name>` and register the image
in the existing build/push tooling. Pin compatible upstream revisions and verify
what is actually imported after dependency resolution.

For external scenes/data, document the source, version, acquisition path, and
expected contents. Use existing download mechanisms and `vla_eval.dirs` helpers
rather than introducing another asset-management layer.

- Harness-managed assets use `assets_cache("<benchmark>")`. The root is
  `VLA_EVAL_ASSETS_CACHE`, otherwise `VLA_EVAL_HOME/assets`, otherwise
  `${XDG_CACHE_HOME:-~/.cache}/vla-eval/assets`; inspect `src/vla_eval/dirs.py`.
- Config volume mounts must connect that host directory to the simulator's
  expected container path. Document both paths and overrides.
- Keep immutable source assets separate from writable generated files. Libraries
  may write beside their installed assets; check the intended runtime, including
  read-only images, and provide writable scratch space where needed.
- Reuse complete cached assets. For lazy acquisition, coordinate initialization
  or warm the cache before parallel workers can see partial files.
- Use `ensure_license` for gated assets and document the required acceptance.

Follow `docs/render-backends.md` for rendering setup. Declare a backend only after
checking real camera output in the image, not merely successful environment creation.

## 3. Benchmark adapter

Implement `StepBenchmark` under `src/vla_eval/benchmarks/<name>/`; inspect the
current base-class contracts. Keep heavy simulator imports lazy so host-side
config discovery does not require the simulator.

Add configs under `configs/benchmarks/<name>/` following `configs/README.md`.
Expose protocol choices as meaningful constructor/config parameters.

- Map simulator observations to named cameras, language, and optional state.
  Preserve image content and distinguish render resolution from policy-specific
  resize/crop, which belongs with model preprocessing.
- Support model-requested inputs through the existing observation negotiation.
  Check that wrist/state requests survive wrapper constructors and reach the
  server; explicit benchmark config overrides take precedence.
- Map actions explicitly: controller, absolute/delta convention, coordinate frame,
  rotation representation, units, and gripper semantics. Assign each conversion
  to one layer to avoid applying it twice.
- Declare observation/action specs and metric aggregation. Supply recording data
  through the harness; keep recording policy out of the adapter.

## 4. Reset, termination, and episode identity

Match the upstream environment lifecycle. Choose reuse or reconstruction based
on reset semantics, simulator resource behavior, and task requirements. Define
cleanup after normal completion and after an interrupted or errored episode.

Preserve task/variation/episode identity independently of shard assignment.
Verify representative initial observations and states across resets: setting one
RNG does not guarantee that robot, object, and scene initialization all repeat.
Document any remaining nondeterminism.

Reconcile task-specific limits with the harness runner cap so the runner cannot
truncate valid episodes. Keep simulation-step limits distinct from wall-clock
watchdogs for initialization and stalled execution. Preserve the original success
predicate and distinguish task failure from infrastructure error.

## 5. Validation and handoff

Run `make check`, `make test`, and `uv run vla-eval test --validate`. Confirm the
new configs appear in the smoke inventory. When container execution is authorized,
smoke-test the real image using the relevant config; use `--dev` for local adapter
changes or rebuild the image.

Choose cases that exercise different task/setup paths, not just the first task:

- Inspect real images and state; black, stale, or incorrectly oriented frames can
  pass shape checks and episode-level smoke tests.
- Compare initialization, action effects, and termination with upstream behavior.
  Use a released baseline or demonstration replay when available.
- Run consecutive episodes to expose reset leaks and resource accumulation;
  check task coverage and identity under sharding when relevant.

Report config validation, runtime checks, and performance reproduction separately.
A mock simulator or echo-policy run cannot establish simulator fidelity. Document
asset setup, a runnable config, supported runtime/rendering, deviations, and checks
that remain unavailable.
