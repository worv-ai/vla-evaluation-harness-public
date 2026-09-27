---
name: add-benchmark
description: Add or adapt a simulation benchmark integration in vla-evaluation-harness, including its adapter, configs, and container definition.
---

# Add a benchmark

Work from the target harness checkout, not the installed skill directory. Read
`AGENTS.md`, `CONTRIBUTING.md`, and the closest existing benchmark and config.
Infer requirements from the simulator's source/docs; ask only for missing choices.

- Implement `src/vla_eval/benchmarks/<name>/benchmark.py` using `StepBenchmark`
  from `src/vla_eval/benchmarks/base.py`. Read its current abstract methods and
  `StepResult` contract instead of copying an old scaffold. Keep heavy simulator
  imports lazy so config discovery works on the host.
- Adapt camera images, language, state, action units/frame, and gripper convention
  explicitly. Declare observation/action specs and metric aggregation. Set episode
  limits via `get_metadata()`; reuse environments only if resets are reliable.
- Add configs under `configs/benchmarks/<name>/`, following `configs/README.md`.
  Use a `module:Class` import string and constructor arguments in `params`.
- Add `docker/Dockerfile.<name>` and register it in `docker/build.sh` and
  `docker/push.sh`. Follow the nearest simulator's rendering setup. For CPU
  rendering, read `docs/render-backends.md` before declaring support.
- For licensed assets, use `vla_eval.dirs.ensure_license` and `assets_cache`;
  follow the Behavior1K adapter and its matching host/container volume paths.

Run `make check`, `make test`, and `uv run vla-eval test --validate`.
When container execution is authorized and its image is available, smoke-test
with `uv run vla-eval test -c configs/benchmarks/<name>/<config>.yaml --dev`.
Use actual simulator smoke tests rather than mocked simulator packages. Report
unavailable runtime checks explicitly. Update the relevant config/reproduction docs.
