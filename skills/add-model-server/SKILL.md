---
name: add-model-server
description: Integrate a VLA model into vla-evaluation-harness with an isolated model server, checkpoint configs, and inference validation.
---

# Add a model server

Work from the target harness checkout, not the installed skill directory. Read
`AGENTS.md`, `CONTRIBUTING.md`, `configs/README.md`, and a comparable implementation
under `src/vla_eval/model_servers/` before choosing the adapter.

- Use `PredictModelServer` for blocking inference; use `ModelServer` directly for
  custom asynchronous protocols. Inspect the current base classes and `serve.py`.
- Create a PEP 723 uv script. Point `vla-eval` to the correct relative checkout with
  `editable = true`, pin upstream git dependencies to commits, and set
  `exclude-newer` to the dependency verification date. Keep model-only dependencies
  isolated from the harness environment.
- Load weights in `__init__`; implement `predict(obs, ctx)` returning `actions` as
  an array shaped `(action_dim,)` or `(chunk_size, action_dim)`. Verify image order,
  state normalization, action denormalization, units/frame, and gripper convention.
  Declare action/observation specs; request wrist/state inputs through
  `get_observation_params()` when needed.
- Use `run_server(MyModelServer)` for the CLI. Add YAMLs under
  `configs/model_servers/<name>/`; share settings with `extends: _base.yaml`.
- Reset policy history and action queues in `on_episode_start`, then await the
  superclass hook. Keep state per session for concurrent clients; `session_id`
  can span episodes. Implement `predict_batch` before enabling batch size > 1.

Run `make check`, `make test`, and `uv run vla-eval test --validate`.
With allocated GPU resources and weights, run
`uv run vla-eval test -c configs/model_servers/<name>/<config>.yaml`.
Use real inference smoke tests rather than mocked model packages; report skipped
runtime checks. Document the checkpoint, supported benchmark, and reproduction caveats.
