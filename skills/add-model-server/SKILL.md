---
name: add-model-server
description: Add or improve a VLA model integration in vla-eval, covering checkpoint setup, inference parity, session state, and serving validation.
---

# Add a model server

Work from the target harness checkout, not the installed skill directory. Read
`AGENTS.md`, `CONTRIBUTING.md`, `configs/README.md`, and a comparable server.
Paths below refer to that checkout. For an existing integration, focus on the
changed behavior rather than rebuilding the entire adapter.

## 1. Integration scope

Identify the checkpoint, upstream inference/evaluation entrypoint, and intended
embodiment/benchmark. Establish required cameras/state, action format, and temporal
behavior; checkpoints in one model family can use different conventions.

First check whether an existing bridge, such as LeRobot, supports the policy.
Choose a config, a focused bridge extension, or a dedicated server according to
what is missing. Avoid duplicating a working inference path solely to add a model.

Use `PredictModelServer` for blocking inference and `ModelServer` for custom async
behavior. Inspect current interfaces and use `run_server()` for the CLI.

## 2. Dependencies and model assets

Resolve all inference inputs: weights/revision, architecture config, tokenizer,
image processor, adapters, and state/action normalization statistics. Identify
checkpoint-specific dataset keys and auxiliary models.

- Use the upstream Hub cache or explicit local checkpoint paths. Document download
  commands and overrides; do not copy weights into the repository.
- Check the requested files and revision, not merely whether any snapshot exists.
  Fail clearly on a missing suite/checkpoint instead of choosing another one.
- Expose supported local processor/tokenizer overrides when saved configs refer to
  inaccessible locations. Do not replace required inputs with guessed defaults.
- Isolate model dependencies in a PEP 723 uv script, following existing scripts for
  the editable harness path and verified dependency pins. Verify actual imports
  and inference in that environment; host tests do not exercise it.
- Prefer installable upstream packages. If packaging is broken or unavailable,
  follow the existing pinned-clone/cache pattern. Never clone over a user-provided
  checkout, and do not assume transitive installation preserves a requested pin.

Load the policy once during initialization. Add checkpoint configs under
`configs/model_servers/<name>/`, sharing common settings through `extends`.

## 3. Inference adapter

Trace one observation through preprocessing, policy inference, action decoding,
and the final action consumed by the benchmark.

Declare specs and required observation parameters, then verify that real benchmark
observations supply them. Cover camera order, language formatting, state fields,
and checkpoint-specific normalization. Do not pad or fabricate missing inputs just
to pass a smoke test unless that behavior is part of the upstream policy.

Assign each transformation to one layer:

- Image resize/crop, aspect ratio, channel order, and pixel normalization.
- State ordering, coordinate frame, and normalization.
- Action decoding/denormalization, absolute versus delta control, rotation format,
  units, and gripper convention.

Use the upstream processor where appropriate. Inspect whether the policy already
resizes or normalizes inputs before adding another transform. Return actions in
the current harness contract, including chunk shape where supported.

## 4. State and concurrency

Identify who owns observation history, recurrent state, action queues, and temporal
ensembling. Preserve the intended replanning frequency. Policies that consume every
executed-step observation may not support skipping calls through cached chunks.

Keep mutable state isolated per session. A session spans multiple episodes: reset
policy-owned state in `on_episode_start`, then await the superclass hook. The base
class clears its own buffers, not the policy's internal history. Clean up state at
session/episode end as appropriate without resetting other active sessions.

Enable batching only when the implementation supports it. Check request/output
ordering, per-session state, mixed episode lengths, and resets. Compare single
and batched inference with controlled inputs and randomness; do not assume bitwise
identity for stochastic or numerically different execution paths.

## 5. Validation and handoff

Run `make check`, `make test`, and config validation. Confirm the integration is
included in the smoke inventory, then exercise real checkpoint loading and inference
when weights and compute are available.

Compare upstream and adapter behavior on the same observation, controlling history
and sampling where possible. Compare intermediate tensors and decoded actions to
locate the first divergence; matching shapes alone is insufficient. Follow with a
short real rollout, consecutive episodes, and concurrent sessions where supported.
If both upstream and adapter fail, report the upstream limitation rather than
adjusting the adapter merely to approach a published score.

After correctness checks, measure warm inference latency, throughput, and memory
with representative inputs. Separate download/load/compile warm-up from steady
state. Record and validate changes to precision, inference steps, or action horizon;
they may alter the evaluated policy.

Document checkpoint acquisition, runnable configs, supported inputs/benchmarks,
state/batching behavior, and the validation actually performed. Distinguish loading,
inference, rollout, and reproduction support; report unavailable checks explicitly.
