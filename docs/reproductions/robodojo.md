# RoboDojo — Reproduction Status

[Repo](https://github.com/RoboDojo-Benchmark/RoboDojo) |
[Paper](https://arxiv.org/abs/2607.04434) |
[Leaderboard](https://robodojo-benchmark.com/leaderboard) |
42 bimanual simulation tasks (dual ARX-X5, Isaac Lab) across five capability dimensions

## Status

**Integration:** ✅ Benchmark + configs + Docker recipe + smoke pass (echo server, result JSON verified).
**Trained-VLA reproduction:** 🚧 π₀.₅ (officially released leaderboard checkpoint, seed 0), Memory
dimension (6 tasks × 50 episodes) — run in progress.

A probe already produced a real success on `classify_objects` (score 1.0, terminated early at
774/1100 steps), with the recorded video showing purposeful bimanual sorting — the observation
mapping, chunked inference, action encoding, and native reward path are all exercised end to end.

## Published protocol (leaderboard freeze 2026-07-03)

- 42 tasks × 50 episodes (Generalization: 25 standard + 25 `_random`), 3 training seeds per
  policy. Report mean ± std of score / success rate per dimension; overall = mean of the five
  dimension means (not the per-task mean).
- Metrics: `success` (binary, native reward manager) and `score` (1.0 on success, else the task's
  partial progress). Both are reported ×100.
- π₀.₅ reference row (score / SR, 3-seed mean): Generalization 13.37 / 8.17%, Precision
  12.40 / 5.50%, Long-Horizon 23.54 / 14.67%, **Memory 5.78 / 4.56%**, Open 1.98 / 1.67%,
  Average 11.41 / 6.91%. (Appendix K: fine-tuned from `pi05_base`, batch 256, 60K steps.)

## Reproduction setup

- Checkpoint: `ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999` from the
  `RoboDojo-Benchmark/RoboDojo` HF dataset (all evaluated policies release their fine-tuned
  checkpoints there; π₀.₅ ships all three seeds).
- Server: `configs/model_servers/robodojo_pi05/pi05.yaml` — OpenPI direct inference through
  XPolicyLab's openpi fork (which carries the RoboDojo train configs), reproducing the upstream
  input contract (3 CHW cameras, 14-D packed qpos, instruction prompt, open-loop action chunk).
- Deviation from the published protocol: one training seed (0) instead of three, so no std is
  reported. Everything else (layout groups, episode counts, step limits, metrics) follows the
  official protocol.

## Integration findings (validated empirically)

- **One task per process.** In principle Isaac Lab supports teardown-and-recreate
  (`DirectRLEnv.close()` → `SimulationContext.clear_instance()`, then build a new env). In
  practice `close()` hangs: measured on a *clean* `stack_blocks` scene (dual-X5, `camera_config`,
  Isaac Sim 5.1), `close()` did not return within 180 s, so `clear_instance()` is never reached
  and the next `create_eval_env` raises `RuntimeError: Simulation context already exists`. So a
  task switch inside a run is not viable (upstream `eval_policy.sh` is one-task-per-process for the
  same reason). `scripts/run_robodojo_protocol.sh` drives one `vla-eval run` per task.
- **`utils` namespace collision.** RoboDojo and XPolicyLab both ship a top-level `utils` package,
  and XPolicyLab's `load_file` is a subset (no `load_object_metadata` / `load_desc_info` /
  `load_pkl`). The image bakes `PYTHONPATH=/workspace/RoboDojo`, so a naive skip-if-present
  `sys.path` insert leaves the RoboDojo root *behind* XPolicyLab and layout loading dies with a
  confusing `NameError`. The adapter force-repositions both roots and verifies the resolution.
- **Bad layouts are skipped, not counted.** Some published layouts fail to build or settle
  (e.g. `press_by_number` layouts 1–3). The groups ship 55–65 layouts for 50 counted episodes;
  the adapter consumes them in order and skips failures, recording each episode's `layout_id`.
  Never `close()` the env to recover: teardown destroys the cameras and the next `reset()` dies
  in `init_cameras`.
- **cuRobo planners are unusable from the public assets** (only `curobo_tmp.yml` templates ship),
  and joint-space VLA evaluation does not need them — including for the Franka competition tasks,
  whose support arm replays recorded trajectories.
- **Throughput is the binding constraint.** Single-env stepping runs at ~30–50 steps/min
  (sim + 3-camera render + chunked inference), and failed episodes run to the task's `step_lim`
  (300–1900), so one task ≈ 12–20 GPU-hours and the Memory dimension ≈ 100 GPU-hours. Isaac
  renderers do not share a GPU gracefully: 4 lanes on one A100 collapsed per-lane throughput
  ~30× (aggregate ~8× *worse* than a single lane). Run one lane per GPU. The paper's own answer
  is batched parallel simulation (`num_envs > 1`, tiled rendering), which the harness cannot
  drive today — see below.
- **Hardware: A100 works, H100 does not.** On H100 (Hopper) the RTX renderer crashes the GPU
  (`ERROR_DEVICE_LOST`, GPU crash dump) as soon as a render product is read, so every episode
  errors out at the first observation. A100 renders fine (with a benign "DLSS-RR not supported"
  warning). Plan capacity on RTX-capable or A100-class GPUs.
- **Vulkan ICD pinning.** The upstream image hard-pins `VK_ICD_FILENAMES` *and* `VK_DRIVER_FILES`
  to `/etc/vulkan/icd.d/nvidia_icd.json`, which only exists if the host's container toolkit
  injects it. Where it doesn't, Vulkan dies with `ERROR_INCOMPATIBLE_DRIVER`. The configs
  override both to the image's own `/usr/share/vulkan/icd.d/nvidia_icd.json`.
- **Docker ignores Slurm's GPU cgroup.** `--gpus device=<index>` resolves against the *host* GPU
  list, so a container can grab a GPU allocated to another job. Under Slurm, always map by the
  UUIDs from `nvidia-smi` inside the allocation (`--gpus device=<UUID>`).

## Known gaps / future work

- **Batched evaluation.** RoboDojo's `EvalEnv` supports `num_envs > 1` with tiled rendering and
  XPolicyLab exposes `update_obs_batch` / `get_action_batch`. Exploiting it needs a vectorized
  benchmark interface in the harness (one benchmark instance driving N concurrent episodes).
  That is the single biggest available speedup for this benchmark.
- **Competition-task metric deviation.** The 3 `dual_x5_and_franka_competition` tasks
  (`imitate_sorting_sequence`, `make_kong`, `play_tic_tac_toe`) can mark an episode "unstable"
  when the scripted Franka move fails; upstream *excludes* those from the denominator, but
  vla-eval's collector counts every episode, so they score as failures here (slightly pessimistic).
- **Isaac Sim crashes are unrecoverable mid-task.** Some tasks (e.g. `press_by_number`,
  `swap_blocks`) occasionally hit an Isaac breakpad crash that takes down the process; the task's
  aggregate is then incomplete and must be re-run. `run_robodojo_protocol.sh` runs one process per
  task, so a crash is isolated to that task.
- Only the Memory dimension is reproduced so far; the other four dimensions use the same
  machinery and configs (`configs/benchmarks/robodojo/eval.yaml` covers all 42 tasks).

## License rationale (NO_REDIST)

The image builds on RoboDojo's official Isaac Sim 5.1 image and therefore bundles NVIDIA
Omniverse/Isaac Sim binaries (NVIDIA EULA) — same rationale as `behavior1k`. RoboDojo itself is
non-commercial research licensed, and the eval assets are distributed separately (HF dataset
`RoboDojo-Benchmark/RoboDojo`) and mounted at runtime. Build locally, do not push.
