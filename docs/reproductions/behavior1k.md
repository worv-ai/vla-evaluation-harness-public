# BEHAVIOR-1K — Reproduction Status

[Challenge site](https://behavior.stanford.edu/challenge/) |
[Evaluation rules](https://behavior.stanford.edu/challenge/evaluation.html) |
[2026 leaderboard](https://huggingface.co/spaces/behavior-1k/2026-challenge-leaderboard) |
100 long-horizon household tasks (B100) on R1Pro / OmniGibson

> **Protocol version:** the default is now the 2026 challenge stack: BEHAVIOR-1K `v3.9.3-post2`,
> Isaac Sim 5.1.0, Python 3.11, B100. The 2025 stack (v3.7.2, Isaac Sim 4.5.0, B50) is no longer
> supported, and **the 2025 results previously published on this page do not apply to the current
> adapter** (different task set, instance split, assets, simulator and timeouts). They are kept below
> only as an archive.

## Status

| Check | State |
|---|---|
| Adapter, Docker recipe, configs on the 2026 stack | ✅ |
| Image builds (Charliecloud) and imports OmniGibson | ✅ 2026-10-30 |
| End-to-end run in Isaac Sim (demo replay) | ✅ renders on H100; 1/2 demos succeed (below) |
| Zero-action baseline run | ✅ 0/2, as expected |
| Trained-VLA reproduction against published 2026 scores | ⬜ none published yet |

## Protocol (2026)

The adapter drives upstream's own `omnigibson.eval.evaluator.BatchedEvaluator` with one logical
environment per task, so scene setup, instance loading, timeouts and metrics are upstream's code; only
the policy transport differs (harness WebSocket instead of upstream's openpi-style server).

| | 2025 stack (dropped) | 2026 stack (default) | Upstream source (v3.9.3-post2) |
|---|---|---|---|
| Tasks | B50 | B100 (`B100_task_misc.csv`) | `omnigibson/eval/utils/eval_utils.py` |
| Instances | TRO instances 1-10 (train ids) | test ids 301-340; public 301-320, hidden 321-340; report indices 0-9, 1 rollout each | `eval_utils.py`, `evaluator.py:resolve_instance_ids`, `docs/challenge/evaluation.md` |
| Timeout | 5000 steps or config | 1.5x mean human-demo length per task | `evaluator.py:load_env` (`EVAL_TIMEOUT_MULTIPLIER`) |
| Scene | task room types | exact room instances from `B100_task_misc.csv` | `evaluator.py:load_env` |
| Robot | `generate_robot_config` (joylo) | `omnigibson/eval/r1pro.yaml`; base mass 250, head aperture 40 | `evaluator.py:_apply_robot_eval_settings` |
| Cameras | 720/480 RGB | `DefaultWrapper` 224x224 RGB (CLI default) or `RGBDFullResWrapper` 720/480 RGB-D (official track) | `omnigibson/eval/wrappers/` |
| Metric | binary success | success + `q_score` (newly satisfied goal predicates, max over options; 1.0 on success) + efficiency | `omnigibson/metrics/task_metric.py`, `agent_metric.py` |
| Language | humanized task name | dataset instruction (`meta/tasks.jsonl` of `2026-challenge-demos`) | — |

The 23-D action layout and controllers are unchanged from 2025: `base[0:3]` holonomic velocity in
[-1, 1], `torso[3:7]`, `left_arm[7:14]`, `left_gripper[14:15]`, `right_arm[15:22]`,
`right_gripper[22:23]` (absolute joint positions).

## Smoke results (2026 stack)

Run on 2026-10-30 with Charliecloud (`--runtime charliecloud`, SquashFS image with the host driver
580.95.05 injected) on one H100 80GB. Isaac Sim 5.1 starts headless and renders through the
Vulkan ICD; the first episode of a fresh process takes ~7-9 min (boot + scene load); stepping then
runs at ~7-8 steps/s with `DefaultWrapper`.

| Run | Task | Instance | Steps | `success` | `q_score` |
|---|---|---|---:|:---:|---:|
| Demo replay, `demo_replay.yaml` (episode 0, 1956 steps) | `turning_on_radio` | train 1 | 1360 | ✅ | 1.0 |
| Demo replay, `demo_replay.yaml` (episode 200, 7667 steps) | `picking_up_trash` | train 1 | 7902 (timeout) | ❌ | 0.0 |
| Zero action, `RGBDFullResWrapper`, `max_steps: 300` | `turning_on_radio` | public 0 (301) | 300 | ❌ | 0.0 |
| Zero action, `RGBDFullResWrapper`, `max_steps: 300` | `turning_on_radio` | public 1 (302) | 300 | ❌ | 0.0 |

The `turning_on_radio` replay succeeds 4 steps earlier than the 2025 stack did (1364). The
`picking_up_trash` replay does not reach any goal predicate; whether that is replay drift over a
7.7k-step trajectory or an env difference is not yet diagnosed, so one passing demo is the only
end-to-end correctness evidence so far. Known issue: the container process segfaults during
interpreter teardown (Isaac Sim is left running, see `cleanup`), after results are written, so
`vla-eval run` exits non-zero (245) on an otherwise complete run.

## Published reference scores

None for 2026 yet: the leaderboard's public `data/results.jsonl` was empty on 2026-10-30 and upstream
publishes no baseline scores. The 2025 leaderboard (B50) is not comparable to B100.

## How to run

```bash
# 1. Build the image (NVIDIA Omniverse EULA: https://docs.omniverse.nvidia.com/eula/).
docker/build.sh behavior1k --accept-license behavior1k
#    Charliecloud: ch-image build -t <tag> --build-arg ACCEPT_NVIDIA_EULA=YES -f docker/Dockerfile.behavior1k .
#    Keep CH_IMAGE_STORAGE on local disk; the image tree is ~200k files.

# 2. Start an R1Pro model server, e.g. the zero-action baseline.
uv run --script src/vla_eval/model_servers/behavior1k_baseline.py --port 8000

# 3. Run.  The first run downloads ~32 GiB (behavior-1k-assets 3.9.0, robot assets 3.8.2, 2026 task
#    instances; ~180k files) into the behavior1k asset cache (see vla_eval.dirs).
uv run vla-eval run -c configs/benchmarks/behavior1k/eval.yaml --accept-license behavior-dataset-tos --yes
```

`eval.yaml` is the challenge self-evaluation (100 tasks x 10 public-test instances). Upstream reports
scene loads of 150-300 s per task and ~20 FPS stepping on an RTX 4090.

## Archive: 2025 stack results (do not apply to the current default)

Measured in 2026-04 with BEHAVIOR-1K v3.7.2 / Isaac Sim 4.5.0 on A100:

- Demo replay of `turning_on_radio` instance 1 (`2025-challenge-demos` episode 00000010, 1956 steps):
  success after 1364 steps
  ([JSON](data/behavior1k_demo_replay_turning_on_radio_inst1.json)).
- Zero-action baseline on `turning_on_radio`, 100 steps: 0% success
  ([JSON](data/behavior1k_baseline_zero_action_turning_on_radio.json)).
