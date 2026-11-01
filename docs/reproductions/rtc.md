# RTC on Kinetix — Reproduction Report

Real-Time Chunking (RTC) for action-chunking flow policies. [Paper](https://arxiv.org/abs/2506.07339) |
[GitHub](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) | 12 Kinetix levels, flow-matching MLP-Mixer, H = 8.

The goal is not only to reproduce Fig. 5. It is to show that vla-eval's **live** mode (wall-clock ticks, real plus injected
inference latency) gives the same solve rates as the paper's **sync** protocol, where the delay is counted in steps.

## Results Summary

| Arm | What | Status |
|-----|------|--------|
| R | Official `eval_flow.py`, run by us (2048 rollouts × 12 levels, `bc/31`, BID weak `bc/7`) | ✅ matches Fig. 5 (≤ 0.7 pp, BID ≤ 1.9 pp) |
| R′ | R with success read as Kinetix v3 reads it (see *Success metric*) | reference for C and L |
| C | vla-eval **sync**, server counts the delay in steps (2048 × 12) | ✅ 11/15 conditions pass both criteria vs R′; mean diff −0.2 pp |
| L | vla-eval **live**, 10 Hz, latency (d − ½)/hz | pending (see *Live arm*) |

### Links

| | |
|---|---|
| **Checkpoints** | `gs://rtc-assets/bc/31/policies` (strong), `bc/7` (BID weak), public HTTPS; see [`configs/model_servers/rtc/README.md`](../../configs/model_servers/rtc/README.md) |
| **Server** | [`src/vla_eval/model_servers/rtc.py`](../../src/vla_eval/model_servers/rtc.py) (`KinetixRTCServer`, imports the official `FlowPolicy`) |
| **Server configs** | [`configs/model_servers/rtc/`](../../configs/model_servers/rtc/) (`naive`, `rtc`, `rtc_hard`, `bid`) |
| **Benchmark configs** | [`eval.yaml`](../../configs/benchmarks/kinetix/eval.yaml) (sync), [`realtime.yaml`](../../configs/benchmarks/kinetix/realtime.yaml) (live) |
| **Results** | [`data/rtc-kinetix/`](data/rtc-kinetix/): `reference/` (R, R′ JSONL), `sync/` (C per condition), scripts |

## Protocol

The official protocol, from `eval_flow.py` @ `9296f31`, with Kinetix `cf7453e`:
- Chunk 0 comes from o₀ with no delay.
- Every s steps, a chunk is sampled from that step's observation. The previous chunk covers the next d steps, then the new chunk covers the rest of the s steps.
- Action noise N(0, 0.1²). The episode is capped at 256 steps. Only the first episode of each rollout counts.

`KinetixRTCServer` runs this same schedule per episode. It answers every observation with the plan's action for that step: a `reply` in sync, a `PUBLISH` in live. The step index, not a time grid, picks the slot. `tests/test_rtc_server.py` checks the executed actions against `eval_flow`'s, for six (d, s) pairs.

A boundary's chunk lands at one of three points, set by `latency`:
- `steps` (C): when step t + d's observation arrives.
- `wall` (L): (d − ½)/hz after its observation was admitted, compute included.
- `wall_exact`: d/hz after admission.

The latency is slept in the episode task, never on the model thread. Methods are naive, RTC soft (`exp`, β = 5), RTC hard (`zeros`), and BID (N = 32, K = 3, weak `bc/7`). Each method calls the official `action` / `realtime_action` / `bid_action` unchanged.

## Reference: R vs Fig. 5 (paper metric)

Each cell is R / Fig. 5, the 12-level average. The Fig. 5 numbers were extracted from the vector PDF, with a read error of about ±0.003. With 24 576 trials, the 95% CI of R's average is about ±0.4 pp.

d sweep, s = max(d, 1):

| method | d=0 | d=1 | d=2 | d=3 | d=4 |
|---|---|---|---|---|---|
| rtc | 0.922 / 0.922 | 0.896 / 0.893 | 0.847 / 0.843 | 0.743 / 0.747 | 0.612 / 0.618 |
| rtc_hard | 0.917 / 0.916 | 0.844 / 0.849 | 0.817 / 0.817 | 0.732 / 0.732 | 0.612 / 0.619 |
| bid | 0.916 / 0.921 | 0.883 / 0.877 | 0.817 / 0.801 | 0.692 / 0.673 | 0.558 / 0.552 |
| naive | 0.916 / 0.915 | 0.754 / 0.753 | 0.709 / 0.708 | 0.585 / 0.584 | 0.490 / 0.493 |

d = 1, s sweep:

| method | s=1 | s=2 | s=3 | s=4 | s=5 | s=6 | s=7 |
|---|---|---|---|---|---|---|---|
| rtc | 0.896 / 0.893 | 0.878 / 0.878 | 0.862 / 0.863 | 0.847 / 0.845 | 0.834 / 0.834 | 0.817 / 0.821 | 0.796 / 0.796 |
| rtc_hard | 0.844 / 0.848 | 0.867 / 0.866 | 0.855 / 0.856 | 0.849 / 0.850 | 0.833 / 0.830 | 0.819 / 0.821 | 0.796 / 0.796 |
| naive | 0.754 / 0.753 | 0.813 / 0.812 | 0.786 / 0.787 | 0.785 / 0.784 | 0.746 / 0.753 | 0.749 / 0.754 | 0.719 / 0.716 |

R reproduces Fig. 5 within 0.7 pp for RTC, hard masking, and naive. BID with the paper setting is 1.6–1.9 pp above the figure at d = 2 and d = 3. The released sweep default for BID (N = 16, no weak policy) gives 0.813/0.686/0.555 at d = 2/3/4 (paper setting: 0.817/0.692/0.558); see `reference/R_bid_default.jsonl`.

## Success metric: Kinetix v3 vs `cf7453e`

vla-eval pins Kinetix v3.0.0, while the paper used `cf7453e`. The two versions read a step's success differently, and that difference is in how the frame-skip substeps are combined. There are 2 substeps per step.
- `cf7453e`: the step reward is the max over its substeps, and `GoalR` is taken from the **last** substep. A goal touched on substep 1 and lost on substep 2 therefore counts as a failure.
- v3: the reward and `GoalR` come from the **first** substep that ends the episode.

The physics is the same. Replaying a fixed 256-action sequence in both versions gives the same observations to about 1e-6 on the non-chaotic levels; the chaotic ones diverge from float noise.

R itself shows the effect: under the official metric, the solve rate is below the return. Examples: cheetah solved 0.884 with return 0.994, swimmer 0.921 with 1.000, walker 0.806 with 0.953.

**R′** keeps the official script and seeds and changes only the logged success, to "the terminal step's reward is +1". That is v3's semantics, except for a step whose first substep is red and whose second is green, which R′ counts as solved. R′ is ≈ +4 pp over R on the level average; see `reference/R_*_v3metric.jsonl`. vla-eval cannot report the paper metric under v3, so C and L are compared against R′.

## Sync arm (C) vs R′

Settings: 2048 rollouts × 12 levels, `latency=steps`, CPU servers (jax 0.4.35, flax 0.10.2), Kinetix v3 image.

The criteria are fixed in advance:
- (1) The level-average difference lies within the 95% CI of the difference.
- (2) At most 2 of 12 levels lie outside their 95% CI (Newcombe), and no level is off by more than 5 pp.

| condition | n/level | ref | arm | diff [95% CI] | levels outside | worst level | (1) | (2) |
|---|---|---|---|---|---|---|---|---|
| naive d=0 s=1 | 2046 | 0.961 | 0.955 | -0.55 ± 0.33 pp | 1/12 | 2.5 pp | FAIL | pass |
| naive d=1 s=1 | 2037 | 0.795 | 0.804 | +0.90 ± 0.60 pp | 1/12 | 14.9 pp | FAIL | pass |
| naive d=2 s=2 | 2048 | 0.746 | 0.748 | +0.17 ± 0.67 pp | 0/12 | 1.5 pp | pass | pass |
| naive d=3 s=3 | 2048 | 0.613 | 0.612 | -0.05 ± 0.73 pp | 0/12 | 2.2 pp | pass | pass |
| naive d=4 s=4 | 2047 | 0.515 | 0.513 | -0.15 ± 0.74 pp | 0/12 | 1.6 pp | pass | pass |
| rtc d=0 s=1 | 2046 | 0.967 | 0.963 | -0.43 ± 0.30 pp | 2/12 | 3.1 pp | FAIL | pass |
| rtc d=1 s=1 | 2048 | 0.940 | 0.937 | -0.30 ± 0.38 pp | 1/12 | 1.9 pp | pass | pass |
| rtc d=2 s=2 | 2048 | 0.887 | 0.882 | -0.50 ± 0.51 pp | 2/12 | 2.4 pp | pass | pass |
| rtc d=3 s=3 | 2048 | 0.781 | 0.779 | -0.22 ± 0.67 pp | 0/12 | 2.1 pp | pass | pass |
| rtc d=4 s=4 | 2048 | 0.642 | 0.640 | -0.12 ± 0.73 pp | 1/12 | 3.3 pp | pass | pass |
| rtc_hard d=0 s=1 | 2048 | 0.961 | 0.956 | -0.43 ± 0.33 pp | 2/12 | 3.8 pp | FAIL | pass |
| rtc_hard d=1 s=1 | 2048 | 0.890 | 0.890 | -0.04 ± 0.48 pp | 1/12 | 3.3 pp | pass | pass |
| rtc_hard d=2 s=2 | 2048 | 0.857 | 0.853 | -0.41 ± 0.56 pp | 0/12 | 1.8 pp | pass | pass |
| rtc_hard d=3 s=3 | 2048 | 0.770 | 0.765 | -0.45 ± 0.68 pp | 1/12 | 1.9 pp | pass | pass |
| rtc_hard d=4 s=4 | 2048 | 0.643 | 0.637 | -0.53 ± 0.73 pp | 1/12 | 3.0 pp | pass | pass |

The instruments are exact in every C run: every boundary's effective delay equals d, inference happens every s steps, and the applied action equals the server's intended action at every step.

All four criterion-(1) failures are d = 0 / d = 1 conditions. There, many levels sit near 100% success, so the Wald CI of the average shrinks to about ±0.3 pp. The per-level criterion passes in all 15 conditions. Across all 15 conditions, C's mean difference is −0.2 pp, slightly negative. This is the direction you would expect from the residual metric case (red then green within one step, which R′ counts as solved).

The single large per-level outlier is trampoline at naive d = 1: R′ 0.488 vs C 0.637. In the same cell, R under the paper metric is 0.483, and 0.787 at d = 2. On trampoline, C is otherwise 1–4 pp below R′. This level is the main open item; see *Open issues*.

Against R with the paper metric, every C condition fails, with a mean difference of +2.3 to +5.0 pp:

| condition | n/level | ref | arm | diff [95% CI] | levels outside | worst level | (1) | (2) |
|---|---|---|---|---|---|---|---|---|
| naive d=0 s=1 | 2046 | 0.916 | 0.955 | +3.89 ± 0.41 pp | 8/12 | 13.8 pp | FAIL | FAIL |
| naive d=1 s=1 | 2037 | 0.754 | 0.804 | +5.00 ± 0.63 pp | 6/12 | 15.4 pp | FAIL | FAIL |
| naive d=2 s=2 | 2048 | 0.709 | 0.748 | +3.90 ± 0.69 pp | 6/12 | 12.6 pp | FAIL | FAIL |
| naive d=3 s=3 | 2048 | 0.585 | 0.612 | +2.75 ± 0.74 pp | 4/12 | 12.9 pp | FAIL | FAIL |
| naive d=4 s=4 | 2047 | 0.490 | 0.513 | +2.31 ± 0.75 pp | 4/12 | 11.3 pp | FAIL | FAIL |
| rtc d=0 s=1 | 2046 | 0.922 | 0.963 | +4.08 ± 0.39 pp | 9/12 | 14.9 pp | FAIL | FAIL |
| rtc d=1 s=1 | 2048 | 0.896 | 0.937 | +4.10 ± 0.45 pp | 7/12 | 12.0 pp | FAIL | FAIL |
| rtc d=2 s=2 | 2048 | 0.847 | 0.882 | +3.58 ± 0.56 pp | 7/12 | 11.6 pp | FAIL | FAIL |
| rtc d=3 s=3 | 2048 | 0.743 | 0.779 | +3.53 ± 0.69 pp | 7/12 | 13.3 pp | FAIL | FAIL |
| rtc d=4 s=4 | 2048 | 0.612 | 0.640 | +2.88 ± 0.75 pp | 6/12 | 12.8 pp | FAIL | FAIL |
| rtc_hard d=0 s=1 | 2048 | 0.917 | 0.956 | +3.98 ± 0.41 pp | 10/12 | 12.6 pp | FAIL | FAIL |
| rtc_hard d=1 s=1 | 2048 | 0.844 | 0.890 | +4.58 ± 0.53 pp | 8/12 | 12.6 pp | FAIL | FAIL |
| rtc_hard d=2 s=2 | 2048 | 0.817 | 0.853 | +3.61 ± 0.60 pp | 6/12 | 12.5 pp | FAIL | FAIL |
| rtc_hard d=3 s=3 | 2048 | 0.732 | 0.765 | +3.31 ± 0.70 pp | 7/12 | 12.5 pp | FAIL | FAIL |
| rtc_hard d=4 s=4 | 2048 | 0.612 | 0.637 | +2.53 ± 0.75 pp | 4/12 | 12.4 pp | FAIL | FAIL |

Per level, R′ / C (\* marks a level whose 95% CI of the difference excludes 0):

**naive**

| level | naive d=0 | naive d=1 | naive d=2 | naive d=3 | naive d=4 |
|---|---|---|---|---|---|
| grasp_easy | 0.992 / 0.988 | 0.738 / 0.731 | 0.661 / 0.675 | 0.645 / 0.625 | 0.620 / 0.613 |
| catapult | 0.756 / 0.732 | 0.242 / 0.243 | 0.282 / 0.285 | 0.367 / 0.358 | 0.341 / 0.350 |
| cartpole_thrust | 1.000 / 0.999 | 0.999 / 0.998 | 0.988 / 0.990 | 0.673 / 0.690 | 0.399 / 0.404 |
| hard_lunar_lander | 0.988 / 0.988 | 0.983 / 0.979 | 0.974 / 0.971 | 0.902 / 0.892 | 0.562 / 0.552 |
| mjc_half_cheetah | 0.995 / 0.996 | 0.997 / 0.995 | 0.983 / 0.980 | 0.916 / 0.920 | 0.829 / 0.829 |
| mjc_swimmer | 1.000 / 1.000 | 0.951 / 0.946 | 0.733 / 0.743 | 0.312 / 0.313 | 0.142 / 0.154 |
| mjc_walker | 0.953 / 0.944 | 0.745 / 0.733 | 0.486 / 0.480 | 0.137 / 0.123 | 0.017 / 0.016 |
| h17_unicycle | 0.999 / 0.999 | 0.765 / 0.753 | 0.729 / 0.734 | 0.646 / 0.669 | 0.682 / 0.686 |
| chain_lander | 0.998 / 0.998 | 0.995 / 0.996 | 0.995 / 0.996 | 0.996 / 0.992 | 0.990 / 0.988 |
| catcher_v3 | 0.998 / 0.996 | 0.834 / 0.831 | 0.538 / 0.531 | 0.318 / 0.313 | 0.251 / 0.248 |
| trampoline | 0.856 / 0.831* | 0.488 / 0.637* | 0.787 / 0.780 | 0.720 / 0.709 | 0.724 / 0.708 |
| car_launch | 0.993 / 0.988 | 0.805 / 0.807 | 0.796 / 0.807 | 0.723 / 0.744 | 0.624 / 0.613 |

**rtc**

| level | rtc d=0 | rtc d=1 | rtc d=2 | rtc d=3 | rtc d=4 |
|---|---|---|---|---|---|
| grasp_easy | 0.998 / 0.998 | 0.970 / 0.974 | 0.943 / 0.952 | 0.913 / 0.913 | 0.890 / 0.883 |
| catapult | 0.745 / 0.727 | 0.588 / 0.570 | 0.540 / 0.532 | 0.509 / 0.523 | 0.410 / 0.414 |
| cartpole_thrust | 1.000 / 1.000 | 1.000 / 1.000 | 0.997 / 0.996 | 0.801 / 0.794 | 0.517 / 0.525 |
| hard_lunar_lander | 0.995 / 0.988* | 0.987 / 0.991 | 0.982 / 0.973* | 0.945 / 0.938 | 0.792 / 0.759* |
| mjc_half_cheetah | 0.990 / 0.992 | 0.989 / 0.989 | 0.983 / 0.984 | 0.973 / 0.970 | 0.955 / 0.957 |
| mjc_swimmer | 1.000 / 1.000 | 0.998 / 1.000 | 0.955 / 0.941 | 0.638 / 0.646 | 0.313 / 0.324 |
| mjc_walker | 0.967 / 0.972 | 0.935 / 0.919 | 0.741 / 0.740 | 0.494 / 0.501 | 0.254 / 0.265 |
| h17_unicycle | 1.000 / 1.000 | 0.993 / 0.995 | 0.965 / 0.958 | 0.832 / 0.817 | 0.644 / 0.627 |
| chain_lander | 0.999 / 0.999 | 0.996 / 0.999 | 0.998 / 0.997 | 0.994 / 0.997 | 0.988 / 0.986 |
| catcher_v3 | 0.998 / 0.998 | 0.982 / 0.973* | 0.792 / 0.788 | 0.597 / 0.590 | 0.423 / 0.420 |
| trampoline | 0.931 / 0.900* | 0.895 / 0.883 | 0.849 / 0.825* | 0.829 / 0.808 | 0.809 / 0.800 |
| car_launch | 0.987 / 0.984 | 0.942 / 0.948 | 0.904 / 0.906 | 0.845 / 0.846 | 0.704 / 0.724 |

**rtc_hard**

| level | rtc_hard d=0 | rtc_hard d=1 | rtc_hard d=2 | rtc_hard d=3 | rtc_hard d=4 |
|---|---|---|---|---|---|
| grasp_easy | 0.993 / 0.991 | 0.909 / 0.897 | 0.905 / 0.903 | 0.881 / 0.881 | 0.892 / 0.888 |
| catapult | 0.753 / 0.760 | 0.431 / 0.424 | 0.417 / 0.416 | 0.446 / 0.453 | 0.408 / 0.412 |
| cartpole_thrust | 1.000 / 1.000 | 1.000 / 1.000 | 0.990 / 0.992 | 0.861 / 0.855 | 0.517 / 0.502 |
| hard_lunar_lander | 0.988 / 0.986 | 0.987 / 0.987 | 0.976 / 0.973 | 0.949 / 0.938 | 0.792 / 0.762* |
| mjc_half_cheetah | 0.996 / 0.995 | 0.997 / 0.996 | 0.989 / 0.988 | 0.967 / 0.974 | 0.957 / 0.951 |
| mjc_swimmer | 1.000 / 1.000 | 0.997 / 0.995 | 0.910 / 0.903 | 0.609 / 0.591 | 0.313 / 0.332 |
| mjc_walker | 0.956 / 0.940* | 0.909 / 0.895 | 0.687 / 0.691 | 0.525 / 0.519 | 0.263 / 0.244 |
| h17_unicycle | 0.999 / 1.000 | 0.931 / 0.937 | 0.895 / 0.877 | 0.766 / 0.748 | 0.644 / 0.625 |
| chain_lander | 0.998 / 0.996 | 0.999 / 0.995 | 0.994 / 0.997 | 0.992 / 0.997* | 0.988 / 0.991 |
| catcher_v3 | 0.998 / 0.997 | 0.944 / 0.946 | 0.805 / 0.794 | 0.605 / 0.590 | 0.424 / 0.432 |
| trampoline | 0.856 / 0.818* | 0.694 / 0.727* | 0.847 / 0.834 | 0.815 / 0.803 | 0.809 / 0.799 |
| car_launch | 0.992 / 0.993 | 0.886 / 0.879 | 0.868 / 0.867 | 0.821 / 0.834 | 0.704 / 0.708 |

## Live arm (L)

The live runner's first tick fires right after the first observation is sent, and it holds. The controller has 0 s to answer o₀, so tick 0 always applies the hold action, which is zero for Kinetix (see below). The server therefore starts the protocol at the second observation. Its sync counterpart is `initial_hold=true`, which applies the same zero action first. The counterpart references are C-hold (sync) and R′-hold, the official script with one zero-action step before step 0.

The effect of the hold, measured by replaying it in sync (rtc d = 1, 256 rollouts per level, against R′):
- Level average: −7.1 pp.
- car_launch: 0.942 → 0.488.
- trampoline: 0.895 → 0.527.
- The other ten levels move by at most 3 pp.

L is therefore compared against R′-hold and C-hold. The no-hold R′/C results above remain the paper-protocol reference.

*Results pending.*

## Hold action

Kinetix v3 continuous actions are rates, not targets:
- A motor action a ∈ [−1, 1] sets a relative angular-velocity target, and a = 0 applies no impulse (`jax2d.joint.apply_motor`).
- A thruster action ∈ [0, 1] scales a force.

Repeating the last action on a stale tick would keep driving the joints. `KinetixBenchmark.get_hold_action` therefore returns zero, which is Kinetix's own `noop_action`.

## Deviations from the paper

- Kinetix v3.0.0 instead of `cf7453e`: the success metric differs as described above, and the obs replay agrees to float noise. The trampoline residual is open.
- RNG streams differ. The server samples from `sha256(episode_id)` and the benchmark seeds with `seed + episode_idx`, so trajectories match only statistically.
- Episodes per level: 2048 for C, as in the paper. L uses 512 for the d sweep.
- Live runs at 10 Hz. A 30 Hz pilot failed: two rows per CPU server at s = 1 need about 60 inferences/s × 20 ms, which overloads the server, and chunks landed many ticks late.
- Temporal ensembling (TE) is not in the released code and is not run.

## Reproducing

```bash
# server (one per 2 shards on CPU: JAX_PLATFORMS=cpu, single-threaded XLA)
vla-eval serve -c configs/model_servers/rtc/rtc.yaml --arg delay=2 --arg horizon=2 --arg latency=steps
# C (sync); add --arg initial_hold=true for C-hold
vla-eval run -c configs/benchmarks/kinetix/eval.yaml --runtime charliecloud --render cpu
# L (live): server --arg latency=wall
vla-eval run -c configs/benchmarks/kinetix/realtime.yaml --runtime charliecloud --render cpu
# R / R′ / R′-hold: data/rtc-kinetix/ref_eval.py in the official repo's venv (RTC_SOLVED=terminal_reward, RTC_INITIAL_HOLD=1)
python ref_eval.py <rtc>/src <run_dir with 7/ and 31/> out.jsonl core
# per-run analysis and comparison
python data/rtc-kinetix/analyze.py recording-<eval_id>.sqlite <d> <s> --json run.json
python data/rtc-kinetix/compare.py reference/R_core_v3metric.jsonl rtc <d> <s> run.json
```

Use a tmpfs `output_dir` (e.g. `/dev/shm`) when running many shards. Sixty writers with SQLite's fsync on local disk starved past the 60 s busy timeout, and lost 7–31 of 24 576 episodes in four C runs.

## Compute

| Arm | Where | Time |
|-----|-------|------|
| R, R′ (34 configs each) | 6 × H200 (CORE helium) | ≈ 65 s per config; BID ≈ 6 min |
| C (15 conditions × 24 576 episodes) | hydrogen CPU, 40 shards + 20 servers per condition | ≈ 25–45 min per condition |

## Open issues

- Trampoline differs between Kinetix versions: C is 1–4 pp below R′ in most conditions and +14.9 pp at naive d = 1. The fixed-action replay diverges by about 3e-4. Pinning `cf7453e` in the image would settle it.
- BID was not run in C or L. CPU servers take about 185 ms per BID call, so BID needs a GPU server.
