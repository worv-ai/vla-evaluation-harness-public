# RTC on Kinetix — Reproduction Report

Real-Time Chunking (RTC) for action-chunking flow policies. [Paper](https://arxiv.org/abs/2506.07339) |
[GitHub](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) | 12 Kinetix levels, flow-matching MLP-Mixer, H = 8.

The goal is not only to reproduce Fig. 5. It is to show that vla-eval's **live** mode (wall-clock ticks, real plus injected
inference latency) gives the same solve rates as the paper's **sync** protocol, where the delay is counted in steps.

## Results Summary

| Arm | What | Result |
|-----|------|--------|
| R | Official `eval_flow.py`, run by us (2048 rollouts × 12 levels, `bc/31`, BID weak `bc/7`) | ✅ matches Fig. 5 (≤ 0.7 pp, BID ≤ 1.9 pp) |
| R′ | R with success read as Kinetix v3 reads it (see *Success metric*) | the reference vla-eval can be held to |
| C | vla-eval **sync**, delay counted in steps (2048 × 12) | ✅ vs R′: mean diff −0.2 pp, 11/15 conditions pass both criteria |
| L | vla-eval **live**, 10 Hz wall clock, latency (d − ½)/hz (512 × 12) | ✅ vs C: 10/10 conditions pass both criteria, \|diff\| ≤ 0.84 pp; vs R′: 9/10 |

Every criterion failure comes from levels near 100% success at d = 0 or 1, where the CI of the average is about ±0.3 pp, or from one cell: trampoline at naive d = 1, which both vla-eval arms place 15–17 pp above R′ (see *Open issues*).

### Links

| | |
|---|---|
| **Checkpoints** | `gs://rtc-assets/bc/31/policies` (strong), `bc/7` (BID weak), public HTTPS; see [`configs/model_servers/rtc/README.md`](../../configs/model_servers/rtc/README.md) |
| **Server** | [`src/vla_eval/model_servers/rtc.py`](../../src/vla_eval/model_servers/rtc.py) (`KinetixRTCServer`, imports the official `FlowPolicy`) |
| **Server configs** | [`configs/model_servers/rtc/`](../../configs/model_servers/rtc/) (`naive`, `rtc`, `rtc_hard`, `bid`) |
| **Benchmark configs** | [`eval.yaml`](../../configs/benchmarks/kinetix/eval.yaml) (sync), [`realtime.yaml`](../../configs/benchmarks/kinetix/realtime.yaml) (live) |
| **Results** | [`data/rtc-kinetix/`](data/rtc-kinetix/): `reference/` (R, R′ JSONL), `sync/`, `live/`, `sync_initial_hold/`, scripts |

## Protocol

The official protocol, from `eval_flow.py` @ `9296f31`, with Kinetix `cf7453e`:
- Chunk 0 comes from o₀ with no delay.
- Every s steps, a chunk is sampled from that step's observation. The previous chunk covers the next d steps, then the new chunk covers the rest of the s steps.
- Action noise N(0, 0.1²). The episode is capped at 256 steps. Only the first episode of each rollout counts.

`KinetixRTCServer` runs this same schedule per episode. It answers every observation with the plan's action for that step: a `reply` in sync, a `PUBLISH` in live. The step is the benchmark's `env_step`, not a time grid. `tests/test_rtc_server.py` checks the executed actions against `eval_flow`'s, sync for six (d, s) pairs and live once.

A boundary's chunk lands at one of three points, set by `latency`:
- `steps` (C): when step t + d's observation arrives.
- `wall` (L): (d − ½)/hz after its observation was admitted, compute included. Half a period from either tick, so jitter below Δt/2 cannot change the effective delay.
- `wall_exact`: d/hz after admission.

The latency is slept in the episode task, never on the model thread. Methods are naive, RTC soft (`exp`, β = 5), RTC hard (`zeros`), and BID (N = 32, K = 3, weak `bc/7`). Each method calls the official `action` / `realtime_action` / `bid_action` unchanged.

**Live start.** The live runner's first tick fires immediately after the first observation, before any action can exist. In Kinetix, sim time 0 is an artifact: objects start moving the instant the sim starts. The real-world analogue is an experimenter releasing the scene once the controller is live. So `KinetixBenchmark(start_on_first_action=true)` does not step the env on a hold that arrives before the first policy action; it returns the same observation, and the first action starts env time.
- Exactly one such pre-start tick happened per live episode.
- The server re-anchors step 0's latency to its latest observation.
- This gives live the paper's "chunk 0 without delay". Without it, the one zero-action step cost 7.1 pp on the average at RTC d = 1 (car_launch 0.94 → 0.49, trampoline 0.89 → 0.53; see *Initial hold*).

**Hold action.** Kinetix v3 continuous actions are rates, not targets:
- A motor action a ∈ [−1, 1] sets a relative angular-velocity target, and a = 0 applies no impulse (`jax2d.joint.apply_motor`).
- A thruster action ∈ [0, 1] scales a force.

A stale tick therefore holds with zero, which is Kinetix's own `noop_action`, rather than repeating the last action.

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

R reproduces Fig. 5 within 0.7 pp for RTC, hard masking, and naive. BID with the paper setting is 1.6–1.9 pp above the figure at d = 2 and d = 3. The released sweep default for BID (N = 16, no weak policy) gives 0.813/0.686/0.555 at d = 2/3/4; see `reference/R_bid_default.jsonl`.

## Success metric: Kinetix v3 vs `cf7453e`

vla-eval pins Kinetix v3.0.0, while the paper used `cf7453e`. The two versions read a step's success differently, and that difference is in how the frame-skip substeps are combined. There are 2 substeps per step.
- `cf7453e`: the step reward is the max over its substeps, and `GoalR` is taken from the **last** substep. A goal touched on substep 1 and lost on substep 2 therefore counts as a failure.
- v3: the reward and `GoalR` come from the **first** substep that ends the episode.

The physics is the same. Replaying a fixed 256-action sequence in both versions gives the same observations to about 1e-6 on the non-chaotic levels; the chaotic ones diverge from float noise.

R itself shows the effect: under the official metric, the solve rate is below the return. Examples: cheetah solved 0.884 with return 0.994, swimmer 0.921 with 1.000, walker 0.806 with 0.953.

**R′** keeps the official script and seeds and changes only the logged success, to "the terminal step's reward is +1". That is v3's semantics, except for a step whose first substep is red and whose second is green, which R′ counts as solved. R′ is about +4 pp over R; its averages:

| method | d=0 | d=1 | d=2 | d=3 | d=4 |
|---|---|---|---|---|---|
| rtc | 0.967 / 0.922 | 0.940 / 0.893 | 0.887 / 0.843 | 0.781 / 0.747 | 0.642 / 0.618 |
| rtc_hard | 0.961 / 0.916 | 0.890 / 0.849 | 0.857 / 0.817 | 0.770 / 0.732 | 0.643 / 0.619 |
| bid | 0.965 / 0.921 | 0.927 / 0.877 | 0.858 / 0.801 | 0.729 / 0.673 | 0.585 / 0.552 |
| naive | 0.961 / 0.915 | 0.795 / 0.753 | 0.746 / 0.708 | 0.613 / 0.584 | 0.515 / 0.493 |

vla-eval cannot report the paper metric under v3, so C and L are compared against R′.

## Results

The criteria are fixed in advance:
- (1) The level-average difference lies within the 95% CI of the difference.
- (2) At most 2 of 12 levels lie outside their 95% CI (Newcombe), and no level is off by more than 5 pp. The 5 pp cap applies to the 2048-rollout arms only.

### Sync (C, 2048 × 12) vs R′

| condition | n/level | ref | arm | diff [95% CI] | levels outside | worst level | (1) | (2) |
|---|---|---|---|---|---|---|---|---|
| naive d=0 s=1 | 2046 | 0.961 | 0.955 | -0.55 ± 0.33 pp | 1/12 | 2.5 pp | FAIL | pass |
| naive d=1 s=1 | 2037 | 0.795 | 0.804 | +0.90 ± 0.60 pp | 1/12 | 14.9 pp | FAIL | FAIL |
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

### Live (L, 512 × 12, 10 Hz, `latency=wall`) vs C — live vs sync, same env and metric

| condition | n/level | ref | arm | diff [95% CI] | levels outside | worst level | (1) | (2) |
|---|---|---|---|---|---|---|---|---|
| naive d=0 s=1 | 512 | 0.955 | 0.957 | +0.16 ± 0.53 pp | 0/12 | 2.9 pp | pass | pass |
| naive d=1 s=1 | 512 | 0.804 | 0.807 | +0.30 ± 0.95 pp | 0/12 | 2.5 pp | pass | pass |
| naive d=2 s=2 | 512 | 0.748 | 0.751 | +0.29 ± 1.05 pp | 0/12 | 3.3 pp | pass | pass |
| naive d=3 s=3 | 512 | 0.612 | 0.605 | -0.76 ± 1.15 pp | 0/12 | 2.8 pp | pass | pass |
| naive d=4 s=4 | 512 | 0.513 | 0.516 | +0.30 ± 1.16 pp | 2/12 | 5.7 pp | pass | pass |
| rtc d=0 s=1 | 512 | 0.963 | 0.965 | +0.19 ± 0.48 pp | 0/12 | 2.9 pp | pass | pass |
| rtc d=1 s=1 | 512 | 0.937 | 0.937 | +0.04 ± 0.61 pp | 1/12 | 3.2 pp | pass | pass |
| rtc d=2 s=2 | 512 | 0.882 | 0.881 | -0.20 ± 0.83 pp | 0/12 | 2.0 pp | pass | pass |
| rtc d=3 s=3 | 512 | 0.779 | 0.787 | +0.84 ± 1.05 pp | 1/12 | 6.1 pp | pass | pass |
| rtc d=4 s=4 | 512 | 0.640 | 0.633 | -0.72 ± 1.16 pp | 0/12 | 4.5 pp | pass | pass |

### Live (L) vs R′

| condition | n/level | ref | arm | diff [95% CI] | levels outside | worst level | (1) | (2) |
|---|---|---|---|---|---|---|---|---|
| naive d=0 s=1 | 512 | 0.961 | 0.957 | -0.40 ± 0.52 pp | 1/12 | 3.6 pp | pass | pass |
| naive d=1 s=1 | 512 | 0.795 | 0.807 | +1.20 ± 0.95 pp | 1/12 | 17.4 pp | FAIL | pass |
| naive d=2 s=2 | 512 | 0.746 | 0.751 | +0.46 ± 1.05 pp | 0/12 | 2.8 pp | pass | pass |
| naive d=3 s=3 | 512 | 0.613 | 0.605 | -0.81 ± 1.16 pp | 0/12 | 4.5 pp | pass | pass |
| naive d=4 s=4 | 512 | 0.515 | 0.516 | +0.14 ± 1.16 pp | 2/12 | 4.8 pp | pass | pass |
| rtc d=0 s=1 | 512 | 0.967 | 0.965 | -0.24 ± 0.47 pp | 1/12 | 1.7 pp | pass | pass |
| rtc d=1 s=1 | 512 | 0.940 | 0.937 | -0.26 ± 0.61 pp | 0/12 | 2.2 pp | pass | pass |
| rtc d=2 s=2 | 512 | 0.887 | 0.881 | -0.70 ± 0.82 pp | 1/12 | 2.1 pp | pass | pass |
| rtc d=3 s=3 | 512 | 0.781 | 0.787 | +0.62 ± 1.05 pp | 1/12 | 6.9 pp | pass | pass |
| rtc d=4 s=4 | 512 | 0.642 | 0.633 | -0.84 ± 1.16 pp | 1/12 | 6.5 pp | pass | pass |

### Live instruments

| Instrument | Pass mark | Result over 10 conditions × 6144 episodes |
|---|---|---|
| Effective delay = d | ≥ 99.5% | 99.96–100% of boundaries |
| Stale ticks | ≤ 0.5% | ≤ 0.003% |
| Pre-start ticks per episode | — | 1.00 |
| Inference cadence | every s steps | exact |
| `step_time_max` < Δt/2 | every episode | 13–99 episodes per condition (0.2–1.6%) have one env step > 50 ms |

The step-time spikes reach 136–322 ms, at random points mid-episode. They come from the shared node, not from JIT, which is pre-compiled. The live runner does not catch up, so a spike shifts a few boundaries by ±1 to 3 steps: 0.04% of boundaries at worst (rtc d = 4). Publish to apply takes Δt − ε (p50 ≈ 95 ms). Inference p99 is about 10–25 ms on a single CPU thread.

Per condition:

| condition | episodes | boundaries | effective delay = d | stale ticks | pre-start ticks/ep | step time max (worst ep) | eps > Δt/2 | land margin min (ms) |
|---|---|---|---|---|---|---|---|---|
| naive d=0 | 6144 | 532806 | 100.000% (—) | 12 / 532806 (0.002%) | 1.00 | 171 ms | 13 | — |
| naive d=1 | 6144 | 639675 | 99.999% (1: 5) | 17 / 645869 (0.003%) | 1.00 | 271 ms | 36 | -0.1 |
| naive d=2 | 6144 | 331930 | 99.996% (-1: 11, 1: 3) | 1 / 673590 (0.000%) | 1.00 | 272 ms | 16 | -217.9 |
| naive d=3 | 6144 | 228851 | 99.991% (-2: 1, -1: 7, 1: 11, 2: 1) | 15 / 697684 (0.002%) | 1.00 | 269 ms | 35 | -213.4 |
| naive d=4 | 6144 | 178434 | 99.982% (-3: 1, -2: 1, -1: 20, 1: 10) | 8 / 731210 (0.001%) | 1.00 | 267 ms | 45 | -117.7 |
| rtc d=0 | 6144 | 524178 | 100.000% (—) | 15 / 524178 (0.003%) | 1.00 | 295 ms | 26 | — |
| rtc d=1 | 6144 | 556177 | 100.000% (1: 1) | 9 / 562344 (0.002%) | 1.00 | 169 ms | 21 | 0.7 |
| rtc d=2 | 6144 | 300865 | 99.992% (-1: 11, 1: 12) | 7 / 611335 (0.001%) | 1.00 | 322 ms | 20 | -268.1 |
| rtc d=3 | 6144 | 214545 | 99.995% (-1: 5, 1: 6) | 2 / 655357 (0.000%) | 1.00 | 136 ms | 18 | -9.4 |
| rtc d=4 | 6144 | 169307 | 99.960% (-3: 1, -2: 2, -1: 39, 1: 25) | 17 / 694190 (0.002%) | 1.00 | 257 ms | 99 | -190.4 |

### Per level

R′ / C (\* marks a level whose 95% CI of the difference excludes 0):

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

R′ / L:

**naive**

| level | naive d=0 | naive d=1 | naive d=2 | naive d=3 | naive d=4 |
|---|---|---|---|---|---|
| grasp_easy | 0.992 / 0.990 | 0.738 / 0.748 | 0.661 / 0.684 | 0.645 / 0.600 | 0.620 / 0.611 |
| catapult | 0.756 / 0.762 | 0.242 / 0.260 | 0.282 / 0.295 | 0.367 / 0.379 | 0.341 / 0.293* |
| cartpole_thrust | 1.000 / 1.000 | 0.999 / 1.000 | 0.988 / 0.994 | 0.673 / 0.670 | 0.399 / 0.436 |
| hard_lunar_lander | 0.988 / 0.982 | 0.983 / 0.980 | 0.974 / 0.971 | 0.902 / 0.893 | 0.562 / 0.588 |
| mjc_half_cheetah | 0.995 / 0.998 | 0.997 / 0.994 | 0.983 / 0.992 | 0.916 / 0.900 | 0.829 / 0.836 |
| mjc_swimmer | 1.000 / 1.000 | 0.951 / 0.932 | 0.733 / 0.713 | 0.312 / 0.303 | 0.142 / 0.156 |
| mjc_walker | 0.953 / 0.938 | 0.745 / 0.732 | 0.486 / 0.514 | 0.137 / 0.117 | 0.017 / 0.018 |
| h17_unicycle | 0.999 / 0.996 | 0.765 / 0.742 | 0.729 / 0.734 | 0.646 / 0.641 | 0.682 / 0.717 |
| chain_lander | 0.998 / 1.000 | 0.995 / 0.990 | 0.995 / 0.992 | 0.996 / 0.994 | 0.990 / 0.975* |
| catcher_v3 | 0.998 / 0.998 | 0.834 / 0.854 | 0.538 / 0.525 | 0.318 / 0.324 | 0.251 / 0.238 |
| trampoline | 0.856 / 0.820* | 0.488 / 0.662* | 0.787 / 0.801 | 0.720 / 0.719 | 0.724 / 0.699 |
| car_launch | 0.993 / 0.994 | 0.805 / 0.791 | 0.796 / 0.793 | 0.723 / 0.719 | 0.624 / 0.629 |

**rtc**

| level | rtc d=0 | rtc d=1 | rtc d=2 | rtc d=3 | rtc d=4 |
|---|---|---|---|---|---|
| grasp_easy | 0.998 / 0.994 | 0.970 / 0.957 | 0.943 / 0.932 | 0.913 / 0.934 | 0.890 / 0.912 |
| catapult | 0.745 / 0.756 | 0.588 / 0.602 | 0.540 / 0.539 | 0.509 / 0.492 | 0.410 / 0.420 |
| cartpole_thrust | 1.000 / 1.000 | 1.000 / 1.000 | 0.997 / 0.998 | 0.801 / 0.805 | 0.517 / 0.492 |
| hard_lunar_lander | 0.995 / 0.986* | 0.987 / 0.980 | 0.982 / 0.963* | 0.945 / 0.941 | 0.792 / 0.727* |
| mjc_half_cheetah | 0.990 / 0.988 | 0.989 / 0.992 | 0.983 / 0.992 | 0.973 / 0.977 | 0.955 / 0.963 |
| mjc_swimmer | 1.000 / 1.000 | 0.998 / 1.000 | 0.955 / 0.938 | 0.638 / 0.707* | 0.313 / 0.295 |
| mjc_walker | 0.967 / 0.959 | 0.935 / 0.928 | 0.741 / 0.730 | 0.494 / 0.516 | 0.254 / 0.236 |
| h17_unicycle | 1.000 / 1.000 | 0.993 / 0.994 | 0.965 / 0.953 | 0.832 / 0.809 | 0.644 / 0.631 |
| chain_lander | 0.999 / 1.000 | 0.996 / 0.998 | 0.998 / 0.996 | 0.994 / 0.996 | 0.988 / 0.996 |
| catcher_v3 | 0.998 / 0.996 | 0.982 / 0.982 | 0.792 / 0.801 | 0.597 / 0.602 | 0.423 / 0.465 |
| trampoline | 0.931 / 0.914 | 0.895 / 0.873 | 0.849 / 0.828 | 0.829 / 0.812 | 0.809 / 0.779 |
| car_launch | 0.987 / 0.986 | 0.942 / 0.938 | 0.904 / 0.896 | 0.845 / 0.854 | 0.704 / 0.682 |

### Initial hold (secondary)

Before the gated start existed, live tick 0 applied one zero-action step before step 0. Its sync counterpart is the same zero step, replayed in C (server option, commit `6803588`) and in R′ (`RTC_INITIAL_HOLD=1`). These two agree as well, which is a second sync/sync check:

| condition | n/level | ref | arm | diff [95% CI] | levels outside | worst level | (1) | (2) |
|---|---|---|---|---|---|---|---|---|
| naive d=0 s=1 | 2048 | 0.918 | 0.920 | +0.19 ± 0.43 pp | 1/12 | 1.5 pp | pass | pass |
| naive d=1 s=1 | 2048 | 0.764 | 0.770 | +0.61 ± 0.62 pp | 1/12 | 10.8 pp | pass | FAIL |
| naive d=2 s=2 | 2048 | 0.698 | 0.696 | -0.23 ± 0.69 pp | 1/12 | 1.9 pp | pass | pass |
| naive d=3 s=3 | 2048 | 0.559 | 0.554 | -0.48 ± 0.74 pp | 0/12 | 2.5 pp | pass | pass |
| naive d=4 s=4 | 2048 | 0.457 | 0.457 | -0.05 ± 0.73 pp | 1/12 | 3.2 pp | pass | pass |
| rtc d=0 s=1 | 2048 | 0.903 | 0.904 | +0.05 ± 0.44 pp | 1/12 | 3.8 pp | pass | pass |
| rtc d=1 s=1 | 2048 | 0.870 | 0.873 | +0.31 ± 0.49 pp | 2/12 | 3.4 pp | pass | pass |
| rtc d=2 s=2 | 2048 | 0.814 | 0.815 | +0.15 ± 0.58 pp | 2/12 | 4.6 pp | pass | pass |
| rtc d=3 s=3 | 2048 | 0.714 | 0.714 | +0.00 ± 0.70 pp | 0/12 | 2.0 pp | pass | pass |
| rtc d=4 s=4 | 2048 | 0.578 | 0.584 | +0.62 ± 0.75 pp | 2/12 | 3.9 pp | pass | pass |

## Deviations from the paper

- Kinetix v3.0.0 instead of `cf7453e`: the success metric differs as described above, and the obs replay agrees to float noise. The trampoline residual is open.
- RNG streams differ. The server samples from `sha256(episode_id)` and the benchmark seeds with `seed + episode_idx`, so trajectories match only statistically.
- Episodes per level: C uses 2048, as in the paper. L uses 512; 2048 for the live core is not run yet.
- Live runs at 10 Hz. A 30 Hz pilot failed: two rows per CPU server at s = 1 need about 60 inferences/s × 20 ms, which overloads the server, and chunks landed many ticks late.
- Live env time starts with the first action (`start_on_first_action`), as described in *Live start*.
- Temporal ensembling (TE) is not in the released code and is not run. BID was not run in C or L, because BID needs a GPU server (about 185 ms per call on CPU).

## Reproducing

```bash
# server, CPU (JAX_PLATFORMS=cpu, single-threaded XLA): one per 2 shards
vla-eval serve -c configs/model_servers/rtc/rtc.yaml --arg delay=2 --arg horizon=2 --arg latency=steps
# C (sync)
vla-eval run -c configs/benchmarks/kinetix/eval.yaml --runtime charliecloud --render cpu
# L (live): server --arg latency=wall
vla-eval run -c configs/benchmarks/kinetix/realtime.yaml --runtime charliecloud --render cpu
# R / R′ / R′-hold: data/rtc-kinetix/ref_eval.py in the official repo's venv (RTC_SOLVED=terminal_reward, RTC_INITIAL_HOLD=1)
python ref_eval.py <rtc>/src <run_dir with 7/ and 31/> out.jsonl core
# per-run analysis and comparison
python data/rtc-kinetix/analyze.py recording-<eval_id>.sqlite <d> <s> [--live] --json run.json
python data/rtc-kinetix/compare.py reference/R_core_v3metric.jsonl rtc <d> <s> run.json
```

Use a tmpfs `output_dir` (e.g. `/dev/shm`) when running many shards. Sixty writers with SQLite's fsync on local disk starved past the 60 s busy timeout, and lost 7–31 of 24 576 episodes in four C runs. For live runs, pin one core per shard and per server.

## Compute

| Arm | Where | Time |
|-----|-------|------|
| R, R′, R′-hold (34 + 34 + 15 configs) | 6 × H200 (CORE helium) | ≈ 65–150 s per config; BID ≈ 6 min |
| C, C-hold (25 conditions × 24 576 episodes) | hydrogen CPU, 40 shards + 20 servers per condition | ≈ 25–45 min per condition |
| L (10 conditions × 6144 episodes) | hydrogen CPU, 48 shards + 24 servers, pinned | ≈ 20–30 min per condition |

## Open issues

- Trampoline differs between Kinetix versions. C is 1–4 pp below R′ in most conditions, and both C and L are 15–17 pp above R′ at naive d = 1 (R′ 0.488; C 0.637, L 0.662). Live agrees with sync there, so this is env-side, not live-side. The fixed-action replay diverges by about 3e-4. Pinning `cf7453e` in the image would settle it.
- L at 2048 rollouts for the core set, plus L for hard masking, BID (GPU server), the horizon sweep and the realism variants (`wall_exact`, `first_chunk_delayed`), are not run yet.
