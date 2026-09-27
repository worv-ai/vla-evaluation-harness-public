# Benchmark resource profiles

What each benchmark costs to run under vla-eval, measured with an instant model server (`scripts/profile/`) so
the numbers are the simulator's own cost: physics, rendering, observation transfer and recording. A real model
adds its inference latency per step, or per chunk; [reproductions/running-guide.md](reproductions/running-guide.md)
has the demand/supply method for combining the two, and the model-server supply numbers.

**Unit of resources** for the wall-clock estimates: one node with 4× H100, 96–128 CPU cores, 1 TB RAM.
Measured on DGX H100 nodes (224 hyperthreads, 2 TB), Charliecloud runtime, recording on, no video.

Columns: `render` is the backend (`gpu` = the image's native path, `cpu` = software rendering with no GPU attached);
`step/s` is aggregate env steps per second at `N` shards; `/shard` is what one process achieves inside an episode;
`cores`, `RAM` are per shard at steady state; `protocol` is the official episode count × mean episode length.

## Summary

| Benchmark | Tier | Simulator / renderer | GPU-free? | Recommended setup | step/s (agg.) | Protocol size | Est. wall on the unit | Notes |
|---|:-:|---|:-:|---|--:|--:|--:|---|
| LIBERO-Plus | 1 | MuJoCo / OSMesa (cpu) or EGL (gpu) | **yes** | cpu, 48–96 shards | 143 @ N=48 (see text) | 10,030 ep × ~300 = 3.0 M | ~3 h cpu-only (128 cores); ~1 h with EGL on 2 GPUs (est.) | Sensor-noise variants 2–3× slower (CPU image corruption) |
| RoboTwin 2.0 | 1 | SAPIEN 3 ray tracer (32 spp + OIDN) | cpu path exists, 60× slower | gpu, 4 GPUs, open-loop chunks | see text | 10,000 ep × ≤1,700 (worst 6.2 M; typical ~2.5 M) | 6–14 h with open-loop chunks on 4 GPUs; 60+ h cpu-only | Expert check skipped; `--render cpu` frames are undenoised |
| RoboCasa365 | 2 | MuJoCo / OSMesa or EGL | yes | cpu, 32–48 shards | 144 @ N=32 | 50 tasks × 50 ep × registry horizon (~800) ≈ 2 M | ~4 h cpu-only | 0.25 s/step/shard, ~8 GB/shard (kitchen assets) |
| RoboDojo | 2 | Isaac Lab RTX | no (A100 only, one lane per GPU) | | ~0.5–0.8 /lane | 42 tasks × 50 ep | ~12–20 GPU-h per task | H100 crashes upstream's renderer ([reproductions/robodojo.md](reproductions/robodojo.md)); not re-measured here |
| LIBERO | ref | MuJoCo / OSMesa or EGL | yes | cpu, 64 shards | 158 @ N=64 | 2,000 ep × ~330 = 0.66 M | ~1.2 h cpu-only | 0.27–0.34 s/step/shard; EGL 2.7× faster per shard |
| LIBERO-Pro | 3 | MuJoCo / OSMesa or EGL | yes | cpu, 64 shards | 101 @ N=64 | 4 suites × 10 tasks × 50 ep × ~220 | ~1 h cpu-only | 0.38–0.47 s/step/shard |
| RoboCasa | 3 | MuJoCo / OSMesa or EGL | yes | cpu, 32 shards (RAM-bound) | 90 @ N=32 | 24 tasks × 50 ep × ~540 = 0.65 M | ~2 h cpu-only | 0.27–0.34 s/step/shard, **7.9 GB/shard**: 64 shards need 500 GB |
| CALVIN | 3 | PyBullet / TinyRenderer | yes | cpu, 32–64 shards | 250 @ N=64 (short run) | 1,000 seq × ≤5×360 | ≤ 2 h cpu-only worst case | 0.065 s/step/shard: the cheapest simulator; model-bound in practice |
| SimplerEnv | 3 | SAPIEN 2 / lavapipe or Vulkan | yes | cpu, 32–64 shards (with #84) | 87 @ N=32 | 4 tasks × 24 ep × ≤120 (WidowX) | minutes | 0.29–0.35 s/step/shard, 1.4 cores, 1 GB; without #84 collapses under disk pressure (see text) |
| VLABench | 3 | dm_control / OSMesa | yes | cpu | per-shard 1.1 steps/s | 3 tasks × 3 ep × 200 (eval.yaml) | minutes | 0.83–0.90 s/step/shard, 1.6 cores, ~6 GB; the slowest MuJoCo family member |
| RoboMME | 3 | SAPIEN 3 / lavapipe | yes | cpu, 32–64 shards | 135 @ N=64 | 16 tasks × 50 ep × ≤1,300 ≈ 0.6 M | ~1.5 h cpu-only | 0.18–0.21 s/step/shard, 0.8 cores, 1.8 GB |

## Two harness changes that these numbers depend on

**Shared work queue (#77, #78).** Shards now claim `(task, episode)` items from a `work_queue` table in the
shared recording SQLite instead of a fixed round-robin split. Under the fixed split a 240-episode RoboCasa run on
20 shards finished in 2,582 s with a 17-minute spread between the first and last shard; with the queue 2,278 s and a
3-minute spread. Worse, an entry with fewer items than shards left shards idle in every entry: LIBERO-Plus with 4
entries × 48 items on 64 or 96 shards only ever used shards 0–47.

**Open-loop action chunks.** A `PredictModelServer(open_loop_chunks=True)` (or `VLA_EVAL_OPEN_LOOP_CHUNKS=1`,
or `GET /config?open_loop_chunks=1` at runtime) sends each predicted chunk whole; the sync runner executes it
without fetching the intermediate observations, which are exactly the ones the server would have ignored while
draining its buffer (`action_ensemble="newest"`). Benchmarks may skip rendering on those steps
(`Benchmark.observation_needed`); RoboTwin does. The actions are identical to the buffered path.

## LIBERO-Plus (tier 1)

CPU rendering (OSMesa), 4 suites × 48 variants × 1 episode = 192 episodes, 63,360 steps, instant server, one
DGX H100 node with 128 cores allocated.

| N shards | step/s (wall) | /shard in-episode | cores | RAM | shard wall min–max |
|--:|--:|--:|--:|--:|--:|
| 16 | 50 | 3.4 (0.29 s/step) | 28 (1.75/shard) | 47 GB (3/shard) | 1112–1240 s |
| 32 | 78 (peak 120) | 3.7 (0.27 s/step) | 40 (1.25/shard) | 52 GB (peak 95) | 343–787 s |
| 64 | 143 (peak 165) | 3.5 (0.29 s/step) | 75 (1.2/shard) | 133 GB (2.1/shard) | 17–434 s (16 idle shards) |
| 96 | 142 | 3.4 | 76 | 132 GB | same: 48 active shards |

Per-shard cost does not degrade up to 48 active shards (0.27–0.29 s/step throughout), so CPU rendering scales
linearly with cores: **~1.2 cores and ~2.5 GB per shard, 3.5 steps/s each.** The wall-time spread at N=32
(343–787 s over 6 episodes per shard) is the fixed split; the queue removes it. Episode mean 96 s; p90 150 s.

GPU rendering (EGL) on the same node with 2 H100s attached to the shards, same 192 episodes:

| N shards | step/s (wall) | /shard in-episode | cores | RAM | GPU util (median of both) |
|--:|--:|--:|--:|--:|--:|
| 16 | 209 | 0.068 s/step | 16 (1.0/shard) | 53 GB (3.3/shard) | 33 % |
| 32 | 308 | 0.083 | 29 | 90 GB | 44 % |
| 64 (48 active) | 373 | 0.116 | 51 | 138 GB | 70 % |

EGL is 4× the OSMesa rate per shard and one GPU pair carries ~48 shards before per-step time doubles.

Full protocol (10,030 episodes, ~3.0 M steps): **cpu-only on 64 cores ≈ 48 shards ≈ 170 steps/s ≈ 5 h; with
2 render GPUs ≈ 370 steps/s ≈ 2.3 h** (two more GPUs would roughly halve that again), plus model inference
wait. Either way the model server is the likely bottleneck: a 4B VLM at 50–100 ms per batched forward needs
chunking or many batch slots to keep 50–100 shards fed.

## RoboTwin 2.0 (tier 1)

Per-step components inside the image (`scripts/profile/robotwin_step_timing.py`, `beat_block_hammer`, one process):

| render path | render (3 cameras, 320×240) | physics (take_action) | step |
|---|--:|--:|--:|
| H100, shipped (rt 32 spp + OIDN) | 0.21 s | 0.03 s | 0.23 s |
| H100, rt 8 spp | 0.07 s | 0.03 s | 0.10 s |
| H100, rasterizer (`fast_render`) | crashes (`vk::DeviceLost`) | | |
| CPU lavapipe (Mesa 26.2), rt 32 spp, 4 / 8 / 16 threads | 28.7 / 13.4 / 9.7 s | 0.03 s | ≈ 110 core-seconds per frame |
| CPU lavapipe, rt 16 / 8 / 4 spp (8 threads) | 6.8 / 3.5 / 1.8 s | | linear in spp |

Rendering is 90 % of a closed-loop step on the GPU, and one H100 saturates at roughly 3–4 frames/s across
processes (earlier measurement: ~3.2 env steps/s per H100 with 1, 2 or 4 processes). With open-loop chunks of
25 (a common RoboTwin chunk), only 1 step in 25 renders, so a step costs ~0.04 s and one GPU serves ~80 steps/s.

End-to-end via `vla-eval run --render cpu`, open-loop chunk 25, 8 lavapipe threads: a 400-step episode takes
288 s (0.72 s/step). Frames are the same ray tracer without OIDN denoising, so visibly noisier
(mean |Δ| 3.9/255 against the GPU frame): a policy trained on denoised frames may react differently; treat the
CPU path as a fallback for hosts without a GPU, not as the protocol.

A 32-shard CPU sweep (50 tasks × 1 episode, open-loop chunk 25, `LP_NUM_THREADS=4` on 64 cores, before #84)
ran at 1.6 s/step per shard, 8 steps/s for the node: the rendering budget of ~110 core-seconds per frame
divided by 64 cores gives at most ~0.6 frames/s, i.e. ~15 steps/s with chunk 25 even without contention.

End-to-end on 4 H100s (`vla-eval run`, instant server, open-loop chunk 25, 50 tasks × 4 seeds): one shard per
GPU runs 0.046 s/step in-episode (physics-bound: 25 × 0.03 s physics per 0.21 s frame, GPU idle), two shards
per GPU 0.064 s/step; the 4th GPU of that allocation was unusable, so aggregate numbers from that run are
being re-measured on the three good ones (see below when filled).

Full protocol (50 tasks × 2 configs × 100 episodes; step limits sum to 31,000 per 100-episode sweep, so the
worst case is 6.2 M steps, a typical run with early successes ~2.5 M): with open-loop chunk 25 each shard
does ~16–22 steps/s and a GPU renders ~4 frames/s = ~100 steps/s, so **4 GPUs ≈ 300–400 steps/s ≈ 4–6 h worst
case**, 2–3 h typical; closed-loop (render every step) ≈ 13 steps/s ≈ 5 days. CPU-only on 64 cores: ≈ 8–15
steps/s, i.e. 5–9 days worst case — a fallback, not a plan. The CuRobo expert check is skipped
(`skip_expert_check`); note that unverified seeds include a few unstable scenes (`UnStableError` at reset),
which the expert check would have filtered.

## The MuJoCo family, CALVIN, SimplerEnv, RoboMME, VLABench (CPU rendering, one node)

| Benchmark | N | step/s (wall) | s/step/shard | cores/shard | GB/shard | notes |
|---|--:|--:|--:|--:|--:|---|
| LIBERO (4 suites × 10 tasks × 5 ep = 200 ep, 66 k steps) | 16 / 32 / 64 | 40 / 99 / 158 | 0.31 / 0.27 / 0.34 | 1.4 / 1.5 / 1.4 | 2.9 / 2.8 / 2.4 | shard wall spread 1205–1662 s at N=16 |
| LIBERO-Pro (10 tasks × 10 ep, 22 k steps) | 16 / 32 / 64 | 35 / 71 / 101 | 0.38 / 0.38 / 0.47 | 1.7 / 1.6 / 1.3 | 2.5 / 2.5 / 2.1 | contention starts at 64 shards on 64 cores |
| RoboCasa365 (50 tasks × 2 ep, ~100 k steps) | 16 / 32 | 59 / 144 | 0.26 / 0.25 | ~1.5 | ~8 | registry horizons 500–1,200 steps |
| RoboCasa (24 tasks × 4 ep, 52 k steps) | 16 / 32 / 64 | 49 / 90 / 101 | 0.29 / 0.27 / 0.34 | 1.3 / 1.2 / 0.8 | 8.4 / 7.9 / 7.8 | peak 499 GB at N=64; n64 shared its node with an image export |
| CALVIN (48 sequences, 360 steps each) | 16 / 32 / 64 | 162 / 227 / 254 | 0.065 / 0.063 / 0.080 | 1.1 / 1.1 / 1.1 | 1.1 / 0.7 / 0.6 | run too short for the aggregate to matter; per-shard rate is the number |
| SimplerEnv WidowX (96 ep × 75 steps), with #84 | 32 | 87 | 0.29 | 1.6 | 1.0 | without #84 (lavapipe frames on /tmp): 0.34 at N=16, then 1.9–3.8 s/step at N=32 and 5.4–6.2 at N=64 whenever the node's disk was busy, the job using 6–11 cores |
| RoboMME (16 tasks × 3 ep, 59 k steps) | 16 / 32 / 64 | 71 / 106 / 135 | 0.18 / 0.19 / 0.21 | 1.0 / 0.8 / 0.6 | 2.8 / 1.8 / 1.5 | 48 episodes: shards idle at N=64 |
| VLABench (3 tasks × 10 ep, 6 k steps) | 16 / 32 / 64 | 13 / 16 / 16 | 0.89 / 0.90 / 0.83 | 1.6 / 1.0 / 0.5 | 5.9 / 1.7 / 0.7 | 30 episodes: shards idle beyond N=16 |

The node allocation of 128 "CPUs" is 64 physical cores with their hyperthreads, so the LIBERO family's ~1.4
cores/shard saturates a node at about 48 shards; beyond that per-step time grows instead of throughput.

**Two things that made runs collapse, both fixed:** (1) lavapipe backs device memory with unlinked files under
`/tmp`, so every software-rendered frame is a disk write and the render thread ends up in `balance_dirty_pages`
as soon as anything else writes to the node's disk (#84 moves it to `/dev/shm`); (2) the progress watchdog was
petted once per episode, so a slow episode longer than 20 min killed its shard (#82). A third is operational:
an image export (`ch-convert`/`mksquashfs`) on the same node distorts a sweep through CephFS and disk traffic;
`LOCAL_IMAGE=1` in the profiler stages the SquashFS on local disk, and exports should not share a node with a run.

## RoboDojo (tier 2)

Not re-measured: the RTX renderer crashes on H100 (`ERROR_DEVICE_LOST`; documented in
[reproductions/robodojo.md](reproductions/robodojo.md)) and no A100 is schedulable here; the ~90 GiB assets are
not on the cluster either. Known figures from the A100 reproduction: 30–50 steps/min per lane, one lane per GPU
(4 lanes on one A100 collapsed throughput ~30×), 12–20 GPU-hours per task, one task per container process.
The upstream `num_envs > 1` batched path is the only route to a practical protocol run and needs a vectorised
benchmark interface in the harness.
