# RFC-0009: Vectorized Benchmarks

- **Author:** @MilkClouds
- **Status:** Implemented
- **Type:** Standards Track
- **Created:** 2026-10-03
- **Requires:** RFC-0004, RFC-0006, RFC-0007
- **Superseded-By:** —

## Summary

Add `VectorStepBenchmark`, a sync benchmark contract over `num_envs` environments that one
process steps together, and `VectorEpisodeRunner`, which runs up to `num_envs` episodes at
once on it over one connection, one `VECTOR_STEP` round trip per step.  Each running episode
is its own server session, so model servers need no change, and a batching model server
(RFC-0007) infers a step's observations in one batch.

## Motivation

GPU-parallel simulators (MuJoCo Warp / mjlab, ManiSkill3 on the GPU, Isaac Lab) step
hundreds of environments in one kernel launch; one environment per process leaves the
device nearly idle and pays the per-step launch and synchronisation cost once per
environment.  Episode sharding (RFC-0006) parallelises across processes, each with its
own GPU context and a single environment, which is the expensive end of the trade.  On
an mjlab benchmark (430 steps per episode, three rendered cameras, one A100, a model
server that answers at once), one environment ran an episode in about 40 s; with this
runner, 32 environments ran 32 episodes in 3 min 11 s, 6.0 s per episode.

## Design

### Contract

```python
class VectorStepBenchmark(BenchmarkCommon):
    partial_reset: ClassVar[bool] = False

    def __init__(self, num_envs: int = 1): ...
    def reset(self, slots: list[int], tasks: list[Task], recorders: list[EpisodeRecorder]) -> list[Any]: ...
    def step(self, actions: dict[int, Action]) -> dict[int, StepResult]: ...
    def make_obs(self, raw_obs: Any, slot: int, task: Task) -> Observation: ...
    def get_step_result(self, slot: int, step_result: StepResult) -> EpisodeResult: ...
    def check_done(self, step_result: StepResult) -> bool: ...  # default: step_result.done
```

A slot is an environment index.  `reset` starts `tasks[i]` in slot `slots[i]`; `step`
advances every environment once and returns a result for each slot in `actions`.  A slot
missing from `actions` is idle and the benchmark keeps it inert.  Everything a benchmark
declares besides stepping (tasks, specs, metric keys, metadata, render backends, the
real-time hold, cleanup) moved to `BenchmarkCommon`, the shared parent of `Benchmark` and
`VectorStepBenchmark`; `Benchmark`'s interface is unchanged.

`reset` may put an exception in place of a slot's observation: that episode alone fails
(`env_start`), as when one layout of a wave fails its stability check.

An episode's outcome depends only on its task (seed, layout), never on the slot, on
`num_envs`, or on the episodes that run beside it.  The native simulators break this by
default: mjlab, ManiSkill3 and Isaac Lab draw reset randomisation from one global RNG in a
batch, so an adapter seeds each slot itself (per-slot generators, or a published layout as in
RoboDojo) and turns native auto-reset off.

`partial_reset` declares per-environment resets.  With it, a slot that finishes takes the
next episode at once.  Without it, the runner resets only when no slot is running and runs
the items in waves of up to `num_envs`.

### Runner

`VectorEpisodeRunner` holds one `Connection`.  Each step is one round trip
(`Connection.exchange`): the episodes that ended since the last one, the ones just started and
one observation per running slot go out in one `VECTOR_STEP`, and the reply carries each slot's
action or error.  Then it steps the environments and ends the episodes that are done or reached
`max_steps`.  The step count, `max_steps` and `elapsed_sec` (wall time from the episode's reset)
are per episode, as in `SyncEpisodeRunner`.

The runner's calls into the benchmark (`reset`, `step`, `make_obs`, `get_step_result`,
`check_done`, `cleanup`) run on one dedicated thread: Isaac Sim's Kit loop, GL/Vulkan contexts
and CUDA state are bound to the thread that created them, and the event loop stays free for the
slots' connections.  The constructor, `get_tasks` and `get_metadata` still run on
the orchestrator's thread, so an adapter builds its simulator lazily in the first `reset`.

### Protocol and server

```
VECTOR_STEP   {"ends": [[slot, result]], "starts": [[slot, episode_start]], "observations": [[slot, obs]]}
VECTOR_RESULT {"actions": [[slot, action]], "errors": [[slot, detail]]}
```

The server handles a step in that order: it ends the slots' sessions, starts new ones (a failed
`on_episode_start` is that slot's error), then hands the observations to
`ModelServer.on_observations(items)`.  Each slot is its own `SessionContext`, whose
`session_id` is the episode's `recording.sid` (else `<connection>:<slot>`), so chunk buffers,
`ctx.is_first`, model-side history keyed by `ctx.session_id` and the episode lifecycle work as
for one connection per episode.  `on_observations` defaults to `on_observation` per item;
`PredictModelServer` serves buffered chunk actions and infers the rest with `predict_batch` in
groups of `max_batch_size` (a subclass that overrides `on_observation`, e.g. to keep a dense
history, keeps the per-item path).  HELLO advertises `capabilities.vector_step`
(continuous-inference servers do not, as their actions come later than the observation), and
the orchestrator refuses a server without it.  Messages other than `VECTOR_STEP` keep the
connection's own session, so the sync and live runners are unchanged.

### Orchestrator

The orchestrator selects the vector path when the benchmark is a `VectorStepBenchmark`
(sync mode only; live mode raises before any episode).  It runs the entry on the shard's one
connection and gives each episode its own recording `sid`, the id of its server session, so
sessions never merge and a reconnect never reuses one.  The running episodes share one task: a
free slot takes an item of that task, and an item of the next task waits until they have all
ended, so an adapter builds one scene at a time.  Work items, the shared work queue, recorders, results, progress and
trackers are handled per episode as in the one-at-a-time loop, and an item is finished in the queue only after its
episode is recorded.  `RecordingStore.claim`
takes `exclude`, the shard's items still running, so a shard holding several claims is
never handed one of them again; a rerun after a crash still resumes its unfinished items.

Failures stay with their episode:

| Failure | Episodes that end | Then |
|---------|-------------------|------|
| `reset` raises | every one on the simulator: those it was starting (`env_start`) and the running ones, as the shared state is unknown | next items |
| `reset` returns an exception for a slot | that slot's (`env_start`) | next items |
| `make_obs` raises | that slot's | next items |
| `step` raises | every running one (`env_step`) | next items |
| a slot's error in `VECTOR_RESULT` (`on_episode_start`, the model) | that slot's (`model_act`) | continue |
| round trip fails (connection closed, timeout, bad reply) | every running one, after the reconnect succeeded | new sessions; if the reconnect fails, as below |
| server unreachable, failed reconnect | every one the runner holds | partial result; their items stay for a rerun |

The unhealthy-shard check (RFC-0006) counts episodes as before; with
`--requeue-unhealthy` it also releases the shard's in-flight items.

### Transport

The harness's own cost per step, measured with a zero-cost environment and model (64-float
state observations, server in its own process):

| `num_envs` | one connection per slot (first design) | one `VECTOR_STEP` per step |
|---:|---:|---:|
| 1 | 2.2 ms | 1.7 ms |
| 64 | 77 ms | 7.1 ms |
| 512 | 533 ms | 32 ms |
| 4500 | — | 188 ms |

One connection per slot paid per message: three quarters for moving each message's encode and
decode to a thread, the rest asyncio and websockets work per message.  What remains, ~40-60 µs
per environment-step, is per-slot Python: a dict and an array per slot each way through msgpack,
`make_obs`, and the server's chunk handling (at 4500: 34 ms encode, 41 ms decode, 27 ms
`on_observations`, 39 ms in the benchmark's per-slot calls).  At a few hundred environments
this is about one MuJoCo Warp physics step.

`Connection.exchange` encodes a step on a worker thread: the default PNG encoding takes ~10 ms
per 224×224 frame.  `server.image_format: raw` sends images unencoded (lossless), the choice when
the server is on the same host or a fast network.

### Composition with sharding

Each shard process holds its own `num_envs` environments and connections, so shards ×
`num_envs` episodes run at once against one model server.

## Measured

One A100 80GB (PCIe), sync mode, a hold-style model server answering at once unless noted.  These
rows were measured with one connection per slot.

| Benchmark | Setup | Before (PR head) | After | Notes |
|-----------|-------|-----------------:|------:|-------|
| Synthetic (harness only) | 32 slots, 128 episodes of 200-400 steps, 10 ms step, chunk 10, server 60 ms + 1 ms/obs | 64.7 s waves, 170.8 s `partial_reset` | 56.7 s waves | with `partial_reset`, refills at arbitrary steps put the slots' chunks out of phase and split each step's inference into batches of 1-5 (below) |
| Synthetic + images | same, three 224×224 cameras | 1183 s | 250 s (PNG), 107 s (`raw`, `max_wait_time` 0.05) | PNG encoding serialised on the event loop |
| Shortstop (mjlab / MuJoCo Warp) | 32 worlds, 32 episodes × 430 steps, three cameras | 231 s, 72 env-steps/s, mean batch 1.2 | 100 s, 152 env-steps/s, batch 32 (`raw`) | 50 ms forward: 264 s → 122 s.  Every episode's outcome identical before and after.  The simulator is 56 s of the 100 s |
| RoboDojo (Isaac Sim 5.1 RTX, A100) | stack_blocks, 150 steps, three 640×480 cameras | 1.03 env-steps/s with one environment | 5.6 (`num_envs` 4), 7.3 (8), 10.6 (8, `raw`) | joint states match `num_envs=1`; frames do not (below) |
| RoboTwin 2.0 (SAPIEN 3 ray tracer) | beat_block_hammer, three 320×240 cameras, 32 spp + OIDN | 2.94 frames/s, one env | 2.9-2.96 with 2-8 envs in one process; 2.5 with 2-4 processes | no adapter: see below |

RoboTwin: the GPU is far from saturated (SM occupancy 0.12, 320×240 dispatches are small), but
SAPIEN 3.0.0b1 serialises every camera's ray tracing on one Vulkan queue (a barrier before each
`traceRays`, `get_picture` waits on the queue, OIDN denoises synchronously inside `take_picture`,
no GIL release), so N scenes' renders never overlap.  Overlapping physics with rendering reaches
+15 % but only without OIDN, which is not the protocol.  A vector adapter pays off only after a
SAPIEN change (asynchronous OIDN, or one launch for many cameras); until then `render_every` and
shards are the levers.

RoboDojo: in one Isaac scene with `num_envs > 1`, the tiled render products return wrist-camera
frames that are ~79 % black (opaque surfaces unshaded; the camera pose is right) and head-camera
frames off by 4-18/255; with `num_envs=1` they are fine, and joint states match to 3e-4 either
way.  The same on an A100 and an H100 (H100: 4.4 env-steps/s with one environment, 12.3 with 4,
15.0 with 8), and upstream's own `src/eval_client/main.py --num_envs 4` reproduces it, so it is
not the adapter.  The wrist near clip, six RTX settings (antialiasing, shadows/AO/GI, sampled
lighting, path tracing) and cleared Kit caches change nothing; one of about fifteen multi-env
runs rendered the wrists correctly, cause unknown.  Neither GPU has RT cores, which Isaac Sim 5.1
requires; until frames match on a supported GPU, the vector adapter's scores are not comparable
to `num_envs=1`.

## Alternatives

- **One connection per slot.** Every model server works unchanged, but each step is `num_envs`
  messages each way, and the harness cost grows ~1 ms per environment (table above).  The
  vector step keeps model servers unchanged too: the server gives each slot its own session.
- **One benchmark view per slot behind a barrier** (each slot driven by the single-episode
  runner, the views joined before each batched call).  It keeps one connection per slot and
  adds barrier, departure and timeout handling the vector runner does not need.
- **Threads around `StepBenchmark`.** GPU-parallel simulators step all environments in one
  call; one thread per environment cannot share that call.

## Limitations

- Sync mode only.  A real-time vector runner needs a per-slot action buffer under one
  shared clock; it is left for a later RFC.
- Lockstep: each step waits for the slowest slot's action.
- Without `partial_reset`, a slot that finishes early idles until its wave ends, so the idle
  share grows with the spread of episode lengths (episodes of 40 to 400 steps leave most
  slot-steps waiting).
- With `partial_reset`, `reset` is one synchronous call during which every running slot waits:
  cheap where an environment resets in place, not where a reset rebuilds a scene.
- `elapsed_sec` is wall time from the episode's reset, so it includes the other slots' resets
  and inference; it is not comparable with the one-at-a-time runner's value.
- An idle slot staying inert is the adapter's job; the harness cannot check it.  The benchmark
  learns that a slot's episode ended early (`max_steps`, a model error) only from that slot's
  absence in `actions`.
- With `partial_reset`, a chunking server's slots fall out of phase (each refill starts a new
  chunk), so a step infers the slots whose chunk ran out rather than all of them every chunk
  length: the same samples in more, smaller batches.  Not measured on a real model yet.
- Per-slot observations and actions cost ~40-60 µs per environment-step in the harness; a
  stacked-array path (observations and actions as `[num_envs, ...]` arrays end to end) would
  remove most of it at the price of a batch-shaped benchmark and model API.
- `Benchmark` and `VectorStepBenchmark` are siblings, so the orchestrator keeps two loops (they
  share the per-episode bookkeeping, `Orchestrator._episode_ended`).  A `StepBenchmark` wrapped as
  a one-slot vector benchmark would leave sync evaluation one runner and one loop; live mode
  needs the async `Benchmark` contract either way.  Left for a later change.
