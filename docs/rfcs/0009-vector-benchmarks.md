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
once on it over one connection, one `VECTOR_STEP` round trip per step.  Observations and
actions travel as columns (one array per key, a row per slot) from the simulator to the model
and back, so a step costs the harness about the same at 4500 environments as at 500.  Each
running episode is a row of the model server, whose lifecycle and inference are batched over rows; a
server that infers columns directly (`predict_batch`) runs at the simulator's own speed.

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
Batch = dict[str, Any]  # a column per key: an array whose first axis is the row, or a list; row i is slots[i]

class VectorStepBenchmark(BenchmarkCommon):
    partial_reset: ClassVar[bool] = False

    def __init__(self, num_envs: int = 1): ...
    def reset(self, slots: list[int], tasks: list[Task], recorders: list[EpisodeRecorder]) -> Batch: ...
    def step(self, slots: list[int], actions: Batch) -> tuple[Batch, np.ndarray]: ...  # observations, done
    def result(self, slot: int) -> EpisodeResult: ...
```

A slot is an environment index.  `reset` starts `tasks[i]` in slot `slots[i]` and returns the
model's first observation of each; the other slots' episodes go on as if nothing happened (an
adapter keeps the observations it returned for them: OCBench's `reset_worlds`, for one,
recomputes every world's).  `step` advances every environment once, `actions` row `i` for
`slots[i]`, and returns those slots' next observations and `done` flags; a slot not in `slots`
is idle and the benchmark keeps it inert, and the observation rows of ended slots are not used.
`result(slot)` gives the episode that just ended in `slot`, done or stopped at the harness's
`max_steps`, before the slot is reset again.  Lists carry what does not stack: images as
`{camera: HWC array}` (so the image codecs apply per frame), strings, actions of different
lengths.  Everything a benchmark
declares besides stepping (tasks, specs, metric keys, metadata, render backends, the
real-time hold, cleanup) moved to `BenchmarkCommon`, the shared parent of `Benchmark` and
`VectorStepBenchmark`; `Benchmark`'s interface is unchanged.

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

The runner keeps the running slots' observations as one `Batch` and touches single slots only
when an episode starts or ends: it selects or appends rows when that happens, never per step.
Its calls into the benchmark run on the orchestrator's thread, where the benchmark was built:
Isaac Sim's Kit loop, GL/Vulkan contexts and CUDA state are bound to the thread that created them,
and the event loop has nothing else to do while the simulator steps (one connection, one round
trip at a time).

### Protocol and server

```
VECTOR_STEP   {"ends": [[slot, result]], "starts": [[slot, episode_start]], "slots": [slot, ...], "observations": Batch}
VECTOR_RESULT {"actions": Batch, "errors": [[slot, detail]]}   # action rows: the slots without an error, in order
```

The server handles a step in that order: it ends the slots' episodes, starts new ones (one
`on_episode_start(rows)` for all of them; a position it fails is that slot's error), then hands the
observation columns to the model.  Every running episode is a row of the model server, a vector slot as much as
a single-episode connection: `rows.ids` are stable while an episode runs (reused after), so a policy keeps
per-episode state in arrays indexed by them, and the lifecycle and inference are batched alike.  The model
implements `predict(obs, row)` (one observation) or `predict_batch(obs, rows)` (columns in, columns out); the
framework slices batches by `max_batch_size`, serves chunked episodes from per-row buffers and infers only the
rows whose buffer ran out.  The same dispatcher coalesces single-episode connections' observations, so
sharded and vector evaluations reach the model the same way.  HELLO advertises `capabilities.vector_step`, and
the orchestrator refuses a server without it.  Messages other than `VECTOR_STEP` are unchanged, so the sync and
live runners are too.

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
| `step` raises | every running one (`env_step`) | next items |
| `result(slot)` raises | that slot's (`env_step`) | next items |
| a slot's error in `VECTOR_RESULT` (`on_episode_start`, the model) | that slot's (`model_act`) | continue |
| round trip fails (connection closed, timeout, bad reply) | every running one, after the reconnect succeeded | new sessions; if the reconnect fails, as below |
| server unreachable, failed reconnect | every one the runner holds | partial result; their items stay for a rerun |

The unhealthy-shard check (RFC-0006) counts episodes as before; with
`--requeue-unhealthy` it also releases the shard's in-flight items.

### Transport

The harness's own cost per step, measured with a zero-cost environment and model (64-float
state observations, server in its own process, a shared and loaded host):

| `num_envs` | one connection per slot (first design) | per-slot rows, one message | columns, per-row `predict` | columns, `predict_batch` |
|---:|---:|---:|---:|---:|
| 1 | 2.2 ms | 1.7 ms | 3.0 ms | 2.6 ms |
| 64 | 77 ms | 7.1 ms | 3.5 ms | 2.6 ms |
| 512 | 533 ms | 32 ms | 9.7 ms | 4.0 ms |
| 4500 | — | 188 ms | 33 ms | 13 ms |

One connection per slot paid per message: three quarters for moving each message's encode and
decode to a thread, the rest asyncio and websockets work per message.  Per-slot rows in one
message still paid ~40-60 µs per environment-step of Python (a dict and an array per slot each
way through msgpack, `make_obs`, the server's per-item handling).  Columns leave a fixed ~2-4 ms
a step (the round trip's thread hops: the server's decode and inference threads) and ~2 µs per environment (the server's per-row bookkeeping).

`Connection.exchange` encodes on the event loop, which has nothing else to do meanwhile.  The
default PNG encoding takes ~10 ms per 224×224 frame, so a step with many cameras is bound by
it; `server.image_format: raw` sends images unencoded (lossless), the choice for vector
benchmarks when the server is on the same host or a fast network.

### Composition with sharding

Each shard process holds its own `num_envs` environments and connections, so shards ×
`num_envs` episodes run at once against one model server.

## Measured

One A100 80GB (PCIe), sync mode, a hold-style model server answering at once unless noted.  The
first rows were measured with one connection per slot; the notes add later revisions.

| Benchmark | Setup | Before (PR head) | After | Notes |
|-----------|-------|-----------------:|------:|-------|
| Synthetic (harness only) | 32 slots, 128 episodes of 200-400 steps, 10 ms step, chunk 10, server 60 ms + 1 ms/obs | 64.7 s waves, 170.8 s `partial_reset` | 56.7 s waves | with `partial_reset`, refills at arbitrary steps put the slots' chunks out of phase and split each step's inference into batches of 1-5 (below) |
| Synthetic + images | same, three 224×224 cameras | 1183 s | 250 s (PNG), 107 s (`raw`, `max_wait_time` 0.05) | PNG encoding serialised on the event loop |
| Shortstop (mjlab / MuJoCo Warp) | 32 worlds, 32 episodes × 430 steps, three cameras | 231 s, 72 env-steps/s, mean batch 1.2 | 100 s, 152 env-steps/s, batch 32 (`raw`) | 50 ms forward: 264 s → 122 s.  Every episode's outcome identical before and after.  The simulator is 56 s of the 100 s.  With `VECTOR_STEP` (hold server, A100): 279 s at #90's first commit, 268 s (per-slot rows, PNG), 76 s (per-slot rows, `raw`), 72 s (columns, `raw`; the adapter ported), every episode's outcome identical |
| OCBench (MuJoCo Warp) | a fixed linear policy, model server with `predict_batch` over columns; OCBench's own loop runs it in-process | own loop, ms a step: block-single 31.9 (500 worlds) / 71.2 (4500), chamber-hard 50.4 / 162.5 | vla-eval: 36.1 / 83.2, 54.1 / 176.7 (+7 to +17 %; medians of three interleaved runs) | first observations bit-identical, first five steps equal; whole trajectories differ as between two runs of OCBench's loop (MuJoCo Warp is not deterministic).  The 4500-world runs (100 steps) include their wave's 4500 episode ends at once, ~3 ms a step of bookkeeping |
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
- **Per-slot rows in one message** (an observation dict per slot, `make_obs(raw, slot, task)` per
  slot): an earlier revision of this RFC.  32 ms a step at 512 environments, 188 ms at 4500, all
  per-slot Python; at a few hundred environments that is one MuJoCo Warp physics step.
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
  learns that a slot's episode ended early (a model error) only from that slot's absence in
  `slots`.
- With `partial_reset`, a chunking server's slots fall out of phase (each refill starts a new
  chunk), so a step infers the slots whose chunk ran out rather than all of them every chunk
  length: the same samples in more, smaller batches.  Not measured on a real model yet.
- A model server that only implements `predict` takes a dict per row: ~5-20 µs per
  environment-step (the split and the per-row calls).  `predict_batch` removes it.
- `Benchmark` and `VectorStepBenchmark` are siblings, so the orchestrator keeps two loops (they
  share the per-episode bookkeeping, `Orchestrator._episode_ended`).  A `StepBenchmark` wrapped as
  a one-slot vector benchmark would leave sync evaluation one runner and one loop; live mode
  needs the async `Benchmark` contract either way.  Left for a later change.
