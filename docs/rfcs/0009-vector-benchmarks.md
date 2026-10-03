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
once on it.  Each running episode keeps its own model-server session, so model servers,
the protocol and recording need no change, and a batching model server (RFC-0007) infers
the concurrent observations in one batch.

## Motivation

GPU-parallel simulators (MuJoCo Warp / mjlab, ManiSkill3 on the GPU, Isaac Lab) step
hundreds of environments in one kernel launch; one environment per process leaves the
device nearly idle and pays the per-step launch and synchronisation cost once per
environment.  Episode sharding (RFC-0006) parallelises across processes, each with its
own GPU context and a single environment, which is the expensive end of the trade.  On
an mjlab benchmark the harness ran an episode in about 40 s with one environment; the
same simulator rolls out 64 environments together at under 2 s per episode.

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

`partial_reset` declares per-environment resets.  With it, a slot that finishes takes the
next episode at once.  Without it, the runner resets only when no slot is running and runs
the items in waves of up to `num_envs`.

### Runner

`VectorEpisodeRunner` holds one `Connection` per slot.  Each step it sends every running
slot's observation concurrently (one anyio task per slot), steps the environments with the
returned actions, and ends the episodes that are done or reached `max_steps`.  The step
count, `max_steps` and `elapsed_sec` (wall time from the episode's reset) are per episode,
as in `SyncEpisodeRunner`.

One session per running episode keeps every per-session mechanism as it is: chunk buffers
and `ctx.is_first` in `PredictModelServer`, model-side history keyed by `ctx.session_id`,
and the `EPISODE_START`/`EPISODE_END` lifecycle.  `PredictModelServer` with
`max_batch_size >= num_envs` infers one step of all slots as one batch.

### Orchestrator

The orchestrator selects the vector path when the benchmark is a `VectorStepBenchmark`
(sync mode only; live mode raises before any episode).  It opens `num_envs` connections
and gives slot `i` its own recording `sid` (slot 0 keeps the shard's), because the server
replaces its session id with `recording.sid` from `EPISODE_START`; a shared sid would merge
the slots' sessions.  Work items, the shared work queue, recorders, results, progress and
trackers are handled per episode as in the one-at-a-time loop.  `RecordingStore.claim`
takes `exclude`, the shard's items still running, so a shard holding several claims is
never handed one of them again; a rerun after a crash still resumes its unfinished items.

Failures stay with their episode:

| Failure | Episodes that end | Then |
|---------|-------------------|------|
| `reset` raises | those it was starting (`env_start`) | next items |
| `make_obs` / `start_episode` raises | that slot's | next items |
| `step` raises | every running one (`env_step`) | next items |
| server `ERROR` reply | that slot's (`model_act`) | continue |
| connection closed / `act` timeout | that slot's | the slot reconnects |
| server unreachable, failed reconnect | every running one (`server_unreachable`) | partial result |

The unhealthy-shard check (RFC-0006) counts episodes as before; with
`--requeue-unhealthy` it also releases the shard's in-flight items.

### Composition with sharding

Each shard process holds its own `num_envs` environments and connections, so shards ×
`num_envs` episodes run at once against one model server.

## Alternatives

- **One connection carrying a batch of observations.** A new protocol message with a slot
  index would need batch-aware session state in every model server; per-slot connections
  reuse the existing per-session code and the server's batching.
- **Threads around `StepBenchmark`.** GPU-parallel simulators step all environments in one
  call; one thread per environment cannot share that call.

## Limitations

- Sync mode only.  A real-time vector runner needs a per-slot action buffer under one
  shared clock; it is left for a later RFC.
- Lockstep: each step waits for the slowest slot's action.
- Without `partial_reset`, a slot that finishes early idles until its wave ends.
