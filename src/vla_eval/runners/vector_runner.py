"""VectorEpisodeRunner: episodes over a VectorStepBenchmark's environments, one model-server session each."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import anyio
import websockets

from vla_eval import watchdog
from vla_eval.benchmarks.base import VectorStepBenchmark
from vla_eval.recording import EpisodeRecorder, RecordingError
from vla_eval.runners.base import EnvStartError, EnvStepError, ModelActError, episode_start_payload, phase
from vla_eval.types import Action, EpisodeResult, Observation, Task


@dataclass
class VectorEpisode:
    """One episode for the runner; ``ref`` is the caller's handle, returned in ``on_end``."""

    task: Task
    recorder: EpisodeRecorder
    ref: Any = None


class OnEnd(Protocol):
    """``on_end(episode, result, error, aborted=...)``: exactly one of ``result`` / ``error`` is set.  ``aborted``:
    the run stops with this episode (the server is unreachable), so the episode is to be redone on a rerun."""

    def __call__(
        self,
        episode: VectorEpisode,
        result: EpisodeResult | None,
        error: BaseException | None,
        *,
        aborted: bool = False,
    ) -> None: ...


@dataclass
class _Running:
    episode: VectorEpisode
    obs: Observation
    t0: float
    steps: int = 0


class VectorEpisodeRunner:
    """Runs episodes on the slots of a :class:`VectorStepBenchmark`, slot ``i`` talking to ``conns[i]``.

    Each step sends every running slot's observation concurrently, so a batching model
    server infers them together, then steps all environments once.  Free slots take the
    next episode from ``next_episode(slot)`` (``None`` when there is none now): at once
    when the benchmark has ``partial_reset``, else when the whole wave has finished.  After
    a ``None`` it asks again once an episode ends (a shared queue can get work back from
    another shard), and returns when no episode runs and none comes.

    Failures stay with their episode, as in the sync runner: an environment error
    (``reset`` fails every episode it starts, ``step`` every running one), a model error,
    a closed connection or an ``act`` timeout (that slot reconnects) end that episode
    through ``on_end`` and the rest go on.  ``ConnectionError`` (the server is unreachable,
    or a reconnect failed) ends every episode the runner holds and is re-raised.
    ``RecordingError`` and exceptions raised by ``on_end`` propagate at once.
    """

    async def run(
        self,
        benchmark: VectorStepBenchmark,
        conns: Sequence[Any],  # Connection, one per slot
        next_episode: Callable[[int], VectorEpisode | None],
        on_end: OnEnd,
        *,
        max_steps: int | None = None,
    ) -> None:
        if len(conns) < benchmark.num_envs:
            raise ValueError(f"{benchmark.num_envs} environments need as many connections, got {len(conns)}")
        running: dict[int, _Running] = {}
        try:
            exhausted = False
            while True:
                if not exhausted and (benchmark.partial_reset or not running):
                    new: list[tuple[int, VectorEpisode]] = []
                    for slot in range(benchmark.num_envs):
                        if slot in running:
                            continue
                        episode = next_episode(slot)
                        if episode is None:
                            exhausted = True
                            break
                        new.append((slot, episode))
                    if new:
                        await self._start(benchmark, conns, new, running, on_end)
                if not running:
                    if exhausted:
                        return
                    continue
                before = len(running)
                actions = await self._act(conns, running, on_end)
                if actions:
                    await self._step(benchmark, conns, actions, running, on_end, max_steps)
                if len(running) < before:
                    exhausted = False
        except ConnectionError as exc:
            for r in running.values():
                on_end(r.episode, None, exc, aborted=True)
            raise

    async def _start(
        self,
        benchmark: VectorStepBenchmark,
        conns: Sequence[Any],
        new: list[tuple[int, VectorEpisode]],
        running: dict[int, _Running],
        on_end: OnEnd,
    ) -> None:
        slots = [slot for slot, _ in new]
        episodes = [episode for _, episode in new]
        t0 = time.monotonic()
        try:
            with phase(EnvStartError):
                raws = benchmark.reset(slots, [e.task for e in episodes], [e.recorder for e in episodes])
                if len(raws) != len(slots):
                    raise ValueError(f"reset() returned {len(raws)} observations for {len(slots)} slots")
        except RecordingError:
            raise
        except Exception as exc:  # an EpisodeError, or a transport-type error phase() lets through
            self._end_all(episodes, exc, on_end)
            return
        pending = list(zip(slots, episodes, raws))
        while pending:
            slot, episode, raw = pending.pop(0)
            try:
                with phase(EnvStartError):
                    obs = benchmark.make_obs(raw, slot, episode.task)
                await conns[slot].start_episode(episode_start_payload(episode.task, episode.recorder))
            except RecordingError:
                raise
            except Exception as exc:
                try:
                    await self._fail(conns[slot], episode, exc, on_end)
                except ConnectionError as abort:
                    for _, rest, _ in pending:
                        on_end(rest, None, abort, aborted=True)
                    raise
                continue
            running[slot] = _Running(episode, obs, t0)

    async def _act(self, conns: Sequence[Any], running: dict[int, _Running], on_end: OnEnd) -> dict[int, Action]:
        out: dict[int, Action | Exception] = {}

        async def act(slot: int, obs: Observation) -> None:
            try:
                with phase(ModelActError):
                    out[slot] = await conns[slot].act(obs)
            except Exception as exc:
                out[slot] = exc

        async with anyio.create_task_group() as tg:
            for slot, r in running.items():
                tg.start_soon(act, slot, r.obs)
        # ConnectionError last: it aborts the run, after the other failures are handled.
        failed = sorted(
            ((slot, x) for slot, x in out.items() if isinstance(x, Exception)),
            key=lambda p: isinstance(p[1], ConnectionError),
        )
        for slot, exc in failed:
            await self._fail(conns[slot], running.pop(slot).episode, exc, on_end)
        return {slot: a for slot, a in out.items() if not isinstance(a, Exception)}

    async def _step(
        self,
        benchmark: VectorStepBenchmark,
        conns: Sequence[Any],
        actions: dict[int, Action],
        running: dict[int, _Running],
        on_end: OnEnd,
        max_steps: int | None,
    ) -> None:
        try:
            with phase(EnvStepError):
                results = benchmark.step(actions)
        except RecordingError:
            raise
        except Exception as exc:
            self._end_all([running.pop(slot).episode for slot in actions], exc, on_end)
            return
        watchdog.pet()  # a slow simulator's episode can outlast the stall timeout
        for slot in actions:
            r = running[slot]
            r.steps += 1
            try:
                with phase(EnvStepError):
                    if slot not in results:
                        raise KeyError(f"step() returned no result for slot {slot}")
                    sr = results[slot]
                    if not benchmark.check_done(sr) and (max_steps is None or r.steps < max_steps):
                        r.obs = benchmark.make_obs(sr.obs, slot, r.episode.task)
                        continue
                    metrics = benchmark.get_step_result(slot, sr)
                del running[slot]
                result: EpisodeResult = {
                    "metrics": metrics,
                    "steps": r.steps,
                    "elapsed_sec": round(time.monotonic() - r.t0, 3),
                }
                await conns[slot].end_episode(result)
            except RecordingError:
                raise
            except Exception as exc:
                running.pop(slot, None)
                await self._fail(conns[slot], r.episode, exc, on_end)
                continue
            on_end(r.episode, result, None)

    @staticmethod
    def _end_all(episodes: list[VectorEpisode], exc: Exception, on_end: OnEnd) -> None:
        aborted = isinstance(exc, ConnectionError)
        for episode in episodes:
            on_end(episode, None, exc, aborted=aborted)
        if aborted:
            raise exc

    @staticmethod
    async def _fail(conn: Any, episode: VectorEpisode, exc: Exception, on_end: OnEnd) -> None:
        """End ``episode`` with ``exc``; a closed connection or a timeout reconnects the slot first, and the
        episode counts as aborted when that fails."""
        if isinstance(exc, (websockets.exceptions.ConnectionClosed, TimeoutError)):
            try:
                await conn.reconnect()
            except Exception as err:
                on_end(episode, None, exc, aborted=True)
                if isinstance(err, ConnectionError):
                    raise
                # e.g. a HELLO timeout: the server cannot take this slot's episodes
                raise ConnectionError(f"reconnect failed: {type(err).__name__}: {err}") from err
        aborted = isinstance(exc, ConnectionError)
        on_end(episode, None, exc, aborted=aborted)
        if aborted:
            raise exc
