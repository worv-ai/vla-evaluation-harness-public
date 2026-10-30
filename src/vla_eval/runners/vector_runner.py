"""VectorEpisodeRunner: episodes over a VectorStepBenchmark's environments, one server round trip per step."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import anyio
import anyio.to_thread

from vla_eval import watchdog
from vla_eval.benchmarks.base import VectorStepBenchmark
from vla_eval.recording import EpisodeRecorder, RecordingError
from vla_eval.runners.base import (
    EnvStartError,
    EnvStepError,
    EpisodeError,
    ModelActError,
    episode_start_payload,
    phase,
)
from vla_eval.types import Action, EpisodeResult, Observation, Task

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _try(error: type[EpisodeError], fn: Callable[..., Any], *args: Any) -> Any:
    """``fn(*args)``, or the exception it raised (tagged with ``error``'s phase); RecordingError propagates."""
    try:
        with phase(error):
            return fn(*args)
    except RecordingError:
        raise
    except Exception as exc:
        return exc


def _raise(exc: Exception) -> None:
    raise exc


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
    """Runs episodes on the slots of a :class:`VectorStepBenchmark` over one connection.

    A step is one round trip (``Connection.exchange``): the episodes ended since the last one,
    the ones just started and one observation per running slot go out together, the server
    infers them together, then every environment steps once.  Each slot's episode is its own
    server session.  Free slots take the next episode from ``next_episode(slot)`` (``None``
    when there is none now): at any step when the benchmark has ``partial_reset``, else when no
    slot runs.  After a ``None`` it asks again once an episode ends, and returns when no
    episode runs and none comes.

    Failures stay with their episode where they can: a per-slot exception returned by ``reset``
    and a server error for a slot end that episode.  An exception raised by ``reset`` or
    ``step`` ends every episode on the simulator, whose state is then unknown.  A failed round
    trip (closed connection, timeout, bad reply) ends every running episode, then the runner
    reconnects, with new sessions, and goes on.  When the reconnect fails or the server is
    unreachable (``ConnectionError``), every episode it holds ends aborted and
    ``ConnectionError`` is raised.  ``RecordingError`` and exceptions from ``on_end`` propagate.
    """

    def __init__(self) -> None:
        # Every call into the benchmark runs on this one thread: simulators keep thread-bound state
        # (GL/Vulkan contexts, Isaac Sim's Kit loop), and the event loop stays free meanwhile.
        self._sim = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vla-eval-sim")

    async def _call(self, fn: Callable[[], T]) -> T:
        return await anyio.to_thread.run_sync(self._sim.submit(fn).result)

    async def cleanup(self, benchmark: VectorStepBenchmark) -> None:
        """``benchmark.cleanup()`` on the simulator's thread, then release the thread."""
        try:
            await self._call(benchmark.cleanup)
        finally:
            self._sim.shutdown(wait=False)

    async def run(
        self,
        benchmark: VectorStepBenchmark,
        conn: Any,  # Connection
        next_episode: Callable[[int], VectorEpisode | None],
        on_end: OnEnd,
        *,
        max_steps: int | None = None,
    ) -> None:
        running: dict[int, _Running] = {}
        ends: list[list[Any]] = []  # [slot, result] of the server sessions to end in the next round trip
        exhausted = False

        def ended(episode: VectorEpisode, result: Any, error: Any, *, aborted: bool = False) -> None:
            nonlocal exhausted
            exhausted = False  # the caller may have work again (a held task, a released queue item)
            on_end(episode, result, error, aborted=aborted)

        try:
            while True:
                starts: list[list[Any]] = []
                if not exhausted and (not running or benchmark.partial_reset):
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
                        starts = await self._start(benchmark, new, running, ends, ended)
                if not running:
                    if exhausted:
                        break
                    continue
                actions = await self._exchange(conn, starts, running, ends, ended)
                if actions:
                    await self._step(benchmark, actions, running, ends, ended, max_steps)
        except ConnectionError as exc:
            for r in running.values():
                on_end(r.episode, None, exc, aborted=True)
            raise
        if ends:  # the last episodes' sessions; their results are booked already
            try:
                await conn.exchange({"ends": ends, "starts": [], "observations": []})
            except Exception as exc:
                logger.warning("Ending the last server sessions failed: %s: %s", type(exc).__name__, exc)

    async def _start(
        self,
        benchmark: VectorStepBenchmark,
        new: list[tuple[int, VectorEpisode]],
        running: dict[int, _Running],
        ends: list[list[Any]],
        on_end: OnEnd,
    ) -> list[list[Any]]:
        """Reset the new slots; return the ``EPISODE_START`` of each that started."""
        slots = [slot for slot, _ in new]
        episodes = [episode for _, episode in new]
        t0 = time.monotonic()

        def reset() -> list[Observation | Exception]:
            with phase(EnvStartError):
                raws = benchmark.reset(slots, [e.task for e in episodes], [e.recorder for e in episodes])
                if len(raws) != len(slots):
                    raise ValueError(f"reset() returned {len(raws)} observations for {len(slots)} slots")
            return [
                _try(EnvStartError, _raise, raw)  # the benchmark failed this slot alone
                if isinstance(raw, Exception)
                else _try(EnvStartError, benchmark.make_obs, raw, slot, e.task)
                for raw, slot, e in zip(raws, slots, episodes)
            ]

        try:
            observations = await self._call(reset)
        except RecordingError:
            raise
        except Exception as exc:
            for episode in episodes:
                on_end(episode, None, exc)
            self._end_running(running, ends, exc, on_end)
            return []
        starts: list[list[Any]] = []
        for slot, episode, obs in zip(slots, episodes, observations):
            if isinstance(obs, Exception):
                on_end(episode, None, obs)
                continue
            running[slot] = _Running(episode, obs, t0)
            starts.append([slot, episode_start_payload(episode.task, episode.recorder)])
        return starts

    async def _exchange(
        self,
        conn: Any,
        starts: list[list[Any]],
        running: dict[int, _Running],
        ends: list[list[Any]],
        on_end: OnEnd,
    ) -> dict[int, Action]:
        """One round trip; return the running slots' actions.  A slot without one ends with the server's error."""
        step = {"ends": list(ends), "starts": starts, "observations": [[s, r.obs] for s, r in running.items()]}
        ends.clear()
        try:
            reply = await conn.exchange(step)
        except (ConnectionError, RecordingError):
            raise
        except Exception as exc:  # closed, timed out or out of sync: the server's sessions are gone
            await self._reconnect(conn, running, exc, on_end)
            return {}
        actions = {slot: action for slot, action in reply.get("actions", [])}
        errors = {slot: detail for slot, detail in reply.get("errors", [])}
        for slot in [s for s in running if s not in actions]:
            r = running.pop(slot)
            on_end(r.episode, None, ModelActError(f"Server error: {errors.get(slot, 'no action for this slot')}"))
            ends.append([slot, {}])
        return {slot: actions[slot] for slot in running}

    async def _step(
        self,
        benchmark: VectorStepBenchmark,
        actions: dict[int, Action],
        running: dict[int, _Running],
        ends: list[list[Any]],
        on_end: OnEnd,
        max_steps: int | None,
    ) -> None:
        for slot in actions:
            running[slot].steps += 1

        def step() -> dict[int, tuple[bool, Any]]:
            """Step, then per slot ``(False, observation)`` to go on, ``(True, metrics)`` at its end, or an error."""
            with phase(EnvStepError):
                results = benchmark.step(actions)
            out: dict[int, Any] = {}
            for slot in actions:
                r = running[slot]

                def settle() -> tuple[bool, Any]:
                    if slot not in results:
                        raise KeyError(f"step() returned no result for slot {slot}")
                    sr = results[slot]
                    if not benchmark.check_done(sr) and (max_steps is None or r.steps < max_steps):
                        return False, benchmark.make_obs(sr.obs, slot, r.episode.task)
                    return True, benchmark.get_step_result(slot, sr)

                out[slot] = _try(EnvStepError, settle)
            return out

        try:
            outcomes = await self._call(step)
        except RecordingError:
            raise
        except Exception as exc:
            self._end_running(running, ends, exc, on_end)
            return
        watchdog.pet()  # a slow simulator's episode can outlast the stall timeout
        for slot, outcome in outcomes.items():
            r = running[slot]
            if isinstance(outcome, Exception):
                del running[slot]
                on_end(r.episode, None, outcome)
                ends.append([slot, {}])
                continue
            ended, value = outcome
            if not ended:
                r.obs = value
                continue
            del running[slot]
            result: EpisodeResult = {
                "metrics": value,
                "steps": r.steps,
                "elapsed_sec": round(time.monotonic() - r.t0, 3),
            }
            on_end(r.episode, result, None)
            ends.append([slot, result])

    @staticmethod
    def _end_running(running: dict[int, _Running], ends: list[list[Any]], exc: Exception, on_end: OnEnd) -> None:
        for slot, r in running.items():
            on_end(r.episode, None, exc)
            ends.append([slot, {}])
        running.clear()

    @staticmethod
    async def _reconnect(conn: Any, running: dict[int, _Running], exc: Exception, on_end: OnEnd) -> None:
        """End every running episode with ``exc`` after reconnecting; aborted when the reconnect fails."""
        try:
            await conn.reconnect()
        except Exception as err:
            for r in running.values():
                on_end(r.episode, None, exc, aborted=True)
            running.clear()
            if isinstance(err, ConnectionError):
                raise
            # e.g. a HELLO timeout: the server cannot take this runner's episodes
            raise ConnectionError(f"reconnect failed: {type(err).__name__}: {err}") from err
        for r in running.values():
            on_end(r.episode, None, exc)
        running.clear()
