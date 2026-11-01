"""Sync sim time: the framework moves a sync episode's clock to ``k * step_dt`` before observation ``k`` is
delivered, waking one sleeper at a time and settling its model calls; live episodes keep wall time."""

from __future__ import annotations

import asyncio
import contextvars
import random
import sys
import time
from typing import Any

import anyio
from anyio.lowlevel import checkpoint
import numpy as np
import pytest
from chrono_daemon import WallClock

from vla_eval.benchmarks.base import VectorStepBenchmark
from vla_eval.connection import Connection
from vla_eval.model_servers.daemons import EpisodeContext, ObservationRequest, SimClock
from vla_eval.recording import NullEpisodeRecorder
from vla_eval.runners.base import ModelActError
from vla_eval.runners.live_runner import LiveEpisodeRunner
from vla_eval.runners.sync_runner import SyncEpisodeRunner
from vla_eval.runners.vector_runner import VectorEpisode, VectorEpisodeRunner

from tests.conftest import Daemon, StubBenchmark, echo, observe, running, serving, start

ZERO = {"actions": np.zeros(2, np.float32)}


class Stepped(StubBenchmark):
    step_dt = 0.1


async def _reply(req: ObservationRequest) -> None:
    await req.reply(ZERO)


# ---------------------------------------------------------------------------
# Observation k at k * step_dt; timers fire on their deadlines, before the observation due with them
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_observation_k_is_handled_at_k_step_dt(free_port):
    seen: list[tuple[int, float, float]] = []

    async def run(ep: EpisodeContext) -> None:
        async for req in ep.observations():
            seen.append((req.row.step, ep.clock.now(), req.admitted_at))
            await _reply(req)

    async with running(Daemon(run), free_port) as url, Connection(url) as conn:
        result = await SyncEpisodeRunner().run_episode(Stepped(done_at_step=5), {"name": "t"}, conn)
    assert result["steps"] == 5
    assert seen == [(k, k * 0.1, k * 0.1) for k in range(5)]


@pytest.mark.parametrize("period", [0.0625, 0.25, 0.375, 1.0])
@pytest.mark.anyio
async def test_every_fires_once_per_deadline_even_several_per_step(period):
    events: list[tuple[float, str]] = []

    async def ticker(ep: EpisodeContext) -> None:
        async for t in ep.clock.every(period):
            assert ep.clock.now() == t
            events.append((t, "fire"))

    async def run(ep: EpisodeContext) -> None:
        ep.start_soon(ticker, ep)
        async for req in ep.observations():
            events.append((ep.clock.now(), "obs"))
            await _reply(req)

    server = Daemon(run, step_dt=0.25)
    async with serving(server):
        (rid,) = await start(server)
        for v in range(9):  # the last at 2.0
            await observe(server, rid, v)
    fires = [t for t, kind in events if kind == "fire"]
    assert fires == [period * n for n in range(1, int(2.0 / period) + 1)]
    assert events == sorted(events, key=lambda e: (e[0], e[1] == "obs"))  # a fire due with an observation precedes it


@pytest.mark.anyio
async def test_decimal_periods_land_on_their_observations_without_drift():
    events: list[tuple[float, str]] = []

    async def ticker(ep: EpisodeContext, period: float) -> None:
        async for _ in ep.clock.every(period):
            events.append((round(ep.clock.now(), 9), f"every {period}"))

    async def run(ep: EpisodeContext) -> None:
        ep.start_soon(ticker, ep, 0.1)
        ep.start_soon(ticker, ep, 0.5)
        async for req in ep.observations():
            events.append((round(ep.clock.now(), 9), "obs"))
            await _reply(req)

    server = Daemon(run, step_dt=0.02)  # 3 * 0.1 > 15 * 0.02 in floats
    async with serving(server):
        (rid,) = await start(server)
        for v in range(151):  # the last at 3.0
            await observe(server, rid, v)
    kinds = [kind for _, kind in events]
    assert kinds.count("every 0.1") == 30 and kinds.count("every 0.5") == 6
    assert events == sorted(events, key=lambda e: (e[0], e[1] == "obs"))  # each fire before its instant's observation


@pytest.mark.anyio
async def test_async_work_and_child_tasks_before_a_call_finish_before_time_moves_on():
    done: list[tuple[str, float, float]] = []  # (who, tick, clock when its call returned)
    admitted: list[float] = []

    async def work(who: str, t: float, ep: EpisodeContext) -> None:
        for _ in range(50):  # async preprocessing: many scheduler hops before the model call
            await checkpoint()
        await ep.call(time.sleep, 0.002)
        done.append((who, t, ep.clock.now()))

    async def ticker(ep: EpisodeContext) -> None:
        async for t in ep.clock.every(0.25):
            ep.start_soon(work, "child", t, ep)  # handed off to another task
            await work("ticker", t, ep)

    async def run(ep: EpisodeContext) -> None:
        ep.start_soon(ticker, ep)
        async for req in ep.observations():
            admitted.append(ep.clock.now())
            assert all(t <= ep.clock.now() for _, t, _ in done)
            assert len(done) == 2 * int(ep.clock.now() / 0.25)  # every fire due so far has finished
            await _reply(req)

    server = Daemon(run, step_dt=0.125)
    async with serving(server):
        (rid,) = await start(server)
        for v in range(9):  # the last at 1.0
            await observe(server, rid, v)
    assert admitted == [k * 0.125 for k in range(9)]
    assert sorted(done) == sorted((who, t, t) for who in ("child", "ticker") for t in (0.25, 0.5, 0.75, 1.0))


@pytest.mark.parametrize("spawn", ["task group", "gather", "create_task"])
@pytest.mark.anyio
async def test_tasks_an_episode_task_creates_hold_time_too(spawn):
    done: list[tuple[float, float]] = []  # (tick, clock when its call returned)
    keep: list[asyncio.Task[None]] = []

    async def work(t: float, ep: EpisodeContext) -> None:
        for _ in range(50):
            await checkpoint()
        await ep.call(time.sleep, 0.002)
        done.append((t, ep.clock.now()))

    async def ticker(ep: EpisodeContext) -> None:
        if spawn == "task group":  # as a chrono Supervisor hosts its daemons
            async with anyio.create_task_group() as tg:
                async for t in ep.clock.every(0.25):
                    tg.start_soon(work, t, ep)
        async for t in ep.clock.every(0.25):
            if spawn == "gather":
                await asyncio.gather(work(t, ep))
            else:
                keep.append(asyncio.create_task(work(t, ep)))

    async def run(ep: EpisodeContext) -> None:
        ep.start_soon(ticker, ep)
        async for req in ep.observations():
            assert len(done) == int(ep.clock.now() / 0.25)
            await _reply(req)

    server = Daemon(run, step_dt=0.125)
    async with serving(server):
        (rid,) = await start(server)
        for v in range(9):
            pending = await observe(server, rid, v)
            assert not pending.errors
    assert done == [(t, t) for t in (0.25, 0.5, 0.75, 1.0)]


@pytest.mark.parametrize("where", ["inside, fresh context", "outside, the episode's context"])
@pytest.mark.anyio
async def test_a_task_belongs_to_the_episode_of_its_context(where):
    contexts: list[contextvars.Context] = []
    keep: list[asyncio.Task[None]] = []

    async def spin() -> None:
        while True:  # never blocks: fails the episode it belongs to, once that episode's time moves
            await checkpoint()

    async def run(ep: EpisodeContext) -> None:
        contexts.append(contextvars.copy_context())
        if where.startswith("inside"):
            keep.append(asyncio.get_running_loop().create_task(spin(), context=contextvars.Context()))
        async for req in ep.observations():
            await _reply(req)

    server = Daemon(run, step_dt=0.25)
    async with serving(server):
        (rid,) = await start(server)
        await observe(server, rid, 0)
        if where.startswith("outside"):
            keep.append(asyncio.get_running_loop().create_task(spin(), context=contexts[0]))
        pending = await observe(server, rid, 1)
        for task in keep:
            task.cancel()
    if where.startswith("inside"):
        assert not pending.errors
    else:
        assert "sim time cannot move on" in str(pending.errors[0])


@pytest.mark.anyio
async def test_a_task_started_on_another_episode_belongs_to_that_episode():
    eps: dict[int, EpisodeContext] = {}
    go = anyio.Event()

    async def spin() -> None:
        while True:
            await checkpoint()

    async def run(ep: EpisodeContext) -> None:
        eps[ep.row_id] = ep
        async for req in ep.observations():
            if req.row.step == 1 and ep.row_id == a:
                eps[b].start_soon(spin)  # from a's task, onto b
                go.set()
            await _reply(req)

    server = Daemon(run, step_dt=0.25)
    async with serving(server):
        a, b = await start(server, n=2)
        await observe(server, b, 0)
        for v in range(2):
            assert not (await observe(server, a, v)).errors
        await go.wait()
        for v in range(2, 5):  # a's time moves on while b's spinner runs
            assert not (await observe(server, a, v)).errors
        failed = await observe(server, b, 1)
    assert "sim time cannot move on" in str(failed.errors[0])


@pytest.mark.skipif(sys.version_info < (3, 12), reason="eager task factories are Python 3.12+")
@pytest.mark.anyio
async def test_episodes_keep_an_eager_task_factory_working():
    loop = asyncio.get_running_loop()
    done: list[tuple[int, float, float]] = []

    async def work(rid: int, t: float, ep: EpisodeContext) -> None:
        with anyio.CancelScope():  # anyio's cancel-scope bookkeeping needs its non-eager child construction
            for _ in range(50):
                await checkpoint()
            await ep.call(time.sleep, 0.002)
            done.append((rid, t, ep.clock.now()))

    async def ticker(ep: EpisodeContext) -> None:
        async with anyio.create_task_group() as tg:
            async for t in ep.clock.every(0.25):
                tg.start_soon(work, ep.row_id, t, ep)

    async def run(ep: EpisodeContext) -> None:
        ep.start_soon(ticker, ep)
        async for req in ep.observations():
            await _reply(req)

    previous = loop.get_task_factory()
    loop.set_task_factory(asyncio.eager_task_factory)  # ty: ignore[unresolved-attribute]
    try:
        server = Daemon(run, step_dt=0.125)
        async with serving(server):
            rids = await start(server, n=2)
            for v in range(5):
                for rid in rids:
                    pending = await observe(server, rid, v)
                    assert not pending.errors
    finally:
        loop.set_task_factory(previous)
    assert sorted(done) == sorted((rid, t, t) for rid in rids for t in (0.25, 0.5))


@pytest.mark.anyio
async def test_a_fires_model_call_finishes_before_time_moves_on():
    calls: list[tuple[float, float, float]] = []  # (tick, clock at the call's end, value)

    def slow(v: float) -> float:
        time.sleep(0.01)
        return v

    async def ticker(ep: EpisodeContext) -> None:
        async for t in ep.clock.every(0.0625):
            v = await ep.call(slow, t)
            v = await ep.call(slow, v)  # a second call after the first: still the same instant
            calls.append((t, ep.clock.now(), v))

    async def run(ep: EpisodeContext) -> None:
        ep.start_soon(ticker, ep)
        async for req in ep.observations():
            await _reply(req)

    server = Daemon(run, step_dt=0.25)
    async with serving(server):
        (rid,) = await start(server)
        for v in range(5):  # the last at 1.0
            await observe(server, rid, v)
    assert calls == [(0.0625 * n, 0.0625 * n, 0.0625 * n) for n in range(1, 17)]


@pytest.mark.anyio
async def test_a_sleeper_cancelled_while_asleep_is_never_woken():
    log: list[str] = []

    async def quitter(ep: EpisodeContext) -> None:
        with anyio.move_on_after(0.01):  # wall time: gives up before sim time reaches 0.3
            await ep.clock.sleep(0.3)
            log.append("woken")
        log.append(f"gave up at {ep.clock.now()}")

    async def run(ep: EpisodeContext) -> None:
        ep.start_soon(quitter, ep)
        async for req in ep.observations():
            if req.row.step == 2:
                log.append(f"obs at {ep.clock.now()}")
            await _reply(req)

    server = Daemon(run, step_dt=0.25)
    async with serving(server):
        (rid,) = await start(server)
        await observe(server, rid, 0)
        await anyio.sleep(0.05)
        for v in range(1, 3):
            await observe(server, rid, v)
    assert log == ["gave up at 0.0", "obs at 0.5"]


@pytest.mark.anyio
async def test_a_task_that_never_blocks_fails_the_episode_instead_of_skipping_time():
    async def ticker(ep: EpisodeContext) -> None:
        async for _ in ep.clock.every(0.1):
            for _ in range(10_050):  # never blocks within the settle bound
                await checkpoint()

    async def run(ep: EpisodeContext) -> None:
        ep.start_soon(ticker, ep)
        async for req in ep.observations():
            await _reply(req)

    server = Daemon(run, step_dt=0.25)
    async with serving(server):
        (rid,) = await start(server)
        await observe(server, rid, 0)
        pending = await observe(server, rid, 1)
    assert "sim time cannot move on" in str(pending.errors[0])


@pytest.mark.anyio
async def test_a_task_that_never_blocks_holds_only_its_own_episode():
    go = anyio.Event()
    busy: list[int] = []  # the spinning episode's row id
    ticks: list[float] = []

    async def spin(ep: EpisodeContext) -> None:
        await go.wait()
        if ep.row_id in busy:
            while True:  # never blocks
                await checkpoint()

    async def ticker(ep: EpisodeContext) -> None:
        async for t in ep.clock.every(0.1):
            ticks.append(t)

    async def run(ep: EpisodeContext) -> None:
        ep.start_soon(spin, ep)
        ep.start_soon(ticker, ep)
        async for req in ep.observations():
            await _reply(req)

    server = Daemon(run, step_dt=0.25)
    async with serving(server):
        a, b = await start(server, n=2)
        busy.append(a)
        go.set()
        for v in range(5):  # b's time moves while a's task spins
            pending = await observe(server, b, v)
            assert not pending.errors
        failed = await observe(server, a, 0)
        after = await observe(server, b, 5)
    assert "sim time cannot move on" in str(failed.errors[0])
    assert not after.errors
    assert ticks == [0.1 * n for n in range(1, 13)]  # b's ticker; a's never reached a deadline


# ---------------------------------------------------------------------------
# A dual system (S2 every 0.5 s, S1 every 0.1 s, each observation answered from S1's latest chunk) is deterministic
# ---------------------------------------------------------------------------


class Reflex(Daemon):
    """S2 plans every 0.5 s from the latest observation, S1 refines every 0.1 s from the latest plan, and each
    observation is answered from S1's latest chunk (a hold before the first).  Real model latency is random."""

    def __init__(self, seed: int) -> None:
        super().__init__(step_dt=0.05)
        self.rng = random.Random(seed)
        self.schedule: list[tuple[str, float, float]] = []

    def _infer(self, x: float, latency: float) -> float:
        time.sleep(latency)
        return x + 1.0

    async def run(self, ep: EpisodeContext) -> None:
        latest = {"obs": -1.0, "plan": -1.0, "chunk": -1.0}

        async def s2() -> None:
            async for t in ep.clock.every(0.5, first="now"):
                seen = latest["obs"]
                latest["plan"] = await ep.call(self._infer, seen, self.rng.uniform(0, 0.02))
                self.schedule.append(("s2", t, seen))

        async def s1() -> None:
            async for t in ep.clock.every(0.1):
                seen = latest["plan"]
                latest["chunk"] = await ep.call(self._infer, seen, self.rng.uniform(0, 0.005))
                self.schedule.append(("s1", t, seen))

        ep.start_soon(s2)
        ep.start_soon(s1)
        async for req in ep.observations():
            latest["obs"] = float(req.obs["value"])
            self.schedule.append(("reply", ep.clock.now(), latest["chunk"]))
            await req.reply({"actions": np.full(2, latest["chunk"], np.float32)})


@pytest.mark.anyio
async def test_a_reflex_dual_loop_runs_the_same_schedule_twice():
    schedules = []
    for seed in (0, 1):
        server = Reflex(seed)
        async with serving(server):
            (rid,) = await start(server)
            for k in range(24):  # 1.15 s of sim time
                await observe(server, rid, float(k))
        schedules.append(server.schedule)
    assert schedules[0] == schedules[1]
    kinds = [kind for kind, _, _ in schedules[0]]
    assert kinds.count("s2") == 3 and kinds.count("s1") == 11 and kinds.count("reply") == 24
    plans = {t: seen + 1.0 for kind, t, seen in schedules[0] if kind == "s2"}
    s1 = {t: seen for kind, t, seen in schedules[0] if kind == "s1"}
    assert s1[0.5] == plans[0.5]  # due together: S2 (asleep first) is woken and settled first


# ---------------------------------------------------------------------------
# Vector slots: each episode's clock on its own
# ---------------------------------------------------------------------------


class SteppedVector(VectorStepBenchmark):
    """Slot ``s`` runs task ``i``'s episode for ``i + 2`` steps; slots restart as soon as they end."""

    partial_reset = True
    step_dt = 0.25

    def __init__(self) -> None:
        super().__init__(num_envs=2)
        self.t, self.length = [0, 0], [0, 0]

    def get_tasks(self) -> list[dict[str, Any]]:
        return [{"name": f"task_{i}", "length": i + 2} for i in range(3)]

    def reset(self, slots, tasks, recorders):
        for slot, task in zip(slots, tasks):
            self.t[slot], self.length[slot] = 0, task["length"]
        return {"value": np.array([float(t["length"]) for t in tasks])}

    def step(self, slots, actions):
        done = []
        for slot in slots:
            self.t[slot] += 1
            done.append(self.t[slot] >= self.length[slot])
        return {"value": np.array([float(self.length[s]) for s in slots])}, np.array(done)

    def result(self, slot):
        return {"success": True}

    def get_metadata(self) -> dict[str, Any]:
        return {"max_steps": 10}


@pytest.mark.anyio
async def test_vector_slots_advance_their_own_clocks(free_port):
    seen: dict[float, list[tuple[int, float]]] = {}  # by episode length: (step, clock)
    configs: list[Any] = []

    async def run(ep: EpisodeContext) -> None:
        configs.append(ep.episode.config.get("step_dt"))
        async for req in ep.observations():
            seen.setdefault(float(req.obs["value"]), []).append((req.row.step, ep.clock.now()))
            await req.reply(ZERO)

    class Vector(Daemon):
        supports_vector_step = True

    queue = [{"name": f"task_{i}", "length": i + 2, "episode_idx": 0} for i in (0, 2, 1)]
    ended: list[Any] = []

    def next_episode(slot: int) -> VectorEpisode | None:
        return VectorEpisode(queue.pop(0), NullEpisodeRecorder()) if queue else None

    def on_end(episode, result, error, *, aborted=False):
        ended.append(error)

    async with running(Vector(run), free_port) as url, Connection(url) as conn:
        await VectorEpisodeRunner().run(SteppedVector(), conn, next_episode, on_end)
    assert ended == [None] * 3 and configs == [0.25] * 3
    # lengths 2, 4, 3: the length-3 episode starts mid-run in the freed slot, its clock fresh at 0
    assert seen == {n: [(k, k * 0.25) for k in range(int(n))] for n in (2.0, 4.0, 3.0)}


# ---------------------------------------------------------------------------
# Unknown step_dt: clock use fails the episode; loops that never touch the clock work
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_without_step_dt_using_the_clock_fails_the_episode_and_other_loops_work(free_port):
    async def timed(ep: EpisodeContext) -> None:
        async for req in ep.observations():
            ep.clock.now()
            await _reply(req)

    async def stamped(ep: EpisodeContext) -> None:
        async for req in ep.observations():
            req.admitted_at
            await _reply(req)

    for run in (timed, stamped):
        server = Daemon(run)
        async with running(server, free_port) as url, Connection(url) as conn:
            with pytest.raises(ModelActError, match="step_dt unknown"):
                await SyncEpisodeRunner().run_episode(StubBenchmark(), {"name": "t"}, conn)
    async with running(Daemon(echo), free_port) as url, Connection(url) as conn:
        result = await SyncEpisodeRunner().run_episode(StubBenchmark(done_at_step=3), {"name": "t"}, conn)
    assert result["steps"] == 3


# ---------------------------------------------------------------------------
# step_dt on the wire: each runner's EPISODE_START, the server's default; live keeps wall time
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_step_dt_reaches_the_episode_from_each_runner_or_the_server(free_port):
    seen: list[tuple[Any, float | None, type]] = []

    async def run(ep: EpisodeContext) -> None:
        seen.append((ep.episode.config.get("step_dt"), ep.step_dt, type(ep.clock)))
        await echo(ep)

    async with running(Daemon(run, step_dt=0.5), free_port) as url, Connection(url) as conn:
        await SyncEpisodeRunner().run_episode(Stepped(), {"name": "t"}, conn, max_steps=2)
        await SyncEpisodeRunner().run_episode(StubBenchmark(), {"name": "t"}, conn, max_steps=2)
        await LiveEpisodeRunner(hz=50.0).run_episode(Stepped(), {"name": "t"}, conn, max_steps=2)
        await conn.start_episode({"step_dt": None})  # null: the server's default
        await conn.act({"value": 1.0})
    assert seen == [
        (0.1, 0.1, SimClock),
        (None, 0.5, SimClock),
        (0.02, 0.02, WallClock),  # live: 1 / hz, wall time
        (None, 0.5, SimClock),
    ]
