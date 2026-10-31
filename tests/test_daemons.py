"""Episode daemons: one scope per episode (mailbox, tasks, model calls, clock), the one exception policy and the
teardown order; PredictionEngine's raw inference; chrono's SupervisorHost inside an episode."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from typing import Any, cast

import anyio
import numpy as np
import pytest
from chrono_daemon import SimClock, Supervisor, WallClock
from chrono_daemon.recipes.supervisor_host import SupervisorHost

from vla_eval import columns
from vla_eval.connection import Connection
from vla_eval.model_servers.base import EpisodeClosed, ModelServer, Rows, _Pending
from vla_eval.model_servers.daemons import (
    DaemonModelServer,
    EpisodeContext,
    EpisodeDaemons,
    Mailbox,
    MailboxOverflow,
    ObservationRequest,
    episode_daemon,
    run_predict,
)
from vla_eval.model_servers.engine import PredictionEngine
from vla_eval.protocol.messages import MessageType
from vla_eval.runners.live_runner import LiveEpisodeRunner
from vla_eval.runners.sync_runner import SyncEpisodeRunner

from tests.conftest import StubBenchmark, start_server, stop_server

LIVE = {"task": {"name": "t"}, "mode": "live"}


class Daemon(DaemonModelServer):
    """``run`` given per test; logs resets and finishes with their thread."""

    def __init__(self, run: Any = None, **kwargs: Any) -> None:
        super().__init__(run, **kwargs)
        self.log: list[tuple[str, list[int]]] = []
        self.threads: set[str] = set()
        self.eps: list[EpisodeContext] = []
        self.reset_fail: set[int] = set()  # positions to fail

    def reset_episodes(self, rows: Rows) -> dict[int, Exception] | None:
        self.threads.add(threading.current_thread().name)
        self.log.append(("reset", rows.ids.tolist()))
        return {i: ValueError("cannot reset") for i in self.reset_fail if i < len(rows)}

    def finish_episodes(self, rows: Rows, results: list[dict[str, Any]]) -> None:
        self.threads.add(threading.current_thread().name)
        self.log.append(("finish", rows.ids.tolist()))


def _req(v: float = 0.0) -> ObservationRequest:
    return ObservationRequest({"value": v}, {}, cast(Any, None), 0, 0.0, 0.0, None, cast(Any, None))


@contextlib.asynccontextmanager
async def serving(server: ModelServer):
    """``server`` as serve_async runs it, without a socket: yields its task group."""
    async with anyio.create_task_group() as tg:
        server._serve_tg = tg
        await server.on_serve_start(tg)
        try:
            yield tg
        finally:
            tg.cancel_scope.cancel()


async def _start(server: ModelServer, n: int = 1, config: dict[str, Any] | None = None, **kw: Any) -> list[int]:
    out = await server._start_episodes([(dict(config or {}), "s")] * n, **kw)
    ids = [r for r in out if isinstance(r, int)]
    assert len(ids) == len(out), out
    return ids


async def _observe(server: ModelServer, rid: int, value: float, *, wait: bool = True) -> _Pending:
    """One sync observation; ``wait=False`` returns at once and leaves it tracked, as a waiting connection does."""
    pending = _Pending(1, seq=1)
    server._open.add(pending)
    await server._observe(columns.stack([{"value": np.float32(value)}]), [rid], pending)
    if wait:
        with anyio.fail_after(3):
            await pending.done.wait()
        server._open.discard(pending)
    return pending


def _value(pending: _Pending) -> float:
    return float(np.asarray(pending.result()["actions"]).reshape(-1)[0])


async def echo(ep: EpisodeContext) -> None:
    async for req in ep.observations():
        await req.reply({"actions": np.full(2, float(req.obs["value"]), np.float32)})


# ---------------------------------------------------------------------------
# Mailbox
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_latest_keeps_the_newest_and_counts_the_dropped():
    box = Mailbox("latest")
    for v in (1.0, 2.0, 3.0):
        box.put(_req(v))
    assert (await box.receive()).obs["value"] == 3.0 and box.dropped == 2
    assert box.receive_nowait() is None


@pytest.mark.anyio
async def test_a_fifo_keeps_every_admission_until_its_bound():
    box = Mailbox(2)
    box.put(_req(1.0))
    box.put(_req(2.0))
    with pytest.raises(MailboxOverflow):
        box.put(_req(3.0))
    assert [(await box.receive()).obs["value"] for _ in range(2)] == [1.0, 2.0]
    box.put(_req(4.0))
    box.close()
    with pytest.raises(EpisodeClosed):  # waiting items are dropped
        await box.receive()


@pytest.mark.anyio
async def test_a_mailbox_has_one_receiving_task():
    box = Mailbox("latest")
    assert box.receive_nowait() is None
    errors: list[Exception] = []

    async def other() -> None:
        try:
            await box.receive()
        except RuntimeError as exc:
            errors.append(exc)

    async with anyio.create_task_group() as tg:
        tg.start_soon(other)
    assert errors and "one receiving task" in str(errors[0])


@pytest.mark.parametrize("backlog", [0, -1, True, "newest", 1.5])
def test_a_bad_backlog_is_refused(backlog):
    with pytest.raises(ValueError):
        Mailbox(backlog)


# ---------------------------------------------------------------------------
# Lifecycle and teardown order
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_several_admissions_before_the_owner_wakes_reach_it_in_order():
    seen: list[float] = []

    async def run(ep: EpisodeContext) -> None:
        await ep.call(time.sleep, 0.2)  # observations pile up meanwhile
        async for req in ep.observations():
            seen.append(float(req.obs["value"]))
            await req.reply({"actions": np.zeros(2, np.float32)})

    server = Daemon(run, backlog=8)
    async with serving(server):
        (rid,) = await _start(server, mode="live")
        for v in (1.0, 2.0, 3.0):
            await _observe(server, rid, v, wait=False)
        with anyio.fail_after(3):
            while len(seen) < 3:
                await anyio.sleep(0.01)
    assert seen == [1.0, 2.0, 3.0]


@pytest.mark.anyio
async def test_end_during_a_model_call_finishes_after_it_and_frees_the_row():
    order: list[str] = []

    def slow() -> str:
        time.sleep(0.2)
        order.append("call done")
        return "late"

    async def run(ep: EpisodeContext) -> None:
        async for req in ep.observations():
            try:
                await ep.call(slow)
            except EpisodeClosed:
                order.append("discarded")
                raise
            await req.reply({"actions": np.zeros(2, np.float32)})

    server = Daemon(run)
    async with serving(server):
        (rid,) = await _start(server)
        pending = await _observe(server, rid, 1.0, wait=False)
        await anyio.sleep(0.05)
        await server._end_episodes([rid], [{"status": "aborted"}])
        order.append("ended")
        assert order == ["call done", "discarded", "ended"] and server.log[-1] == ("finish", [rid])
        assert not pending.done.is_set() and server._running() == 0
        assert await _start(server) == [rid]  # reusable


@pytest.mark.anyio
async def test_closure_before_the_host_runs_spawns_no_root():
    ran: list[int] = []

    async def run(ep: EpisodeContext) -> None:
        ran.append(ep.row_id)

    server = Daemon(run)
    async with serving(server):
        (rid,) = await _start(server)
        await server._end_episodes([rid], [{}])  # no checkpoint between: the host never entered
        await anyio.sleep(0.05)
    assert ran == [] and server.log == [("reset", [rid]), ("finish", [rid])]


@pytest.mark.anyio
async def test_a_root_returning_with_no_task_left_fails_the_episode():
    async def run(ep: EpisodeContext) -> None:
        return

    server = Daemon(run)
    async with serving(server):
        (rid,) = await _start(server)
        await anyio.sleep(0.05)
        assert "no task is left" in str(server._failed[rid])
        pending = await _observe(server, rid, 1.0)  # later observations fail with that error
        assert "no task is left" in str(pending.errors[0])


@pytest.mark.anyio
async def test_a_child_tasks_exception_fails_only_its_episode():
    async def boom() -> None:
        raise ValueError("child exploded")

    async def run(ep: EpisodeContext) -> None:
        if ep.episode.config.get("boom"):
            ep.start_soon(boom)
        await echo(ep)

    server = Daemon(run)
    async with serving(server):
        bad, good = (await _start(server, config={"boom": True}))[0], (await _start(server))[0]
        await anyio.sleep(0.05)
        assert isinstance(server._failed.get(bad), ValueError) and good not in server._failed
        assert _value(await _observe(server, good, 2.0)) == 2.0
        assert "child exploded" in str((await _observe(server, bad, 1.0)).errors[0])


@pytest.mark.anyio
async def test_server_shutdown_is_teardown_not_failure():
    calls: list[str] = []

    def slow() -> None:
        time.sleep(0.2)
        calls.append("done")

    async def run(ep: EpisodeContext) -> None:
        async for _ in ep.observations():
            await ep.call(slow)

    server = Daemon(run)
    async with serving(server):
        (rid,) = await _start(server)
        await _observe(server, rid, 1.0, wait=False)
        await anyio.sleep(0.05)
    assert calls == ["done"] and rid not in server._failed  # the call in flight finished; no EPISODE_ERROR
    await server._end_episodes([rid], [{}])  # the connection's cleanup still finishes it
    assert server.log[-1] == ("finish", [rid])


@pytest.mark.anyio
async def test_keyboard_interrupt_drains_every_episode_then_leaves_grouped():
    done: list[str] = []

    def slow() -> None:
        time.sleep(0.3)
        done.append("sibling call")

    async def run(ep: EpisodeContext) -> None:
        if ep.episode.config.get("interrupt"):
            await anyio.sleep(0.05)
            raise KeyboardInterrupt
        await ep.call(slow)  # the sibling's real outstanding future

    server = Daemon(run)
    with pytest.raises(BaseExceptionGroup) as info:
        async with serving(server):
            await _start(server)
            await _start(server, config={"interrupt": True})
            await anyio.sleep(2)
    leaves = [e for g in info.value.exceptions for e in getattr(g, "exceptions", [g])]
    assert any(isinstance(e, KeyboardInterrupt) for e in _flat(info.value)), leaves
    assert done == ["sibling call"]


def _flat(exc: BaseException) -> list[BaseException]:
    return [x for e in exc.exceptions for x in _flat(e)] if isinstance(exc, BaseExceptionGroup) else [exc]


@pytest.mark.anyio
async def test_mixed_leaves_record_the_error_and_raise_control_after_the_drain():
    async def run(ep: EpisodeContext) -> None:
        cancelled = anyio.get_cancelled_exc_class()
        nested = ExceptionGroup("inner", [ValueError("ordinary")])
        raise BaseExceptionGroup("mixed", [nested, cancelled(), SystemExit(3)])

    server = Daemon(run)
    with pytest.raises(BaseExceptionGroup) as info:
        async with serving(server):
            (rid,) = await _start(server)
            await anyio.sleep(1)
    assert isinstance(server._failed[rid], ValueError)
    assert [type(e) for e in _flat(info.value)] == [SystemExit]


@pytest.mark.anyio
async def test_closed_episode_closed_is_teardown_but_an_ordinary_error_beside_it_is_kept():
    async def run(ep: EpisodeContext) -> None:
        try:
            await ep.call(time.sleep, 0.2)
        except EpisodeClosed as exc:
            raise ExceptionGroup("teardown", [exc, KeyError("cleanup bug")]) from None

    server = Daemon(run)
    async with serving(server):
        (rid,) = await _start(server)
        await anyio.sleep(0.05)
        await server._end_episodes([rid], [{}])
    ep_failed = server.daemons._eps.get(rid)
    assert ep_failed is None and server._running() == 0  # ended cleanly; the KeyError was recorded, not raised


@pytest.mark.anyio
async def test_call_refuses_after_closing_and_drops_a_late_result():
    submitted: list[str] = []
    results: list[Any] = []

    async def run(ep: EpisodeContext) -> None:
        with anyio.CancelScope() as scope:
            scope.cancel()
            await ep.call(submitted.append, "cancelled before submission")
        results.append(await ep.call(lambda: "first"))
        results.append(await ep.call(lambda: "second"))  # owned 0 -> 1 again
        ep.fail(ValueError("stop"))
        with anyio.CancelScope(shield=True):  # past the cancellation closing delivers
            with pytest.raises(EpisodeClosed):
                await ep.call(submitted.append, "after closing")
            with pytest.raises(EpisodeClosed):
                ep.start_soon(anyio.sleep, 0)
            assert ep._tasks == 1
            results.append("checked")

    server = Daemon(run)
    async with serving(server):
        await _start(server)
        await anyio.sleep(0.1)
    assert submitted == [] and results == ["first", "second", "checked"]


@pytest.mark.anyio
async def test_a_root_registration_failure_fails_the_episode_and_restores_the_count(monkeypatch):
    server = Daemon(echo)
    original = EpisodeContext.start_soon

    def failing(self: EpisodeContext, fn: Any, *args: Any) -> None:
        if fn == server.run:
            self._tasks += 1  # reserved, then rolled back by start_soon's own handler
            self._tasks -= 1
            raise OSError("cannot spawn")
        original(self, fn, *args)

    monkeypatch.setattr(EpisodeContext, "start_soon", failing)
    async with serving(server):
        (rid,) = await _start(server)
        await anyio.sleep(0.05)
        assert isinstance(server._failed[rid], OSError)
        (ep,) = server.daemons._eps.values()
        assert ep._tasks == 0 and ep._exited.is_set()


# ---------------------------------------------------------------------------
# Startup: validate before reset, roll back after it (Astra round 9 (i), (ii))
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("server_kw", "config", "error"),
    [
        ({"control_hz": None}, {"control_hz": "fast"}, "control_hz"),
        ({"control_hz": None}, {"control_hz": True}, "control_hz"),
        ({"control_hz": None}, {"control_hz": float("inf")}, "control_hz"),
        ({"control_hz": None}, {"control_hz": -2}, "control_hz"),
        ({"clock": lambda e, m: (_ for _ in ()).throw(OSError("no clock"))}, {}, "no clock"),
    ],
)
@pytest.mark.anyio
async def test_a_start_that_fails_validation_never_resets(server_kw, config, error):
    server = Daemon(echo, **server_kw)
    async with serving(server):
        (out,) = await server._start_episodes([(config, "s")])
    assert isinstance(out, Exception) and error in str(out)
    assert server.log == [] and server._running() == 0 and not server.daemons._eps


@pytest.mark.anyio
async def test_a_missing_capability_never_resets():
    class Needs(Daemon):
        requires = frozenset({"publish"})

    server = Needs(echo)
    async with serving(server):
        (refused,) = await server._start_episodes([({}, "s")], mode="live")
        (sync,) = await server._start_episodes([({}, "s")], capabilities=frozenset({"publish"}))
    assert "missing capabilities: publish" in str(refused) and "live episode" in str(sync)
    assert server.log == [] and server._running() == 0


class Warm(Daemon):
    def __init__(self, fail: int = 0, **kwargs: Any) -> None:
        super().__init__(echo, **kwargs)
        self.fail = fail  # warmups to fail

    def warmup(self, rows: Rows) -> None:
        self.threads.add(threading.current_thread().name)
        self.log.append(("warmup", rows.ids.tolist()))
        if self.fail:
            self.fail -= 1
            raise RuntimeError("cold")


@pytest.mark.anyio
async def test_warmup_runs_once_before_the_first_reset_after_validation_and_a_failure_retries():
    server = Warm(fail=1)
    async with serving(server):
        (bad,) = await server._start_episodes([({"control_hz": 0}, "s")])  # refused: no warmup
        (cold,) = await server._start_episodes([({}, "s")])
        (a,) = await _start(server)
        (b,) = await _start(server)
    assert isinstance(bad, ValueError) and isinstance(cold, RuntimeError) and "cold" in str(cold)
    assert server.log == [("warmup", [0]), ("warmup", [a]), ("reset", [a]), ("reset", [b])]
    assert server.threads == {"model_0"}


@pytest.mark.anyio
async def test_reset_runs_for_accepted_positions_only_and_failures_keep_their_positions():
    server = Daemon(echo)
    server.reset_fail = {1}
    async with serving(server):
        out = await server._start_episodes([({}, "s"), ({"control_hz": 0}, "s"), ({}, "s"), ({}, "s")])
    assert isinstance(out[1], ValueError) and "control_hz" in str(out[1])
    assert isinstance(out[2], ValueError) and "cannot reset" in str(out[2])  # reset's position 1 is batch 2
    assert server.log == [("reset", [0, 2, 3])] and isinstance(out[0], int) and isinstance(out[3], int)


@pytest.mark.anyio
async def test_a_start_aborted_after_reset_finishes_the_reset_rows_once(monkeypatch):
    server = Daemon(echo)
    async with serving(server) as tg:
        calls = 0
        original = tg.start_soon

        def flaky(fn: Any, *args: Any, **kw: Any) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:  # the second position's host registration
                raise OSError("cannot schedule")
            original(fn, *args, **kw)

        monkeypatch.setattr(tg, "start_soon", flaky)
        out = await server._start_episodes([({}, "s"), ({}, "s"), ({}, "s")])
        monkeypatch.setattr(tg, "start_soon", original)
    assert all(isinstance(o, OSError) for o in out)  # every position, the scheduled first one included
    assert server.log == [("reset", [0, 1, 2]), ("finish", [0, 1, 2])]
    assert server._running() == 0 and not server.daemons._eps and not server._publishers


@pytest.mark.anyio
async def test_a_reset_resuming_after_the_shutdown_sweep_starts_nothing():
    gate = threading.Event()

    class Slow(Daemon):
        def reset_episodes(self, rows):
            gate.wait(2)
            return super().reset_episodes(rows)

    server = Slow(echo)
    async with serving(server):
        async with anyio.create_task_group() as tg:
            out: list[Any] = []

            async def start() -> None:
                out.extend(await server._start_episodes([({}, "s")]))

            tg.start_soon(start)
            await anyio.sleep(0.05)
            server.daemons._stopping = True  # the sweep ran while reset was on the model thread
            gate.set()
    assert isinstance(out[0], RuntimeError) and "stopped" in str(out[0])
    assert server.log == [("reset", [0]), ("finish", [0])] and server._running() == 0


# ---------------------------------------------------------------------------
# Composition without DaemonModelServer, vector slots and transport-stamped mode
# ---------------------------------------------------------------------------


class Composed(ModelServer):
    """EpisodeDaemons inside a direct ModelServer, with plain functions and no reset hook; vector capable."""

    def __init__(self, run: Any, **kw: Any) -> None:
        super().__init__()
        self.daemons = EpisodeDaemons(self, run, **kw)

    async def on_serve_start(self, tg):
        await super().on_serve_start(tg)
        await self.daemons.on_serve_start(tg)

    async def on_episode_start(self, rows):
        return await self.daemons.on_episode_start(rows)

    async def on_observation(self, obs, rows):
        await self.daemons.on_observation(obs, rows)

    async def on_episode_end(self, rows, results):
        await self.daemons.on_episode_end(rows, results)


@pytest.mark.anyio
async def test_a_vector_slot_saying_live_is_still_sync_on_a_sim_clock(free_port):
    seen: list[tuple[str, type, float | None]] = []

    async def run(ep: EpisodeContext) -> None:
        seen.append((ep.mode, type(ep.clock), ep.sim_dt))
        await echo(ep)

    server = Composed(run)
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            starts = [[0, {"mode": "live", "control_hz": 20}], [1, {"control_hz": 5.0}]]
            reply = await conn.exchange(
                {"ends": [], "starts": starts, "slots": [0, 1], "observations": {"value": np.asarray([1.0, 2.0])}}
            )
        assert reply["errors"] == [] and np.asarray(reply["actions"]["actions"])[:, 0].tolist() == [1.0, 2.0]
        assert seen == [("sync", SimClock, 0.05), ("sync", SimClock, 0.2)]  # per-slot rates
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_each_episode_gets_its_own_clock_by_transport_mode():
    clocks: dict[int, Any] = {}

    async def run(ep: EpisodeContext) -> None:
        clocks[ep.row_id] = ep.clock
        if isinstance(ep.clock, SimClock) and ep.row_id == 0:
            await ep.clock.advance(1.5)
        await echo(ep)

    server = Composed(run, control_hz=4.0)
    async with serving(server):
        await _start(server, 2)
        await _start(server, mode="live")
        await anyio.sleep(0.05)
    assert [type(clocks[i]) for i in range(3)] == [SimClock, SimClock, WallClock]
    assert clocks[0].now() == 1.5 and clocks[1].now() == 0.0  # independent, fresh at 0


@pytest.mark.anyio
async def test_a_clock_factory_replaces_auto():
    made: list[tuple[str, str]] = []

    def factory(episode, mode):
        made.append((episode.session_id, mode))
        return SimClock(10.0)

    starts: list[float] = []

    async def run(ep: EpisodeContext) -> None:
        starts.append(ep.clock.now())
        await echo(ep)

    server = Composed(run, clock=factory)
    async with serving(server):
        await _start(server, mode="live")
        await anyio.sleep(0.05)
    assert made == [("s", "live")] and starts == [10.0]


# ---------------------------------------------------------------------------
# Sim time: the loop advances; rates from EPISODE_START or the server
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_advancing_by_sim_dt_fires_a_daemon_on_its_boundaries_and_stamps_admissions():
    fires: list[float] = []
    admitted: list[float] = []

    async def ticker(ep: EpisodeContext) -> None:
        async for t in ep.clock.every(0.5, first="now"):
            fires.append(t)

    async def run(ep: EpisodeContext) -> None:
        ep.start_soon(ticker, ep)
        async for req in ep.observations():
            admitted.append(req.admitted_at)
            assert ep.sim_dt is not None and isinstance(ep.clock, SimClock)
            await ep.clock.advance(ep.sim_dt)
            await req.reply({"actions": np.zeros(2, np.float32)})

    server = Daemon(run, control_hz=4.0)  # sim_dt 0.25
    async with serving(server):
        (rid,) = await _start(server)
        for v in range(5):
            await _observe(server, rid, v)
    assert admitted == [0.0, 0.25, 0.5, 0.75, 1.0] and fires == [0.0, 0.5, 1.0]


@pytest.mark.anyio
async def test_a_never_advanced_sim_clock_warns_once_and_still_replies(caplog):
    server = Daemon(echo)
    async with serving(server):
        (rid,) = await _start(server)
        with caplog.at_level(logging.WARNING):
            for v in range(60):
                assert _value(await _observe(server, rid, v)) == v
    assert sum("never advanced" in r.message for r in caplog.records) == 1


@pytest.mark.anyio
async def test_control_hz_reaches_sim_dt_from_each_runner_or_the_server_config(free_port):
    rates: list[float | None] = []

    async def run(ep: EpisodeContext) -> None:
        rates.append(ep.sim_dt)
        await echo(ep)

    class RatedBenchmark(StubBenchmark):
        control_hz = 20.0

    server = Daemon(run, control_hz=2.0)
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await SyncEpisodeRunner().run_episode(RatedBenchmark(), {"name": "t"}, conn, max_steps=2)
            await SyncEpisodeRunner().run_episode(StubBenchmark(), {"name": "t"}, conn, max_steps=2)
            await LiveEpisodeRunner(hz=50.0).run_episode(StubBenchmark(), {"name": "t"}, conn, max_steps=2)
            await conn.start_episode({"control_hz": None})  # null: the server's rate
            await conn.act({"value": 1.0})
    finally:
        await stop_server(task)
    assert rates == [0.05, 0.5, 0.02, 0.5]


@pytest.mark.anyio
async def test_one_sim_dt_spanning_several_deadlines_needs_an_acknowledged_driver():
    """Astra round 9 (iii): a raw advance over several deadlines loses a fire while a forward runs; advancing
    deadline by deadline and waiting for each due fire's completion keeps every fire (the user's driver)."""

    async def fires_with(driver: str) -> list[float]:
        clock, fired, idle = SimClock(0.0), [], anyio.Event()

        async def s1() -> None:
            nonlocal idle
            async for t in clock.every(0.1, first="now"):
                fired.append(round(t, 3))
                await anyio.sleep(0.02)  # a slow forward, in wall time
                idle.set()

        async with anyio.create_task_group() as tg:
            tg.start_soon(s1)
            await idle.wait()
            if driver == "raw":
                await clock.advance(0.25)
            else:
                for deadline in (0.1, 0.2):  # each deadline up to the observation time, acknowledged
                    idle = anyio.Event()
                    await clock.advance_to(deadline)
                    await idle.wait()
                await clock.advance_to(0.25)
            tg.cancel_scope.cancel()
        return fired

    assert await fires_with("raw") == [0.0, 0.1]  # 0.2 lost: the 0.1 forward still ran when time moved on
    assert await fires_with("acknowledged") == [0.0, 0.1, 0.2]


# ---------------------------------------------------------------------------
# chrono's SupervisorHost inside an episode, through the episode_daemon adapter
# ---------------------------------------------------------------------------


class Hosted(Daemon):
    """Runs a chrono supervisor per episode: a daemon replying from the mailbox, optionally failing."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.events: list[str] = []

    async def body(self, ctx: Any, ep: EpisodeContext) -> None:
        async for req in ep.observations():
            if float(req.obs["value"]) < 0:
                raise ValueError("daemon exploded")
            out = await ep.call(self._slow, float(req.obs["value"]))
            await req.reply({"actions": np.full(2, out, np.float32)})

    def _slow(self, v: float) -> float:
        time.sleep(0.1 if v == 7 else 0)
        self.events.append(f"call {v}")
        return v

    async def run(self, ep: EpisodeContext) -> None:
        sup = Supervisor(clock=ep.clock)
        sup.spawn(episode_daemon(ep, self.body), ep, name="replier")
        host = SupervisorHost(sup, on_stopping=lambda: self.events.append("stopping"), default_grace=0.0)
        await host.start(ep.task_group)
        try:
            await ep.stop_event.wait()
        finally:
            with anyio.CancelScope(shield=True):
                await host.stop()
                self.events.append("host stopped")

    def finish_episodes(self, rows, results):
        self.events.append("finish")
        super().finish_episodes(rows, results)


@pytest.mark.anyio
async def test_a_hosted_daemons_error_fails_its_episode_and_the_sibling_goes_on():
    server = Hosted()
    async with serving(server):
        bad, good = await _start(server, 2)
        pending = await _observe(server, bad, -1.0, wait=False)
        await anyio.sleep(0.1)
        assert isinstance(server._failed[bad], ValueError) and "daemon exploded" in str(pending.errors[0])
        assert _value(await _observe(server, good, 3.0)) == 3.0


@pytest.mark.anyio
async def test_end_stops_the_host_after_the_daemons_call_and_before_finish():
    server = Hosted()
    async with serving(server):
        (rid,) = await _start(server)
        await _observe(server, rid, 7.0, wait=False)  # a 0.1 s call
        await anyio.sleep(0.03)
        await server._end_episodes([rid], [{}])
    assert server.events == ["stopping", "call 7.0", "host stopped", "finish"]  # the call drains before finish


# ---------------------------------------------------------------------------
# PredictionEngine: raw inference from episode loops
# ---------------------------------------------------------------------------


class Raw(Daemon):
    """Each episode's loop calls a shared engine's infer_one: cross-episode batches on the model thread."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.batches: list[tuple[list[int], str]] = []
        self.steps: list[list[int]] = []
        self.engine = PredictionEngine(self, predict_batch=self._forward, max_batch_size=8, max_wait_time=0.2)

    def _forward(self, obs, rows):
        self.batches.append((rows.ids.tolist(), threading.current_thread().name))
        self.steps.append(rows.step.tolist())
        v = np.asarray(obs["value"], np.float32)
        return {"actions": v[:, None, None] + np.arange(6, dtype=np.float32).reshape(1, 3, 2)}  # full chunks

    async def on_serve_start(self, tg):
        await super().on_serve_start(tg)
        await self.engine.on_serve_start(tg)

    async def run(self, ep: EpisodeContext) -> None:
        async for req in ep.observations():
            chunk = await self.engine.infer_one(req.obs, req.row)
            await req.reply({"actions": chunk["actions"][0]})
            ep.logger.debug("chunk %s", chunk["actions"].shape)
            self.shapes.append(np.shape(chunk["actions"]))

    shapes: list[tuple[int, ...]] = []


@pytest.mark.anyio
async def test_infer_one_batches_episodes_on_the_model_thread_and_returns_full_chunks():
    server = Raw()
    server.shapes = []
    async with serving(server):
        a, b = await _start(server, 2)
        async with anyio.create_task_group() as tg:
            tg.start_soon(_observe, server, a, 1.0)
            tg.start_soon(_observe, server, b, 2.0)
    assert server.batches == [([a, b], "model_0")] and server.shapes == [(3, 2), (3, 2)]
    assert server.steps == [[0, 0]]  # each observation's own step


@pytest.mark.anyio
async def test_infer_one_refuses_a_stale_row():
    server = Raw()
    async with serving(server):
        (rid,) = await _start(server)
        row = server._rows([rid])[0]
        await server._end_episodes([rid], [{}])
        await _start(server)  # the same id, another episode
        with pytest.raises(EpisodeClosed):
            await server.engine.infer_one({"value": np.float32(1.0)}, row)
    assert server.batches == []


@pytest.mark.anyio
async def test_infer_cancelled_while_queued_is_withdrawn_and_while_running_waits_for_the_call():
    server = Raw()
    server.engine.max_wait_time = 0.5
    async with serving(server):
        (rid,) = await _start(server)
        rows = server._rows([rid])
        obs = {"value": np.asarray([1.0], np.float32)}
        with anyio.move_on_after(0.05):  # queued, waiting for a batch
            await server.engine.infer(obs, rows)
        assert server.engine._queue == [] and server.batches == []

        def slow_forward(o, r):
            time.sleep(0.2)
            return {"actions": np.zeros((len(r), 2), np.float32)}

        server.engine.predict_batch = slow_forward
        t0 = time.monotonic()
        with anyio.move_on_after(0.05):
            await server.engine.infer(obs, rows, wait_for_batch=False)
        assert time.monotonic() - t0 >= 0.2  # joined the model call before re-raising


@pytest.mark.anyio
async def test_a_stopped_engine_refuses_and_fails_whats_queued():
    server = Raw()
    server.engine.max_wait_time = 5.0
    got: list[Any] = []
    async with serving(server) as tg:
        (rid,) = await _start(server)

        async def infer() -> None:
            try:
                await server.engine.infer({"value": np.asarray([1.0], np.float32)}, server._rows([rid]))
            except Exception as exc:
                got.append(exc)
            else:
                got.append("result")

        tg.start_soon(infer)
        await anyio.sleep(0.05)
    assert server.engine._stopping
    with pytest.raises(RuntimeError, match="stopped"):
        await server.engine.infer({"value": np.asarray([1.0], np.float32)}, server._rows([rid]))


# ---------------------------------------------------------------------------
# run_predict over a real socket
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_run_predict_as_a_plain_function_server(free_port):
    from functools import partial

    def predict(obs, row):
        return {"actions": np.full(3, float(obs["value"]) * 10, np.float64)}

    server = DaemonModelServer(run=partial(run_predict, predict=predict))
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode({"task": {}})
            action = await conn.act({"value": 2.0})
            assert action["actions"].dtype == np.float64 and action["actions"].tolist() == [20.0] * 3
            await conn.start_episode(LIVE)
            await conn.send(MessageType.OBSERVATION, {"value": 3.0})
            msg = await conn.recv(timeout=2)
            assert msg.type == MessageType.ACTION and msg.payload["actions"].tolist() == [30.0] * 3
            await conn.end_episode({})
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_an_overflowing_fifo_fails_the_episode_and_a_queued_end_still_ends():
    gate = threading.Event()

    async def run(ep: EpisodeContext) -> None:
        await ep.call(gate.wait, 2)
        await echo(ep)

    server = Daemon(run, backlog=2)
    async with serving(server):
        (rid,) = await _start(server, mode="live")
        for v in range(3):
            await _observe(server, rid, v, wait=False)
        assert isinstance(server._failed[rid], MailboxOverflow)
        gate.set()
        await server._end_episodes([rid], [{}])
    assert server.log[-1] == ("finish", [rid]) and server._running() == 0


@pytest.mark.anyio
async def test_the_hooks_run_on_the_model_thread():
    server = Daemon(echo)
    async with serving(server):
        (rid,) = await _start(server)
        await _observe(server, rid, 1.0)
        await server._end_episodes([rid], [{}])
    assert server.threads == {"model_0"}


@pytest.mark.anyio
async def test_infer_drops_a_row_that_ended_during_the_model_call():
    server = Raw()

    def slow_forward(o, r):
        time.sleep(0.2)
        return {"actions": np.zeros((len(r), 2), np.float32)}

    server.engine.predict_batch = slow_forward
    async with serving(server):
        a, b = await _start(server, 2)
        rows = server._rows([a, b])
        got: list[Any] = []

        async def infer() -> None:
            got.append(await server.engine.infer({"value": np.asarray([1.0, 2.0], np.float32)}, rows))

        async with anyio.create_task_group() as tg:
            tg.start_soon(infer)
            await anyio.sleep(0.05)
            server._gen[a] += 1  # a ends while the call runs
    (result,) = got
    assert list(result.errors) == [0] and isinstance(result.errors[0], EpisodeClosed)
    assert np.asarray(result.actions["actions"]).shape == (1, 2)  # b's alone
