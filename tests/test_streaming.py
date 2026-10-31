"""Server-initiated messages: terminal failure (fail_episode / EPISODE_ERROR), PUBLISH and its listener rules,
capability negotiation and old-client refusal; RTCServer's time-grid streaming."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, cast

import anyio
import numpy as np
import pytest
import websockets
from websockets.asyncio.server import serve as ws_serve

from vla_eval import connection as connection_module
from vla_eval.connection import Connection, ServerEpisodeError
from vla_eval.model_servers.base import Episode, Rows, _Link, _Pending
from vla_eval.model_servers.daemons import EpisodeContext, ObservationRequest
from vla_eval.model_servers.live import RTCServer
from vla_eval.protocol.messages import Message, MessageType, pack_message, unpack_message
from vla_eval.runners.action_buffer import ActionBuffer
from vla_eval.runners.base import failure_reason
from vla_eval.runners.live_runner import LiveEpisodeRunner
from vla_eval.runners.sync_runner import SyncEpisodeRunner

from tests.conftest import BatchLogServer, LiveLogServer, LogServer, StubBenchmark, start_server, stop_server
from tests.test_daemons import Daemon, _observe, _start, echo, serving

LIVE = {"task": {"name": "t"}, "mode": "live"}


@pytest.fixture
def legacy(monkeypatch):
    """Clients that advertise no capabilities, as before this protocol addition."""
    monkeypatch.setattr(connection_module, "CAPABILITIES", {})


async def _until(cond: Any, timeout: float = 3.0) -> None:
    with anyio.fail_after(timeout):
        while not cond():
            await anyio.sleep(0.01)


# ---------------------------------------------------------------------------
# Negotiation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_capabilities_round_trip_and_only_exactly_true_counts(free_port):
    seen: list[frozenset[str]] = []

    async def run(ep: EpisodeContext) -> None:
        seen.append(ep.episode.capabilities)
        await echo(ep)

    server = Daemon(run)
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            assert {k: conn.server_info["capabilities"][k] for k in ("publish", "episode_error")} == {
                "publish": True,
                "episode_error": True,
            }
            await conn.start_episode(LIVE)
            await conn.end_episode({})
        async with websockets.connect(f"ws://127.0.0.1:{free_port}") as ws:
            hello = {"harness_version": "x", "capabilities": {"publish": "yes", "episode_error": 1}}
            await ws.send(pack_message(Message(MessageType.HELLO, hello, 1)))
            await ws.recv()
            await ws.send(pack_message(Message(MessageType.EPISODE_START, LIVE, 2)))
            await _until(lambda: len(seen) == 2)
    finally:
        await stop_server(task)
    assert seen == [frozenset({"publish", "episode_error"}), frozenset()]


@pytest.mark.anyio
async def test_a_server_requiring_a_capability_refuses_an_old_client_and_closes_its_socket(free_port, legacy):
    class Needs(Daemon):
        requires = frozenset({"episode_error"})

    task = await start_server(Needs(echo), free_port)
    try:
        conn = Connection(f"ws://127.0.0.1:{free_port}")
        await conn.connect()
        await conn.start_episode(LIVE)
        msg = await conn.recv(timeout=2)
        assert msg.type == MessageType.ERROR and "missing capabilities: episode_error" in msg.payload["error"]
        with pytest.raises(websockets.exceptions.ConnectionClosed) as info:
            await conn.recv(timeout=2)
        assert info.value.rcvd is not None and info.value.rcvd.code == 1008
        await conn.close()
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_an_old_live_client_against_rtc_ends_connection_closed(free_port, legacy):
    task = await start_server(Chunker(hz=50.0), free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            with pytest.raises(websockets.exceptions.ConnectionClosed) as info:
                await LiveEpisodeRunner(hz=50.0).run_episode(StubBenchmark(), {"name": "t"}, conn, max_steps=200)
        assert failure_reason(info.value) == "connection_closed"
    finally:
        await stop_server(task)


# ---------------------------------------------------------------------------
# fail_episode on the base
# ---------------------------------------------------------------------------


class FailingAtStart(BatchLogServer):
    """Fails the episodes whose task says so as soon as they start, before any observation."""

    async def on_episode_start(self, rows: Rows) -> dict[int, Exception] | None:
        failed = await super().on_episode_start(rows)
        doomed = [i for i, e in enumerate(rows.episodes) if e.config.get("task", {}).get("doomed")]
        if doomed:
            self.fail_episode(rows.take(doomed), ValueError("model gave up"))
        return failed


@pytest.mark.anyio
async def test_a_sync_episode_failed_before_its_first_observation_errors_it(free_port):
    server = FailingAtStart()
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            with pytest.raises(Exception) as info:
                await SyncEpisodeRunner().run_episode(StubBenchmark(), {"name": "t", "doomed": True}, conn)
            assert failure_reason(info.value) == "model_act" and "model gave up" in str(info.value)
            await _until(lambda: server.results)
        assert server.predicted == [] and server.results == [{"status": "error", "failure_reason": "model_act"}]
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_a_vector_slot_failed_at_start_fails_alone(free_port):
    server = FailingAtStart()
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            starts = [[0, {"task": {"doomed": True}}], [1, {"task": {}}]]
            reply = await conn.exchange(
                {"ends": [], "starts": starts, "slots": [0, 1], "observations": {"value": np.asarray([1.0, 2.0])}}
            )
        assert [s for s, _ in reply["errors"]] == [0] and "model gave up" in reply["errors"][0][1]
        assert np.asarray(reply["actions"]["actions"])[:, 0].tolist() == [2.0] and server.batches == [[1]]
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_fail_episode_completes_pending_positions_once_and_ignores_stale_rows():
    server = LogServer()
    hooks: list[tuple[int, int]] = []
    sent: list[str] = []

    def bad_hook(rid: int, gen: int, episode: Episode, error: Exception) -> None:
        raise RuntimeError("hook exploded")

    async def sink(error: Exception) -> None:
        sent.append(str(error))
        raise RuntimeError("sink exploded")

    server._fail_hooks += [bad_hook, lambda rid, gen, ep, err: hooks.append((rid, gen))]
    async with serving(server):
        a, b = await _start(server, 2)
        server._sinks[a] = sink
        stale = server._rows([b])
        await server._end_episodes([b], [{}])
        (b2,) = await _start(server)  # the same id, another incarnation
        pending = _Pending(3)
        pending.ids = np.asarray([a, a, b2])
        pending.gens = server._gen[pending.ids].copy()
        server._open.add(pending)
        gen_a = int(server._gen[a])
        server.fail_episode(server._rows([a, a]), ValueError("first"))  # a row listed twice: once
        server.fail_episode(server._rows([a]), ValueError("second"))  # first failure wins
        server.fail_episode(stale, ValueError("stale"))  # an ended incarnation: ignored
        await anyio.sleep(0.05)
    assert hooks == [(a, gen_a)] and sent == ["first"]
    assert {i: str(e) for i, e in pending.errors.items()} == {0: "first", 1: "first"} and pending.left == 1
    assert server._failed == {a: server._failed[a]} and str(server._failed[a]) == "first"
    await server._end_episodes([a], [{}])
    assert a not in server._failed and a not in server._sinks


@pytest.mark.anyio
async def test_a_cancelled_observation_leaves_nothing_tracked():
    class Hang(LogServer):
        async def on_observation(self, obs, rows):
            await anyio.sleep_forever()

    server = Hang()
    (rid,) = await _start(server)
    pending = _Pending(1)
    with anyio.move_on_after(0.05):
        with server._track(pending):
            await server._observe({"value": np.asarray([1.0])}, [rid], pending)
    assert not server._open


# ---------------------------------------------------------------------------
# EPISODE_ERROR through a real server, listener and runner
# ---------------------------------------------------------------------------


class Breaks(Daemon):
    """A live daemon that fails its episode after ``after`` observations (an S2 context overflow, say)."""

    def __init__(self, after: int = 2) -> None:
        super().__init__()
        self.after = after
        self.results: list[dict[str, Any]] = []

    async def run(self, ep: EpisodeContext) -> None:
        n = 0
        async for req in ep.observations():
            n += 1
            if n > self.after:
                raise OverflowError("context overflow")
            await req.reply({"actions": np.zeros(1, np.float32)})

    def finish_episodes(self, rows, results):
        self.results += [{k: r.get(k) for k in ("status", "failure_reason")} for r in results]


@pytest.mark.anyio
async def test_a_failed_live_episode_ends_model_act_through_the_runner(free_port):
    server = Breaks(after=2)
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            with anyio.fail_after(5), pytest.raises(ServerEpisodeError) as info:
                await LiveEpisodeRunner(hz=50.0).run_episode(StubBenchmark(), {"name": "t"}, conn, max_steps=10_000)
            assert failure_reason(info.value) == "model_act" and "context overflow" in str(info.value)
            await _until(lambda: server.results)
            # the connection goes on: a sync episode next
            server.after = 100
            await SyncEpisodeRunner().run_episode(StubBenchmark(), {"name": "t"}, conn, max_steps=2)
        assert server.results[0] == {"status": "error", "failure_reason": "model_act"}
    finally:
        await stop_server(task)


def _raw_server(on_message: Any) -> Any:
    async def handler(ws: Any) -> None:
        async for raw in ws:
            msg = unpack_message(raw)
            if msg.type == MessageType.HELLO:
                hello = Message(MessageType.HELLO, {"harness_version": "x"}, msg.seq)
                await ws.send(pack_message(hello))
            else:
                await on_message(ws, msg)

    return handler


@pytest.mark.anyio
async def test_the_listener_takes_publish_and_episode_error_for_the_running_episode_only(free_port):
    """A raw server: on each EPISODE_START it sends a wrong-seq PUBLISH with a high index first, then its script."""
    starts: list[int] = []

    async def on_message(ws: Any, msg: Message) -> None:
        if msg.type == MessageType.EPISODE_START:
            starts.append(msg.seq)
            s = msg.seq
            send = [
                (MessageType.PUBLISH, {"action": {"a": -1}, "index": 99}, s + 1000),  # no such episode
                (MessageType.EPISODE_ERROR, {"error": "x"}, s - 1),  # an earlier episode's
                (MessageType.PUBLISH, {"action": {"a": 0}, "index": 0}, s),
                (MessageType.PUBLISH, {"action": {"a": -1}, "index": 0}, s),  # stale index
                (MessageType.ACTION, {"a": 1}, s + 1),
                (MessageType.PUBLISH, {"action": {"a": 2}, "index": 3}, s),
            ]
            for t, p, q in send:
                await ws.send(pack_message(Message(t, p, q)))
        elif msg.type == MessageType.OBSERVATION and msg.payload.get("fail"):
            await ws.send(pack_message(Message(MessageType.EPISODE_ERROR, {"error": "boom"}, starts[-1])))

    async with ws_serve(_raw_server(on_message), "127.0.0.1", free_port):
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            got: list[Any] = []
            conn.on_action(lambda a: got.append(("action", a["a"])))
            conn.on_publish(lambda a: got.append(("publish", a["a"])))
            async with conn.listening():
                await conn.start_episode({})
                await _until(lambda: len(got) == 3)
                await conn.start_episode({})  # index 0 counts again
                await _until(lambda: len(got) == 6)
            assert got == [("publish", 0), ("action", 1), ("publish", 2)] * 2

            with pytest.raises(ServerEpisodeError, match="boom"):
                async with conn.listening():
                    await conn.send_observation({"fail": True})
                    await anyio.sleep(2)
            await conn.end_episode({})  # raw END: the episode's EPISODE_ERROR no longer counts
            async with conn.listening():
                await conn.send_observation({"fail": True})
                await anyio.sleep(0.2)


@pytest.mark.anyio
async def test_a_failed_start_send_and_close_clear_the_running_episode(free_port):
    async def on_message(ws: Any, msg: Message) -> None:
        pass

    async with ws_serve(_raw_server(on_message), "127.0.0.1", free_port):
        conn = Connection(f"ws://127.0.0.1:{free_port}")
        await conn.connect()
        await conn.start_episode({})
        assert conn._episode_active

        class Broken:
            state = conn._ws.state

            async def send(self, data: bytes) -> None:
                raise OSError("send failed")

        real, conn._ws = conn._ws, Broken()
        setattr(conn, "_ensure_connected", _noop)
        with pytest.raises(OSError):
            await conn.start_episode({})
        assert not conn._episode_active
        conn._ws = real
        await conn.start_episode({})
        await conn.close()
        assert not conn._episode_active


async def _noop() -> None:
    return None


@pytest.mark.anyio
async def test_latest_wins_between_replies_and_publishes_in_the_action_buffer(free_port):
    async def on_message(ws: Any, msg: Message) -> None:
        if msg.type == MessageType.OBSERVATION:
            await ws.send(pack_message(Message(MessageType.PUBLISH, {"action": {"v": "p"}, "index": 0}, msg.seq - 1)))
            await ws.send(pack_message(Message(MessageType.ACTION, {"v": "a"}, msg.seq)))
            await ws.send(pack_message(Message(MessageType.PUBLISH, {"action": {"v": "p2"}, "index": 1}, msg.seq - 1)))

    async with ws_serve(_raw_server(on_message), "127.0.0.1", free_port):
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            buffer = ActionBuffer(hold_fn=lambda last: {"v": "hold"})
            conn.on_action(buffer.update)
            conn.on_publish(buffer.update)
            async with conn.listening():
                await conn.start_episode({})
                await conn.send_observation({})
                await _until(lambda: buffer.update_count == 3)
            assert buffer.get() == {"v": "p2"} and buffer.get() == {"v": "hold"}


# ---------------------------------------------------------------------------
# ep.publish
# ---------------------------------------------------------------------------


class Publisher(Daemon):
    """Publishes ``n`` actions per observation, the last from a separate task too (concurrent publishers)."""

    def __init__(self, n: int = 3) -> None:
        super().__init__()
        self.n = n
        self.errors: list[str] = []

    async def run(self, ep: EpisodeContext) -> None:
        async for req in ep.observations():
            for k in range(self.n):
                ep.start_soon(ep.publish, {"actions": np.full(1, k, np.float32)})
            await ep.publish({"actions": np.full(1, 100.0, np.float32)}, source=req)


@pytest.mark.anyio
async def test_publish_sends_indexed_actions_bound_to_the_episode(free_port):
    server = Publisher(n=4)
    task = await start_server(server, free_port)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{free_port}") as ws:
            caps = {"capabilities": {"publish": True, "episode_error": True}}
            await ws.send(pack_message(Message(MessageType.HELLO, caps, 1)))
            await ws.recv()
            await ws.send(pack_message(Message(MessageType.EPISODE_START, LIVE, 2)))
            await ws.send(pack_message(Message(MessageType.OBSERVATION, {"value": 1.0}, 3)))
            msgs = [unpack_message(cast(bytes, await ws.recv())) for _ in range(5)]
        assert {m.type for m in msgs} == {MessageType.PUBLISH} and {m.seq for m in msgs} == {2}
        assert [m.payload["index"] for m in msgs] == [0, 1, 2, 3, 4]  # unique, increasing in send order
        sources = {float(m.payload["action"]["actions"][0]): m.payload["source_seq"] for m in msgs}
        assert sources == {0.0: None, 1.0: None, 2.0: None, 3.0: None, 100.0: 3}
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_publish_needs_a_live_capable_episode_and_its_own_source():
    errors: list[tuple[str, str]] = []
    other: list[ObservationRequest] = []

    async def run(ep: EpisodeContext) -> None:
        async for req in ep.observations():
            if ep.can_publish:
                try:
                    await ep.publish({"a": 1}, source=other[0])
                except ValueError as exc:
                    errors.append(("foreign", str(exc)))
            try:
                await ep.publish({"a": 1}, source=req)
            except RuntimeError as exc:
                errors.append((ep.mode, str(exc)))
            other.append(req)
            await req.reply({"actions": np.zeros(1, np.float32)})

    sent: list[Any] = []

    async def publish(action: Any, index: int, source_seq: int | None) -> None:
        sent.append(index)

    server = Daemon(run)
    async with serving(server):
        (sync,) = await _start(server, capabilities=frozenset({"publish"}))
        (live_old,) = await _start(server, mode="live")  # no publish capability
        (live,) = await _start(server, mode="live", capabilities=frozenset({"publish"}), link=_Link(None, publish))
        for rid in (sync, live_old, live):
            await _observe(server, rid, 1.0, wait=False)
            await anyio.sleep(0.05)
    assert [e[0] for e in errors] == ["sync", "live", "foreign"]
    assert sent == [0]


@pytest.mark.anyio
async def test_no_publish_starts_after_closing_even_while_one_is_blocked_in_send():
    entered, release = anyio.Event(), anyio.Event()
    sent: list[int] = []

    async def publish(action: Any, index: int, source_seq: int | None) -> None:
        if index == 0:
            entered.set()
            await release.wait()
        sent.append(index)

    async def admitted() -> None:
        with anyio.CancelScope(shield=True):  # this send completes despite closing
            await ep_ref[0].publish({"a": 0})

    ep_ref: list[EpisodeContext] = []

    async def run(ep: EpisodeContext) -> None:
        ep_ref.append(ep)
        ep.start_soon(admitted)  # blocks in send
        await entered.wait()
        ep.start_soon(ep.publish, {"a": 1})  # waits for the lock
        await anyio.sleep_forever()

    server = Daemon(run)
    async with serving(server):
        (rid,) = await _start(server, mode="live", capabilities=frozenset({"publish"}), link=_Link(None, publish))
        await entered.wait()
        await anyio.sleep(0.02)
        async with anyio.create_task_group() as tg:
            tg.start_soon(server._end_episodes, [rid], [{}])
            await anyio.sleep(0.02)
            release.set()
    assert sent == [0]  # the admitted send completed; the queued one never started


# ---------------------------------------------------------------------------
# RTCServer: the time grid, against chrono's ticker semantics on a scripted clock
# ---------------------------------------------------------------------------


class Chunker(RTCServer):
    """Chunk ``k``: ``value * 100 + k`` per row."""

    def __init__(self, chunk: int = 8, **kw: Any) -> None:
        super().__init__(chunk_size=chunk, **kw)
        self.shape: tuple[int, ...] | None = None

    def predict(self, obs, row):
        if self.shape is not None:
            return np.zeros(self.shape, np.float32)
        return float(obs["value"]) * 100 + np.arange(self.chunk_size, dtype=np.float32)[:, None] * np.ones((1, 2))


class ScriptClock:
    """Wakes ``every`` at scripted times (missed ticks skipped, as chrono's WallClock), setting ``now``."""

    def __init__(self, wakes: list[float]) -> None:
        self.wakes, self.t = wakes, wakes[0]

    def now(self) -> float:
        return self.t

    async def every(self, period: float, *, first: str = "after"):
        for t in self.wakes:
            self.t = t
            yield t
            for _ in range(5):  # let a refill task run
                await anyio.sleep(0)


class FakeEp:
    """The parts of EpisodeContext RTCServer uses; requests arrive as scripted ``(time, value)``."""

    def __init__(self, server: RTCServer, clock: ScriptClock, arrivals: list[tuple[float, float, float]]) -> None:
        self.server, self.clock, self.arrivals = server, clock, list(arrivals)  # (arrival, admitted_at, value)
        self.closing = False
        self.published: list[tuple[float, float]] = []
        self.errors: list[str] = []
        self.block: anyio.Event | None = None  # blocks the publish at t=2 until the second inference
        self.calls = 0
        self.tg: Any = None

    def receive_nowait(self) -> Any:
        due = [a for a in self.arrivals if a[0] <= self.clock.now()]
        if not due:
            return None
        newest = due[-1]
        self.arrivals = [a for a in self.arrivals if a[0] > self.clock.now()]

        async def error(exc: Exception) -> None:
            self.errors.append(str(exc))

        return SimpleNamespace(obs={"value": newest[2]}, row=None, admitted_at=newest[1], error=error)

    def start_soon(self, fn: Any, *args: Any) -> None:
        self.tg.start_soon(fn, *args)

    async def call(self, fn: Any, *args: Any) -> Any:
        await anyio.sleep(0)
        self.calls += 1
        if self.calls == 2 and self.block is not None:
            self.block.set()  # the blocked publish returns after this refill replaced the plan
        return fn(*args)

    async def publish(self, action: Any, **kw: Any) -> None:
        if self.block is not None and self.clock.now() == 2.0:
            await self.block.wait()
            await anyio.sleep(0)
        self.published.append((self.clock.now(), float(action["actions"][0])))


async def _rtc(server: RTCServer, wakes: list[float], arrivals: list[tuple[float, float, float]]) -> FakeEp:
    ep = FakeEp(server, ScriptClock(wakes), arrivals)
    async with anyio.create_task_group() as tg:
        ep.tg = tg
        await server.run(cast(Any, ep))
    return ep


@pytest.mark.anyio
async def test_rtc_publishes_the_current_slot_once_and_never_a_future_one():
    """hz=1, admitted at 0.3: ticks at 2.8 and 3.0 give slot 2 once; slot 3 waits for 3.3."""
    ep = await _rtc(Chunker(hz=1.0), [0.3, 2.8, 3.0, 3.3, 4.0], [(0.3, 0.3, 1.0)])
    assert ep.published == [(2.8, 102.0), (3.3, 103.0)]


@pytest.mark.anyio
async def test_rtc_skips_expired_slots_and_discards_a_fully_expired_chunk():
    ep = await _rtc(Chunker(chunk=2, hz=1.0), [5.0, 5.5, 6.0], [(5.0, 0.0, 1.0)])  # admitted 5 s before
    assert ep.published == []
    ep = await _rtc(Chunker(hz=10.0), [0.0, 0.1, 0.35, 0.4], [(0.0, 0.0, 1.0)])
    assert ep.published == [(0.1, 101.0), (0.35, 103.0), (0.4, 104.0)]  # slot 2 expired unseen


@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.anyio
async def test_rtc_refills_ahead_and_a_replacement_during_a_blocked_publish_keeps_its_cursor(blocked):
    """Plan 1 (admitted 0.0) streams slots 1, 2; at 2.0 two slots are left, so the observation admitted at 1.0 is
    inferred; its plan (anchored at 1.0, slot 0 expired) takes over at slot 2 even if it lands mid-publish."""
    server = Chunker(chunk=4, hz=1.0, refill_at=2)
    ep = FakeEp(server, ScriptClock([0.0, 1.0, 2.0, 3.0, 4.0]), [(0.0, 0.0, 1.0), (1.0, 1.0, 2.0)])
    ep.block = anyio.Event() if blocked else None
    async with anyio.create_task_group() as tg:
        ep.tg = tg
        await server.run(cast(Any, ep))
    assert ep.published == [(1.0, 101.0), (2.0, 102.0), (3.0, 202.0), (4.0, 203.0)] and ep.calls == 2


@pytest.mark.parametrize("shape", [(0, 2), (2,)])
@pytest.mark.anyio
async def test_rtc_an_empty_or_flat_chunk_fails_its_observation_only(shape):
    server = Chunker(hz=1.0)
    server.shape = shape
    ep = await _rtc(server, [0.0, 1.0, 2.0], [(0.0, 0.0, 1.0), (1.0, 1.0, 2.0)])
    assert ep.published == [] and len(ep.errors) == 2 and "[T, D]" in ep.errors[0]


def test_rtc_validates_its_parameters():
    bad: list[dict[str, Any]] = [{"hz": 0.0}, {"hz": math.inf}, {"chunk_size": 0}, {"refill_at": -1}]
    for kw in bad:
        with pytest.raises(ValueError):
            RTCServer(**kw)


@pytest.mark.anyio
async def test_rtc_streams_to_a_live_runner(free_port):
    server = Chunker(hz=20.0)
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            result = await LiveEpisodeRunner(hz=20.0).run_episode(
                StubBenchmark(done_at_step=30), {"name": "t"}, conn, max_steps=30
            )
        assert result["rt_metrics"]["update_count"] >= 10 and result["rt_metrics"]["fresh_count"] >= 10
    finally:
        await stop_server(task)


# ---------------------------------------------------------------------------
# LiveModelServer LAAS under a sync client, with model-queue delay
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_live_laas_under_a_sync_client_counts_wall_time_from_admission(free_port):
    server = LiveLogServer(delay=0.12, out="ramp", chunk_size=4, laas=True, hz=10.0)
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode({"task": {}})
            action = await conn.act({"value": 1.0})
        assert np.array_equal(action["actions"], 1.0 + np.arange(7, 14))  # step 1 of the chunk
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_a_live_episode_left_without_tasks_is_failed_before_any_observation(free_port):
    async def run(ep: EpisodeContext) -> None:
        return

    task = await start_server(Daemon(run), free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            with anyio.fail_after(3), pytest.raises(ServerEpisodeError, match="no task is left"):
                async with conn.listening():
                    await conn.start_episode(LIVE)
                    await anyio.sleep_forever()
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_serving_twice_registers_the_failure_hook_once():
    server = Daemon(echo)
    for _ in range(2):
        async with serving(server):
            pass
    assert server._fail_hooks == [server.daemons._on_fail]


@pytest.mark.anyio
async def test_an_observation_after_the_dispatcher_stopped_fails_instead_of_waiting():
    server = LogServer()
    async with serving(server):
        (rid,) = await _start(server)
    pending = await _observe(server, rid, 1.0)
    assert "stopped" in str(pending.errors[0]) and server.predicted == []
