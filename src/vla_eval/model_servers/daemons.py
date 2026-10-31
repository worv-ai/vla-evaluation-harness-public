"""Episode daemons: a model server that owns its real-time loop per episode.

``run(ep)`` runs once per episode as the root task of its scope (``EpisodeContext``).  The loop decides when to
infer, what to reply (``request.reply``) or, live only, stream (``ep.publish``), and how its clock advances: a sync
episode gets a fresh ``SimClock`` the loop advances, a live one ``WallClock``.  The framework supplies guard rails
rather than a scheduler: one mailbox, one model thread (``ep.call``), owned work drained before ``finish``, and
terminal failure (``ep.fail``) for any ordinary exception escaping an episode task.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import anyio
from anyio.lowlevel import checkpoint
from chrono_daemon import Clock, SimClock, WallClock

from vla_eval import columns
from vla_eval.model_servers.base import (
    CAPABILITIES,
    Episode,
    EpisodeClosed,
    MissingCapabilities,
    ModelServer,
    Row,
    Rows,
    _Publisher,
)
from vla_eval.types import Action, Batch, EpisodeResult, Observation

if TYPE_CHECKING:
    from anyio.abc import TaskGroup

__all__ = [
    "Clock",
    "DaemonModelServer",
    "EpisodeClosed",
    "EpisodeContext",
    "EpisodeDaemons",
    "Mailbox",
    "MailboxOverflow",
    "ObservationRequest",
    "SimClock",
    "WallClock",
    "episode_daemon",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")
Mode = Literal["sync", "live"]
Backlog = Literal["latest"] | int
Run = Callable[["EpisodeContext"], Awaitable[None]]
Reset = Callable[[Rows], "dict[int, Exception] | None"]
Finish = Callable[[Rows, "list[EpisodeResult]"], None]
ClockFactory = Callable[[Episode, Mode], Clock]

_UNADVANCED_WARN_AT = 50  # observations admitted on a SimClock still at t0


class MailboxOverflow(RuntimeError):
    """A FIFO mailbox's backlog was exceeded; fails the episode."""


@dataclass(frozen=True, eq=False)
class ObservationRequest:
    """One admitted observation; ``reply`` / ``error`` answer it (once, while its episode runs)."""

    obs: Observation  # this row's observation, as predict() takes it
    batch: Batch  # the same as a one-row column batch
    rows: Rows  # exactly one row, bound to this observation
    step: int
    admitted_at: float  # ep.clock.now() at admission
    admitted_wall: float  # time.monotonic() at admission
    seq: int | None  # the OBSERVATION's wire seq (None for a vector slot)
    _server: ModelServer = field(repr=False)

    @property
    def row(self) -> Row:
        return self.rows[0]

    async def reply(self, action: Action) -> None:
        await self._server.send_action(self.rows, columns.stack([action]))

    async def error(self, error: Exception) -> None:
        """Fail this observation alone; the episode goes on."""
        await self._server.send_error(self.rows, error)


class Mailbox:
    """One episode's admitted observations: ``"latest"`` keeps the newest, an int is a bounded FIFO.  One owner
    task receives; ``put`` never blocks."""

    def __init__(self, backlog: Backlog) -> None:
        if backlog != "latest" and not (isinstance(backlog, int) and not isinstance(backlog, bool) and backlog >= 1):
            raise ValueError(f'backlog must be "latest" or a positive int, got {backlog!r}')
        self.backlog = backlog
        self.dropped = 0
        self._items: deque[ObservationRequest] = deque()
        self._wake = anyio.Event()
        self._closed = False
        self._owner: int | None = None

    def put(self, request: ObservationRequest) -> None:
        if self._closed:
            return
        if self.backlog == "latest":
            if self._items:
                self._items.clear()
                self.dropped += 1
        elif len(self._items) >= self.backlog:
            raise MailboxOverflow(f"more than {self.backlog} observations waiting")
        self._items.append(request)
        self._wake.set()

    async def receive(self) -> ObservationRequest:
        self._own()
        while True:
            if self._closed:
                raise EpisodeClosed("the episode is closing")
            if self._items:
                return self._items.popleft()
            await self._wake.wait()
            self._wake = anyio.Event()

    def receive_nowait(self) -> ObservationRequest | None:
        self._own()
        return self._items.popleft() if self._items and not self._closed else None

    def close(self) -> None:
        self._closed = True
        self._items.clear()
        self._wake.set()

    def _own(self) -> None:
        task = anyio.get_current_task().id
        if self._owner is None:
            self._owner = task
        elif self._owner != task:
            raise RuntimeError("an episode's observations have one receiving task")


class _TaskGroupView:
    """``ep.task_group``: ``start_soon`` only (what ``SupervisorHost.start`` uses), as an episode task."""

    def __init__(self, ep: EpisodeContext) -> None:
        self._ep = ep

    def start_soon(self, fn: Callable[..., Awaitable[Any]], *args: Any, name: object = None) -> None:
        self._ep.start_soon(fn, *args)


class EpisodeContext:
    """One episode's scope, given to ``run(ep)`` and its tasks; invalid after ``on_episode_end``.  Event-loop only."""

    def __init__(
        self,
        daemons: EpisodeDaemons,
        rows: Rows,
        clock: Clock,
        sim_dt: float | None,
        mailbox: Mailbox,
        publisher: _Publisher | None,
    ) -> None:
        row = rows[0]
        self.episode = row.episode
        self.row_id = row.id
        self.mode: Mode = row.episode.mode
        self.clock = clock
        self.sim_dt = sim_dt
        self.logger = logging.getLogger(f"vla_eval.episode.{row.episode.episode_id[:8]}")
        self.stop_event = anyio.Event()
        self.task_group = _TaskGroupView(self)
        self.can_publish = self.mode == "live" and publisher is not None
        self._daemons = daemons
        self._server = daemons.server
        self._rows = rows  # the start Rows: this incarnation
        self._mailbox = mailbox
        self._publisher = publisher
        self._failed: Exception | None = None
        self._control: BaseException | None = None  # KeyboardInterrupt / SystemExit, raised after the drain
        self._tasks = 0
        self._owned = 0
        self._drained = anyio.Event()
        self._drained.set()
        self._exited = anyio.Event()
        self._tg: TaskGroup | None = None
        self._scope = anyio.CancelScope()
        self._scheduled = False
        self._t0 = clock.now()
        self._admitted = 0

    @property
    def closing(self) -> bool:
        return self.stop_event.is_set()

    stopping = closing  # chrono-daemon's name for it

    async def observations(self) -> AsyncIterator[ObservationRequest]:
        """The admitted observations, in order; ends when the episode closes."""
        while True:
            try:
                request = await self._mailbox.receive()
            except EpisodeClosed:
                return
            yield request

    def receive_nowait(self) -> ObservationRequest | None:
        return self._mailbox.receive_nowait()

    def start_soon(self, fn: Callable[..., Awaitable[Any]], *args: Any) -> None:
        """Run ``fn(*args)`` as a task of this episode (its exceptions fail the episode)."""
        if self.closing:
            raise EpisodeClosed("the episode is closing")
        if self._tg is None:
            raise RuntimeError("the episode's scope has not started")
        self._tasks += 1
        try:
            self._tg.start_soon(self._daemons._task, self, fn, args)
        except BaseException:
            self._tasks -= 1
            raise

    async def call(self, fn: Callable[..., T], *args: Any) -> T:
        """``fn(*args)`` on the model thread; the episode's release waits for it.  ``EpisodeClosed`` if the episode
        is closing, or closed before the call completed (its result is dropped)."""
        await checkpoint()  # a cancelled caller submits nothing
        if self.closing:
            raise EpisodeClosed("the episode is closing")
        if self._owned == 0:
            self._drained = anyio.Event()
        self._owned += 1
        try:
            result = await self._server._run_submitted(fn, *args)
        except Exception as exc:
            if self.closing:
                raise EpisodeClosed("the episode closed during the call") from exc
            raise
        finally:
            self._owned -= 1
            if self._owned == 0:
                self._drained.set()
        if self.closing:
            raise EpisodeClosed("the episode closed during the call")
        return result

    async def publish(self, action: Action, *, source: ObservationRequest | None = None) -> None:
        """Send ``action`` to a live client now (``PUBLISH``, latest-wins there).  Dropped once closing."""
        if not self.can_publish or self._publisher is None:
            raise RuntimeError("publish needs a live episode whose client advertised 'publish'")
        if source is not None and not self._owns(source):
            raise ValueError("the source observation belongs to another episode")
        pub = self._publisher
        async with pub.lock:
            if self.closing:  # an admitted send may complete; none starts after closing
                self.logger.debug("publish dropped: the episode is closing")
                return
            index, pub.index = pub.index, pub.index + 1
            await pub.send(action, index, None if source is None else source.seq)

    def fail(self, error: Exception) -> None:
        """End the episode as failed by the model (terminal; first failure wins)."""
        if self._failed is None:
            self._failed = error
            if not self.closing:
                self.logger.error("episode failed: %s: %s", type(error).__name__, error, exc_info=error)
                try:
                    self._server.fail_episode(self._rows, error)
                finally:
                    self._close()
                return
        self._close()

    def _owns(self, request: ObservationRequest) -> bool:
        return (
            int(request.rows.ids[0]) == self.row_id
            and int(request.rows._gen[0]) == int(self._rows._gen[0])
            and request.row.episode is self.episode
        )

    def _close(self) -> None:
        self.stop_event.set()
        self._mailbox.close()
        self._scope.cancel()


def _leaves(exc: BaseException) -> list[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for e in exc.exceptions for leaf in _leaves(e)]
    return [exc]


def _classify(ep: EpisodeContext, exc: BaseException) -> None:
    """The one exception policy: ordinary errors fail the episode, closing ``EpisodeClosed`` and cancellation are
    teardown, process control is captured (raised grouped after the drain)."""
    cancelled = anyio.get_cancelled_exc_class()
    for leaf in _leaves(exc):
        if isinstance(leaf, EpisodeClosed) and ep.closing:
            continue
        if isinstance(leaf, Exception):
            ep.fail(leaf)
        elif not isinstance(leaf, cancelled):
            ep._control = ep._control or leaf
            ep._close()


def episode_daemon(
    ep: EpisodeContext, body: Callable[..., Awaitable[None]]
) -> Callable[..., Coroutine[Any, Any, None]]:
    """A chrono daemon body's boundary adapter: its exceptions fail ``ep`` before chrono's own cleanup runs, and
    process control never escapes bare.  ``sup.spawn(episode_daemon(ep, body), *args)``."""

    async def run(ctx: Any, *args: Any, **kwargs: Any) -> None:
        try:
            await body(ctx, *args, **kwargs)
        except BaseException as exc:
            _classify(ep, exc)
            if isinstance(exc, anyio.get_cancelled_exc_class()):
                raise

    return run


class EpisodeDaemons:
    """One scope per started row, running ``run(ep)``; a ``ModelServer`` forwards its four lifecycle callbacks.

    Args:
        run: The episode loop.
        backlog: ``"latest"`` (newest observation wins) or a FIFO bound (overflow fails the episode).
        clock: ``"auto"`` (``SimClock(0.0)`` per sync episode, ``WallClock()`` per live one) or a factory.
        control_hz: ``ep.sim_dt = 1 / control_hz`` when ``EPISODE_START`` carries no ``control_hz``.
        reset / finish: Model-thread episode hooks (``finish`` runs after the episode's tasks and calls drained).
        requires: Client capabilities the loop needs (e.g. ``{"publish"}``); unmet starts fail.
        warn_unadvanced: Warn once when a sync loop never advances its ``SimClock``.
    """

    def __init__(
        self,
        server: ModelServer,
        run: Run,
        *,
        backlog: Backlog = "latest",
        clock: Literal["auto"] | ClockFactory = "auto",
        control_hz: float | None = None,
        reset: Reset | None = None,
        finish: Finish | None = None,
        requires: frozenset[str] = frozenset(),
        warn_unadvanced: bool = True,
    ) -> None:
        Mailbox(backlog)  # validates
        if control_hz is not None:
            _check_rate(control_hz)
        if unknown := set(requires) - set(CAPABILITIES):
            raise ValueError(f"unknown capabilities {sorted(unknown)}; known: {list(CAPABILITIES)}")
        self.server, self.run, self.backlog, self.clock = server, run, backlog, clock
        self.control_hz, self.reset, self.finish = control_hz, reset, finish
        self.requires, self.warn_unadvanced = frozenset(requires), warn_unadvanced
        self._eps: dict[int, EpisodeContext] = {}  # by row id
        self._tg: TaskGroup | None = None
        self._stopping = True  # until on_serve_start

    async def on_serve_start(self, tg: TaskGroup) -> None:
        self._tg, self._stopping = tg, False
        if self._on_fail not in self.server._fail_hooks:
            self.server._fail_hooks.append(self._on_fail)
        tg.start_soon(self._watch)

    async def on_episode_start(self, rows: Rows) -> dict[int, Exception] | None:
        failed: dict[int, Exception] = {}
        plans: dict[int, tuple[Clock, float | None]] = {}
        for i, row in enumerate(rows):  # validated before reset
            try:
                plans[i] = self._admit(row)
            except Exception as exc:
                failed[i] = exc
        accepted = [i for i in range(len(rows)) if i not in failed]
        if not accepted:
            return failed
        if self.reset is not None:
            try:
                out = await self.server.run_on_model_thread(self.reset, rows.take(accepted)) or {}
            except Exception as exc:
                out = dict.fromkeys(range(len(accepted)), exc)
            failed.update({accepted[j]: exc for j, exc in out.items()})
        ready = [i for i in accepted if i not in failed]
        created: list[EpisodeContext] = []
        try:
            if self._stopping or self._tg is None:
                raise RuntimeError("model server stopped")
            for i in ready:
                one = rows.take([i])
                rid = int(one.ids[0])
                ep = EpisodeContext(self, one, *plans[i], Mailbox(self.backlog), self.server._publishers.get(rid))
                created.append(ep)
                self._eps[rid] = ep
                self._tg.start_soon(self._host, ep)
                ep._scheduled = True
        except BaseException as exc:  # roll back: close, join, finish the reset rows once, then fail them
            for ep in created:
                ep._close()
                if self._eps.get(ep.row_id) is ep:
                    del self._eps[ep.row_id]
            with anyio.CancelScope(shield=True):
                for ep in created:
                    if ep._scheduled:
                        await ep._exited.wait()
                if self.finish is not None and ready:
                    aborted: list[EpisodeResult] = [{"status": "aborted", "failure_reason": None}] * len(ready)
                    try:
                        await self.server.run_on_model_thread(self.finish, rows.take(ready), aborted)
                    except Exception:
                        logger.exception("finish failed while rolling back a start")
            if not isinstance(exc, Exception):
                raise
            failed.update(dict.fromkeys(ready, exc))
        return failed

    async def on_observation(self, obs: Batch, rows: Rows) -> None:
        seq = rows._pending.seq if rows._pending is not None else None
        wall = time.monotonic()  # admission: latency counts the model queue too
        n = len(rows)
        for i in range(n):
            one, batch = (rows, obs) if n == 1 else (rows.take([i]), columns.take(obs, [i]))
            ep = self._eps.get(int(one.ids[0]))
            if ep is None or ep.closing or ep.episode is not one._episodes[0]:
                await self.server.send_error(one, EpisodeClosed("the episode is not running"))
                continue
            request = ObservationRequest(
                columns.split(batch, 1)[0], batch, one, int(one.step[0]), ep.clock.now(), wall, seq, self.server
            )
            try:
                ep._mailbox.put(request)
            except MailboxOverflow as exc:
                ep.fail(exc)
                continue
            if self.warn_unadvanced and isinstance(ep.clock, SimClock):
                ep._admitted += 1
                if ep._admitted == _UNADVANCED_WARN_AT and ep.clock.now() == ep._t0:
                    ep.logger.warning(
                        "%d observations and the SimClock never advanced; advance ep.clock in run() "
                        "(warn_unadvanced=False silences this)",
                        _UNADVANCED_WARN_AT,
                    )

    async def on_episode_end(self, rows: Rows, results: list[EpisodeResult]) -> None:
        eps = [ep for rid in rows.ids.tolist() if (ep := self._eps.pop(rid, None)) is not None]
        for ep in eps:
            ep._close()
        for ep in eps:  # tasks unwound and model calls drained before finish
            await ep._exited.wait()
        if self.finish is not None:
            await self.server.run_on_model_thread(self.finish, rows, results)

    # ------------------------------------------------------------------

    def _admit(self, row: Row) -> tuple[Clock, float | None]:
        episode = row.episode
        missing = self.requires - episode.capabilities
        if missing:
            raise MissingCapabilities(f"missing capabilities: {', '.join(sorted(missing))}")
        if "publish" in self.requires and episode.mode != "live":
            raise RuntimeError("this server publishes actions: it needs a live episode")
        clock = (SimClock(0.0) if episode.mode == "sync" else WallClock()) if self.clock == "auto" else None
        if clock is None:
            assert callable(self.clock)
            clock = self.clock(episode, episode.mode)
        hz = episode.config.get("control_hz")
        hz = self.control_hz if hz is None else _check_rate(hz)
        return clock, None if hz is None else 1.0 / hz

    async def _host(self, ep: EpisodeContext) -> None:
        with anyio.CancelScope(shield=True):  # cancellation reaches the episode only through ep._close()
            try:
                with ep._scope:
                    if ep.closing:
                        return
                    async with anyio.create_task_group() as tg:
                        ep._tg = tg
                        try:
                            ep.start_soon(self.run, ep)
                        except BaseException as exc:
                            _classify(ep, exc)
            finally:
                ep._close()
                await ep._drained.wait()
                ep._exited.set()
        if ep._control is not None:
            raise BaseExceptionGroup("episode control", [ep._control])

    async def _task(self, ep: EpisodeContext, fn: Callable[..., Awaitable[Any]], args: tuple[Any, ...]) -> None:
        try:
            if ep.closing:
                return
            await fn(*args)
        except BaseException as exc:
            _classify(ep, exc)
            if isinstance(exc, anyio.get_cancelled_exc_class()):
                raise
        finally:
            ep._tasks -= 1
            if ep._tasks == 0 and not ep.closing:
                ep.fail(RuntimeError("no task is left to answer observations"))

    def _on_fail(self, rid: int, gen: int, episode: Episode, error: Exception) -> None:
        ep = self._eps.get(rid)
        if ep is not None and ep.episode is episode:
            ep._failed = ep._failed or error
            ep._close()

    async def _watch(self) -> None:
        try:
            await anyio.sleep_forever()
        finally:  # server shutdown: admission closes, then every episode closes
            self._stopping = True
            for ep in list(self._eps.values()):
                ep._close()


def _check_rate(hz: Any) -> float:
    if isinstance(hz, bool) or not isinstance(hz, (int, float)) or not math.isfinite(hz) or hz <= 0:
        raise ValueError(f"control_hz must be a positive finite number, got {hz!r}")
    return float(hz)


class DaemonModelServer(ModelServer):
    """A ``ModelServer`` whose episodes each run ``run(ep)`` (override it, or pass ``run=``) with ``EpisodeDaemons``;
    ``reset_episodes`` / ``finish_episodes`` run on the model thread.  ``requires``: client capabilities needed.

    Continuous inference is a few lines (``backlog="latest"``: each inference takes the newest observation)::

        async def run(self, ep):
            async for request in ep.observations():
                await request.reply(await ep.call(self.predict, request.obs, request.row))

    Streaming a chunk on a time grid (RTC): ``examples/live_servers/rtc.py``.
    """

    supports_vector_step = False
    requires: frozenset[str] = frozenset()

    def __init__(
        self,
        run: Run | None = None,
        *,
        backlog: Backlog = "latest",
        clock: Literal["auto"] | ClockFactory = "auto",
        control_hz: float | None = None,
        max_rows: int | None = None,
        warn_unadvanced: bool = True,
    ) -> None:
        super().__init__()
        self.max_rows = max_rows
        self._run_fn = run
        self._warmed = False
        self.daemons = EpisodeDaemons(
            self,
            self.run,
            backlog=backlog,
            clock=clock,
            control_hz=control_hz,
            reset=self._reset,
            finish=self.finish_episodes,
            requires=self.requires,
            warn_unadvanced=warn_unadvanced,
        )

    async def run(self, ep: EpisodeContext) -> None:
        """One episode's loop: answer ``ep.observations()`` (``request.reply``), now or later."""
        if self._run_fn is None:
            raise NotImplementedError(f"{type(self).__name__} must override run() or pass run=")
        await self._run_fn(ep)

    def warmup(self, rows: Rows) -> None:
        """Runs once before the first ``reset_episodes`` (same thread and ``rows``), so an episode's first inference
        after ``EPISODE_READY`` runs at steady-state latency.  Raising fails that start; the next one retries."""

    def _reset(self, rows: Rows) -> dict[int, Exception] | None:
        if not self._warmed:
            self.warmup(rows)
            self._warmed = True
        return self.reset_episodes(rows)

    def reset_episodes(self, rows: Rows) -> dict[int, Exception] | None:
        """Reset the state of ``rows`` (``episode.config``: the ``EPISODE_START`` payload).  Blocking.
        Return ``{position: exception}`` for starts that failed, or raise to fail them all."""
        return None

    def finish_episodes(self, rows: Rows, results: list[EpisodeResult]) -> None:
        """Episodes in ``rows`` ended (``results[i]``: the ``EPISODE_END`` payload).  Blocking."""

    async def on_serve_start(self, tg: TaskGroup) -> None:
        await super().on_serve_start(tg)
        await self.daemons.on_serve_start(tg)

    async def on_episode_start(self, rows: Rows) -> dict[int, Exception] | None:
        return await self.daemons.on_episode_start(rows)

    async def on_observation(self, obs: Batch, rows: Rows) -> None:
        await self.daemons.on_observation(obs, rows)

    async def on_episode_end(self, rows: Rows, results: list[EpisodeResult]) -> None:
        await self.daemons.on_episode_end(rows, results)
