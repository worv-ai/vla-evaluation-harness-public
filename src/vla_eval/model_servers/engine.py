"""PredictionEngine: a model's per-row ``predict`` / column ``predict_batch`` behind one dispatcher that coalesces
requests across episodes into batches on the server's model thread.  ``PredictModelServer`` is one instance;
episode loops call ``infer`` / ``infer_one`` for fresh inference with cross-episode batching."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import anyio
import numpy as np
from anyio.lowlevel import checkpoint

from vla_eval import columns
from vla_eval.model_servers.base import EpisodeClosed, ModelServer, Row, Rows
from vla_eval.types import Action, Batch, EpisodeResult, Observation

if TYPE_CHECKING:
    from anyio.abc import TaskGroup

logger = logging.getLogger(__name__)

PredictOne = Callable[[Observation, Row], Action]
PredictBatch = Callable[[Batch, Rows], Batch]
Reset = Callable[[Rows], "dict[int, Exception] | None"]
Finish = Callable[[Rows, "list[EpisodeResult]"], None]

_ENDED = EpisodeClosed("the episode ended")
_STOPPED = RuntimeError("model server stopped")


@dataclass(eq=False)
class _Request:
    obs: Batch
    rows: Rows
    wait: bool  # may wait max_wait_time for other connections' requests
    raw: bool = False  # infer(): the model's full output, no chunk buffers
    done: anyio.Event = field(default_factory=anyio.Event)
    actions: Batch = field(default_factory=dict)
    errors: dict[int, Exception] = field(default_factory=dict)
    queued_at: float = field(default_factory=anyio.current_time)
    dispatched: bool = False


@dataclass
class PredictionResult:
    actions: Batch  # successful rows, in their relative order
    errors: dict[int, Exception]  # by position in the submitted Rows


class PredictionEngine:
    """Args:
    predict / predict_batch: the model (``predict_batch`` wins; the default runs ``predict`` per row).
    reset / finish: the episode hooks, on the model thread.
    chunk_size: Actions per inference for the observation path; the rest of each ``[T, D]`` chunk is buffered.
    max_batch_size: Most rows per ``predict_batch`` call (``None``: no limit).
    max_wait_time: Seconds to wait for other requests before inferring a partial batch.
    """

    def __init__(
        self,
        server: ModelServer,
        *,
        predict: PredictOne | None = None,
        predict_batch: PredictBatch | None = None,
        reset: Reset | None = None,
        finish: Finish | None = None,
        chunk_size: int | None = None,
        max_batch_size: int | None = 1,
        max_wait_time: float = 0.01,
    ) -> None:
        if predict is None and predict_batch is None:
            raise ValueError("PredictionEngine needs predict or predict_batch")
        self.server = server
        self.predict, self.predict_batch, self.reset, self.finish = predict, predict_batch, reset, finish
        self.chunk_size = chunk_size
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self._chunked = np.zeros(0, bool)  # by row id: the episode has a chunk size
        self._buffers: dict[int, deque[Any]] = {}  # chunked rows' remaining actions
        self._queue: list[_Request] = []
        self._wake = anyio.Event()
        self._stopping = True  # until on_serve_start

    # ------------------------------------------------------------------
    # ModelServer callbacks (an episode-owning server forwards these)
    # ------------------------------------------------------------------

    async def on_serve_start(self, tg: TaskGroup) -> None:
        self._queue, self._wake, self._stopping = [], anyio.Event(), False
        tg.start_soon(self._dispatch)

    async def on_episode_start(self, rows: Rows) -> dict[int, Exception] | None:
        for episode in rows.episodes:
            episode.chunk_size = self.chunk_size
        failed = await self.server.run_on_model_thread(self.reset, rows) if self.reset is not None else None
        table = self.server._table
        if len(self._chunked) < len(table):
            self._chunked = np.concatenate([self._chunked, np.zeros(len(table) - len(self._chunked), bool)])
        for rid, episode in zip(rows.ids.tolist(), rows.episodes):
            self._chunked[rid] = episode.chunk_size is not None
        return failed

    async def on_observation(self, obs: Batch, rows: Rows) -> None:
        idx = rows.ids
        if self._chunked[idx].all() and all(self._buffers.get(i) for i in idx.tolist()):  # no inference to wait for
            await self.server.send_action(
                rows, columns.stack([{"actions": self._buffers[i].popleft()} for i in idx.tolist()])
            )
            return
        if self._stopping:
            await self.server.send_error(rows, _STOPPED)
            return
        vector = rows._pending is not None and rows._pending.vector
        req = _Request(obs, rows, wait=len(rows) == 1 and not vector)
        self._queue.append(req)
        self._wake.set()
        await req.done.wait()
        await reply_prediction(self.server, rows, PredictionResult(req.actions, req.errors))

    async def on_episode_end(self, rows: Rows, results: list[EpisodeResult]) -> None:
        try:
            if self.finish is not None:
                await self.server.run_on_model_thread(self.finish, rows, results)
        finally:
            for rid in rows.ids.tolist():
                self._buffers.pop(rid, None)

    # ------------------------------------------------------------------
    # Raw inference for episode loops
    # ------------------------------------------------------------------

    async def infer(self, obs: Batch, rows: Rows, *, wait_for_batch: bool = True) -> PredictionResult:
        """Fresh inference of ``rows`` (a row each of ``obs``) on the model thread, batched with other callers'
        requests within ``max_wait_time``; the full output, no chunk buffers.  Rows that ended fail."""
        await checkpoint()
        if self._stopping:
            raise RuntimeError("model server stopped")
        req = _Request(obs, rows, wait=wait_for_batch, raw=True)
        self._queue.append(req)
        self._wake.set()
        try:
            await req.done.wait()
        except BaseException:
            if not req.dispatched:  # still queued: withdrawn
                if req in self._queue:
                    self._queue.remove(req)
                self._wake.set()
            else:  # running: returns only after the model call
                with anyio.CancelScope(shield=True):
                    await req.done.wait()
            raise
        return PredictionResult(req.actions, req.errors)

    async def infer_one(self, obs: Observation, row: Row, *, wait_for_batch: bool = True) -> Action:
        """:meth:`infer` for one row of a running episode; raises that row's error."""
        server, rid = self.server, row.id
        if not (0 <= rid < server._size and server._table[rid] is row.episode) or server._closing[rid]:
            raise EpisodeClosed("the episode ended")
        if rid in server._failed:
            raise EpisodeClosed("the episode failed")
        idx = np.asarray([rid], np.int64)  # this incarnation, at the row's own step
        rows = Rows(idx, np.asarray([row.step], np.int64), server._table[idx], server._gen[idx])
        result = await self.infer(columns.stack([obs]), rows, wait_for_batch=wait_for_batch)
        if result.errors:
            error = result.errors[0]
            raise EpisodeClosed(str(error)) if error is _ENDED else error
        return columns.split(result.actions, 1)[0]

    # ------------------------------------------------------------------
    # Dispatch: coalesce requests, serve chunk buffers, infer
    # ------------------------------------------------------------------

    async def _dispatch(self) -> None:
        try:
            await self._dispatch_loop()
        except (KeyboardInterrupt, SystemExit) as exc:  # never bare out of an asyncio task
            raise BaseExceptionGroup("model control", [exc]) from None
        finally:  # shutdown: no request waits forever
            self._stopping = True
            _fail(self._queue, _STOPPED)
            self._queue = []

    async def _dispatch_loop(self) -> None:
        while True:
            await self._wake.wait()
            self._wake = anyio.Event()
            while self._queue:
                # wait for single-episode requests of other connections; a vector step is a whole step already
                if self.max_wait_time > 0 and not self._full() and all(r.wait for r in self._queue):
                    deadline = self._queue[0].queued_at + self.max_wait_time  # from the oldest request's arrival
                    with anyio.CancelScope(deadline=deadline):
                        while self._queue and not self._full() and all(r.wait for r in self._queue):
                            await self._wake.wait()
                            self._wake = anyio.Event()
                    if not self._queue:  # withdrawn meanwhile
                        continue
                batch = [self._queue.pop(0)]
                rows = len(batch[0].rows)
                limit = self.max_batch_size
                while (
                    self._queue
                    and self._queue[0].raw == batch[0].raw
                    and (limit is None or rows + len(self._queue[0].rows) <= limit)
                ):
                    rows += len(self._queue[0].rows)
                    batch.append(self._queue.pop(0))
                for r in batch:
                    r.dispatched = True
                await self._run(batch)

    def _full(self) -> bool:
        return self.max_batch_size is not None and sum(len(r.rows) for r in self._queue) >= self.max_batch_size

    async def _run(self, batch: list[_Request]) -> None:
        if len(batch) > 1:
            try:
                obs = columns.concat([r.obs for r in batch])
            except Exception:  # requests of different shapes: one at a time
                try:
                    for r in batch:
                        await self._run([r])
                finally:  # cancelled: the rest fail
                    _fail(batch, _STOPPED)
                return
            rows = _merge([r.rows for r in batch])
        else:
            obs, rows = batch[0].obs, batch[0].rows
        try:
            actions, errors = await self._infer(obs, rows, batch[0].raw)
            start = ok_seen = 0
            for r in batch:
                n = len(r.rows)
                r.errors = {i - start: errors[i] for i in range(start, start + n) if i in errors}
                ok = n - len(r.errors)
                r.actions = actions if len(batch) == 1 else columns.take(actions, range(ok_seen, ok_seen + ok))
                ok_seen += ok
                start += n
            for r in batch:
                r.done.set()
        except Exception as exc:
            logger.exception("dispatch failed")
            _fail(batch, exc)
        finally:  # cancelled
            _fail(batch, _STOPPED)

    async def _infer(self, obs: Batch, rows: Rows, raw: bool = False) -> tuple[Batch, dict[int, Exception]]:
        """The actions of every position that did not fail, in order, and the failures by position.  One
        model-thread call."""
        n = len(rows)
        ids = rows.ids
        await checkpoint()  # the last await before the model call, so a row ending meanwhile is filtered here
        current = self.server._current(rows)
        errors: dict[int, Exception] = {}
        if not current.all():  # ended while queued: not inferred
            errors = dict.fromkeys(np.flatnonzero(~current).tolist(), _ENDED)
        chunked = [] if raw else np.flatnonzero(self._chunked[ids] & current).tolist()
        served: dict[int, Any] = {}
        for i in chunked:
            buffer = self._buffers.get(int(ids[i]))
            if buffer:
                served[i] = buffer.popleft()
        todo = [i for i in range(n) if i not in served and i not in errors] if served or errors else list(range(n))
        parts: list[tuple[list[int], Batch]] = []
        if todo:
            failed, parts = await self.server._run_submitted(self._work, obs, rows, todo)
            errors.update(failed)
        if raw and parts and not (current := self.server._current(rows)).all():  # ended during the call
            parts = [_drop_ended(positions, out, current, errors) for positions, out in parts]
        if not chunked:
            return columns.concat([p for _, p in parts]), errors
        # A chunked row returns its chunk's first action and keeps the rest; every row returns "actions" only.
        current = self.server._current(rows)  # a row that ended meanwhile keeps no buffer: its id may run another
        action: dict[int, Any] = {i: a for i, a in served.items() if i not in errors}
        for positions, out in parts:
            for j, i in enumerate(positions):
                try:
                    a = out["actions"][j]
                    cs = rows._episodes[i].chunk_size
                    if cs is not None and np.ndim(a) == 2:
                        a = a[:cs]
                        if current[i]:
                            self._buffers[int(ids[i])] = deque(a[1:])
                        a = a[0]
                    action[i] = a
                except Exception as exc:  # e.g. an empty chunk: this row alone
                    errors[i] = exc
        return columns.stack([{"actions": action[i]} for i in range(n) if i not in errors]), errors

    def _work(
        self, obs: Batch, rows: Rows, todo: list[int]
    ) -> tuple[dict[int, Exception], list[tuple[list[int], Batch]]]:
        """On the model thread: predict the ``todo`` positions, in slices of ``max_batch_size``."""
        errors: dict[int, Exception] = {}
        n = len(rows)
        parts: list[tuple[list[int], Batch]] = []
        batched = self.predict_batch is not None
        size = (self.max_batch_size if batched else None) or max(len(todo), 1)  # predict runs row by row anyway
        for start in range(0, len(todo), size):
            group = todo[start : start + size]
            sub, sub_rows = (obs, rows) if len(group) == n else (columns.take(obs, group), rows.take(group))
            try:
                if self.predict_batch is None:
                    out, failed = self._predict_rows(sub, sub_rows)
                else:
                    out, failed = self.predict_batch(sub, sub_rows), {}
                    if not isinstance(out, dict) or columns.num_rows(out) != len(group):
                        raise RuntimeError(
                            f"predict_batch returned {type(out).__name__} for {len(group)} rows: it takes and returns "
                            "column batches (a row per episode), no longer lists of dicts"
                        )
                    out = _float32(out)
            except Exception as exc:
                errors.update(dict.fromkeys(group, exc))
                continue
            errors.update({group[j]: exc for j, exc in failed.items()})
            parts.append(([i for j, i in enumerate(group) if j not in failed], out))
        return errors, parts

    def _predict_rows(self, obs: Batch, rows: Rows) -> tuple[Batch, dict[int, Exception]]:
        """``predict`` per row; a row's exception fails it alone."""
        assert self.predict is not None
        out: list[Action] = []
        failed: dict[int, Exception] = {}
        for i, row_obs in enumerate(columns.split(obs, len(rows))):
            try:
                action = _float32(self.predict(row_obs, rows[i]))
                if out and action.keys() != out[0].keys():
                    raise RuntimeError(f"action keys {sorted(action)} differ from the batch's {sorted(out[0])}")
                out.append(action)
            except Exception as exc:
                failed[i] = exc
        return columns.stack(out), failed


async def reply_prediction(server: ModelServer, rows: Rows, result: PredictionResult) -> None:
    """Answer ``rows`` with ``result``: its actions to the rows that succeeded, each error to its rows."""
    actions, errors = result.actions, result.errors
    if not errors:
        await server.send_action(rows, actions)
        return
    ok = [i for i in range(len(rows)) if i not in errors]
    if ok:
        await server.send_action(rows.take(ok), actions)
    by_error: dict[int, tuple[Exception, list[int]]] = {}
    for i, exc in errors.items():
        by_error.setdefault(id(exc), (exc, []))[1].append(i)
    for exc, positions in by_error.values():
        await server.send_error(rows.take(positions), exc)


def _drop_ended(
    positions: list[int], out: Batch, current: np.ndarray, errors: dict[int, Exception]
) -> tuple[list[int], Batch]:
    """``positions`` and their rows of ``out`` whose episode still runs; the others fail as ended."""
    keep = [j for j, i in enumerate(positions) if current[i]]
    errors.update({i: _ENDED for i in positions if not current[i]})
    return [positions[j] for j in keep], columns.take(out, keep) if len(keep) < len(positions) else out


def _fail(requests: list[_Request], exc: Exception) -> None:
    """Complete the requests not completed yet with ``exc`` for every row."""
    for r in requests:
        if not r.done.is_set():
            r.actions, r.errors = {}, dict.fromkeys(range(len(r.rows)), exc)
            r.done.set()


def _merge(parts: list[Rows]) -> Rows:
    """The rows of several requests, one after another (for inference: no replies through them)."""
    return Rows(
        np.concatenate([r.ids for r in parts]),
        np.concatenate([r.step for r in parts]),
        np.concatenate([r._episodes for r in parts]),
        np.concatenate([r._gen for r in parts]),
    )


def _float32(actions: Batch) -> Batch:
    """``actions`` as float32 (an array, or each item of a list column), as every model's actions travel."""
    a = actions.get("actions")
    if a is None:
        return actions
    if isinstance(a, list):
        return {**actions, "actions": [np.asarray(x, np.float32) for x in a]}
    return {**actions, "actions": np.asarray(a, np.float32)}
