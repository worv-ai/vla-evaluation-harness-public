"""PredictModelServer: override ``predict(obs, row)`` (one episode's observation), or ``predict_batch(obs, rows)``
(column batch, a row per episode) when batched inference beats per-row; the default calls ``predict`` per row."""

from __future__ import annotations

import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import anyio
import numpy as np
from anyio.abc import TaskGroup
from anyio.to_thread import run_sync as _run_in_thread

from vla_eval import columns
from vla_eval.types import Action, Batch, EpisodeResult, Observation
from vla_eval.model_servers.base import ModelServer, Row, Rows

logger = logging.getLogger(__name__)

_ENDED = RuntimeError("the episode ended")
_HERE = np.zeros(1, np.int64)


@dataclass(eq=False)
class _Request:
    obs: Batch
    rows: Rows
    pre: list[tuple[Batch, Rows, np.ndarray]]  # observe these first: (obs, rows, their positions in this request)
    wait: bool  # may wait max_wait_time for other connections' requests
    done: anyio.Event = field(default_factory=anyio.Event)
    actions: Batch = field(default_factory=dict)
    errors: dict[int, Exception] = field(default_factory=dict)


class _Stream:
    """Continuous inference of one row: its observations not observed yet, and the latest with its arrival time."""

    def __init__(self) -> None:
        self.unobserved: list[tuple[Batch, Rows]] = []
        self.latest: tuple[Batch, Rows, float] | None = None
        self.wake = anyio.Event()
        self.stopped = False
        self.exited = anyio.Event()


class PredictModelServer(ModelServer):
    """Args:
    chunk_size: Actions per inference; the rest of each ``[T, D]`` chunk is served from a per-episode buffer.
        Leave unset if the model must infer every observation (``observe`` sees every one anyway).
    max_batch_size: Most rows per ``predict_batch`` call (``None``: no limit); raise it to batch episodes.
    max_wait_time: Seconds to wait for other connections' requests before inferring a partial batch.
    max_rows: Most concurrent episodes (``None``: no limit); 1 for a model whose state is not per episode.
    continuous_inference: Infer each episode's latest observation in a loop and send its action, chunks
        unbuffered (draft, untested on a real model).  laas: send the action ``int(latency * hz)`` steps in.

    The hooks below run on one model thread, in arrival order; ``__init__`` runs on the caller's thread, so
    create thread-bound state (CUDA graphs, JAX) in the first ``reset_episodes`` / ``predict``.
    """

    def __init__(
        self,
        *,
        chunk_size: int | None = None,
        max_batch_size: int | None = 1,
        max_wait_time: float = 0.01,
        max_rows: int | None = None,
        continuous_inference: bool = False,
        laas: bool = False,
        hz: float = 10.0,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.max_rows = max_rows
        self.continuous_inference = continuous_inference
        self.laas = laas
        self.hz = hz
        self._chunked = np.zeros(0, bool)  # by row id: the episode has a chunk size
        self._buffers: dict[int, deque[Any]] = {}  # chunked rows' remaining actions
        self._streams: dict[int, _Stream] = {}  # continuous inference, by row id
        self._observes = type(self).observe is not PredictModelServer.observe
        self._tg: TaskGroup | None = None
        self._queue: list[_Request] = []
        self._wake = anyio.Event()
        # Every model call on one thread, one at a time: CUDA graphs (torch.compile) and JAX are thread-bound.
        self._model_thread = ThreadPoolExecutor(1, thread_name_prefix="model")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name, hook in (("on_episode_start", "reset_episodes"), ("on_episode_end", "finish_episodes")):
            if name in cls.__dict__:  # the pre-callback hook names: a silent no-op otherwise
                raise TypeError(f"{cls.__name__}.{name}: PredictModelServer implements it; override {hook}()")

    # ------------------------------------------------------------------
    # Override these
    # ------------------------------------------------------------------

    def reset_episodes(self, rows: Rows) -> dict[int, Exception] | None:
        """Reset the state of ``rows`` (``episode.config``: the ``EPISODE_START`` payload).  Blocking.
        Return ``{position: exception}`` for starts that failed, or raise to fail them all."""
        return None

    def observe(self, obs: Batch, rows: Rows) -> dict[int, Exception] | None:
        """Every observation, once, in order, before it is inferred or answered from a chunk buffer (e.g. to
        keep a state history).  Blocking; ``{position: exception}`` fails those rows' observations."""
        return None

    def finish_episodes(self, rows: Rows, results: list[EpisodeResult]) -> None:
        """Episodes in ``rows`` ended (``results[i]``: the ``EPISODE_END`` payload).  Blocking."""

    def predict(self, obs: Observation, row: Row) -> Action:
        """One observation of the episode in ``row``.  Blocking."""
        raise NotImplementedError(f"{type(self).__name__} must override predict() or predict_batch()")

    def predict_batch(self, obs: Batch, rows: Rows) -> Batch:
        """Observation columns of ``rows`` to action columns, a row each.  Blocking."""
        actions, errors = self._predict_rows(obs, rows)
        if errors:
            raise next(iter(errors.values()))
        return actions

    def pick_action(self, actions: np.ndarray, obs_time: float) -> np.ndarray:
        """Continuous inference: the action to send from ``actions`` for an observation received at ``obs_time``."""
        if actions.ndim == 1:
            return actions
        if not (self.laas and self.hz > 0):
            return actions[0]
        steps = int((time.monotonic() - obs_time) * self.hz)
        if steps >= len(actions):
            logger.warning("LAAS: %d steps late for a %d-step chunk; sending its last action", steps, len(actions))
        return actions[min(steps, len(actions) - 1)]

    # ------------------------------------------------------------------
    # ModelServer callbacks
    # ------------------------------------------------------------------

    async def on_serve_start(self, tg: TaskGroup) -> None:
        self._tg = tg
        self._queue, self._wake = [], anyio.Event()
        tg.start_soon(self._dispatch)

    async def on_episode_start(self, rows: Rows) -> dict[int, Exception] | None:
        for episode in rows.episodes:
            episode.chunk_size = self.chunk_size
        failed = await self._call(self.reset_episodes, rows) or {}
        if len(self._chunked) < len(self._table):
            self._chunked = np.concatenate([self._chunked, np.zeros(len(self._table) - len(self._chunked), bool)])
        for i, (rid, episode) in enumerate(zip(rows.ids.tolist(), rows.episodes)):
            self._chunked[rid] = episode.chunk_size is not None
            if self.continuous_inference and i not in failed:
                assert self._tg is not None, "serve_async calls on_serve_start first"
                self._streams[rid] = stream = _Stream()
                self._tg.start_soon(self._stream, stream, rid)
        return failed

    async def on_observation(self, obs: Batch, rows: Rows) -> None:
        if self.continuous_inference:  # a stream infers it; returns at once
            t = time.monotonic()
            for i in range(len(rows)):
                o, r = (obs, rows) if len(rows) == 1 else (columns.take(obs, [i]), rows.take([i]))
                stream = self._streams[int(r.ids[0])]
                if self._observes:
                    stream.unobserved.append((o, r))
                stream.latest = (o, r, t)
                stream.wake.set()
            return
        idx = rows.ids
        here = [(obs, rows, np.arange(len(rows)))] if self._observes else []
        if self._chunked[idx].all() and all(self._buffers.get(i) for i in idx.tolist()):  # no inference to wait for
            if not here:
                await self.send_action(
                    rows, columns.stack([{"actions": self._buffers[i].popleft()} for i in idx.tolist()])
                )
                return
            actions, errors = await self._infer(obs, rows, here)
        else:
            vector = rows._pending is not None and rows._pending.vector
            req = _Request(obs, rows, here, wait=len(rows) == 1 and not vector)
            await self._submit(req)
            actions, errors = req.actions, req.errors
        await self._reply(rows, actions, errors)

    async def on_episode_end(self, rows: Rows, results: list[EpisodeResult]) -> None:
        streams = [s for rid in rows.ids.tolist() if (s := self._streams.pop(rid, None)) is not None]
        for s in streams:
            s.stopped = True
            s.wake.set()
        for s in streams:  # an inference in flight finishes; its action is dropped
            await s.exited.wait()
        try:
            await self._call(self._finish, [x for s in streams for x in s.unobserved], rows, results)
        finally:
            for rid in rows.ids.tolist():
                self._buffers.pop(rid, None)

    # ------------------------------------------------------------------
    # Requests
    # ------------------------------------------------------------------

    def _finish(self, unobserved: list[tuple[Batch, Rows]], rows: Rows, results: list[EpisodeResult]) -> None:
        """On the model thread: observe what continuous inference left unobserved, then ``finish_episodes``."""
        for o, r in unobserved:
            try:
                self.observe(o, r)
            except Exception:
                logger.exception("observe failed")
        self.finish_episodes(rows, results)

    async def _call(self, fn: Any, *args: Any) -> Any:
        return await _run_in_thread(self._model_thread.submit(fn, *args).result)

    async def _submit(self, req: _Request) -> None:
        self._queue.append(req)
        self._wake.set()
        await req.done.wait()

    async def _reply(self, rows: Rows, actions: Batch, errors: dict[int, Exception]) -> None:
        if not errors:
            await self.send_action(rows, actions)
            return
        ok = [i for i in range(len(rows)) if i not in errors]
        if ok:
            await self.send_action(rows.take(ok), actions)
        by_error: dict[int, tuple[Exception, list[int]]] = {}
        for i, exc in errors.items():
            by_error.setdefault(id(exc), (exc, []))[1].append(i)
        for exc, positions in by_error.values():
            await self.send_error(rows.take(positions), exc)

    async def _stream(self, s: _Stream, rid: int) -> None:
        """Continuous inference of one row: infer its latest observation, send the action, repeat."""
        try:
            while True:
                await s.wake.wait()
                s.wake = anyio.Event()
                if s.stopped or s.latest is None:
                    return
                obs, rows, t = s.latest
                pre, s.unobserved = [(o, r, _HERE) for o, r in s.unobserved], []
                req = _Request(obs, rows, pre, wait=True)
                await self._submit(req)
                if s.stopped:
                    return
                try:
                    if req.errors:
                        raise req.errors[0]
                    action = columns.split(req.actions, 1)[0]
                    if "actions" in action:
                        action["actions"] = self.pick_action(np.asarray(action["actions"]), t)
                    out = columns.stack([action])
                except Exception as exc:  # this observation fails; the loop goes on
                    await self.send_error(rows, exc)
                    continue
                await self.send_action(rows, out)
        except Exception:
            logger.exception("continuous inference stopped row=%d", rid)
        finally:
            s.exited.set()

    # ------------------------------------------------------------------
    # Dispatch: coalesce requests, serve chunk buffers, infer
    # ------------------------------------------------------------------

    async def _dispatch(self) -> None:
        try:
            await self._dispatch_loop()
        finally:  # shutdown: no request waits forever
            stopped = RuntimeError("model server stopped")
            for r in self._queue:
                r.actions = {}
                r.errors = {i: stopped for i in range(len(r.rows))}
                r.done.set()
            self._queue = []

    async def _dispatch_loop(self) -> None:
        while True:
            await self._wake.wait()
            self._wake = anyio.Event()
            while self._queue:
                # wait for single-episode requests of other connections; a vector step is a whole step already
                if self.max_wait_time > 0 and not self._full() and all(r.wait for r in self._queue):
                    with anyio.move_on_after(self.max_wait_time):
                        while not self._full():
                            await self._wake.wait()
                            self._wake = anyio.Event()
                batch = [self._queue.pop(0)]
                rows = len(batch[0].rows)
                limit = self.max_batch_size
                while self._queue and (limit is None or rows + len(self._queue[0].rows) <= limit):
                    rows += len(self._queue[0].rows)
                    batch.append(self._queue.pop(0))
                await self._run(batch)

    def _full(self) -> bool:
        return self.max_batch_size is not None and sum(len(r.rows) for r in self._queue) >= self.max_batch_size

    async def _run(self, batch: list[_Request]) -> None:
        if len(batch) > 1:
            try:
                obs = columns.concat([r.obs for r in batch])
            except Exception:  # requests of different shapes: one at a time
                for r in batch:
                    await self._run([r])
                return
            rows = _merge([r.rows for r in batch])
            pre, start = [], 0
            for r in batch:
                pre += [(o, rs, where + start) for o, rs, where in r.pre]
                start += len(r.rows)
        else:
            obs, rows, pre = batch[0].obs, batch[0].rows, batch[0].pre
        try:
            actions, errors = await self._infer(obs, rows, pre)
            start = ok_seen = 0
            for r in batch:
                n = len(r.rows)
                r.errors = {i - start: errors[i] for i in range(start, start + n) if i in errors}
                ok = n - len(r.errors)
                r.actions = actions if len(batch) == 1 else columns.take(actions, range(ok_seen, ok_seen + ok))
                ok_seen += ok
                start += n
        except Exception as exc:
            logger.exception("dispatch failed")
            for r in batch:
                r.actions, r.errors = {}, dict.fromkeys(range(len(r.rows)), exc)
        finally:
            for r in batch:
                r.done.set()

    async def _infer(
        self, obs: Batch, rows: Rows, pre: list[tuple[Batch, Rows, np.ndarray]]
    ) -> tuple[Batch, dict[int, Exception]]:
        """Observe ``pre``, then the actions of every position that did not fail, in order, and the failures by
        position.  One model-thread call."""
        n = len(rows)
        ids = rows.ids
        current = self._current(rows)
        errors: dict[int, Exception] = {}
        if not current.all():  # ended while queued: not inferred; observed unless its row runs another episode
            errors = dict.fromkeys(np.flatnonzero(~current).tolist(), _ENDED)
            pre = _without(pre, {i for i in errors if self._table[ids[i]] is not rows._episodes[i]})
        chunked = np.flatnonzero(self._chunked[ids] & current).tolist()
        served: dict[int, Any] = {}
        for i in chunked:
            buffer = self._buffers.get(int(ids[i]))
            if buffer:
                served[i] = buffer.popleft()
        todo = [i for i in range(n) if i not in served and i not in errors] if served or errors else list(range(n))
        parts: list[tuple[list[int], Batch]] = []
        if todo or pre:
            failed, parts = await self._call(self._work, obs, rows, todo, pre)
            errors.update(failed)
        if not chunked:
            return columns.concat([p for _, p in parts]), errors
        # A chunked row returns its chunk's first action and keeps the rest; every row returns "actions" only.
        current = self._current(rows)  # a row that ended meanwhile keeps no buffer: its id may run another episode
        action: dict[int, Any] = {i: a for i, a in served.items() if i not in errors}
        for positions, out in parts:
            for j, i in enumerate(positions):
                try:
                    a = out["actions"][j]
                    cs = rows._episodes[i].chunk_size
                    if cs is not None and np.ndim(a) == 2:
                        a = a[:cs]
                        if not self.continuous_inference:  # CI sends the whole chunk to pick_action
                            if current[i]:
                                self._buffers[int(ids[i])] = deque(a[1:])
                            a = a[0]
                    action[i] = a
                except Exception as exc:  # e.g. an empty chunk: this row alone
                    errors[i] = exc
        return columns.stack([{"actions": action[i]} for i in range(n) if i not in errors]), errors

    def _work(
        self, obs: Batch, rows: Rows, todo: list[int], pre: list[tuple[Batch, Rows, np.ndarray]]
    ) -> tuple[dict[int, Exception], list[tuple[list[int], Batch]]]:
        """On the model thread: ``observe`` each of ``pre`` in order, then predict the ``todo`` positions it did not
        fail, in slices of ``max_batch_size``."""
        errors: dict[int, Exception] = {}
        for o, r, where in pre:
            try:
                bad = self.observe(o, r) or {}
            except Exception as exc:
                bad = dict.fromkeys(range(len(r)), exc)
            for j, exc in bad.items():
                errors.setdefault(int(where[j]), exc)
        if errors:
            todo = [i for i in todo if i not in errors]
        n = len(rows)
        parts: list[tuple[list[int], Batch]] = []
        batched = type(self).predict_batch is not PredictModelServer.predict_batch
        size = (self.max_batch_size if batched else None) or max(len(todo), 1)  # predict runs row by row anyway
        for start in range(0, len(todo), size):
            group = todo[start : start + size]
            sub, sub_rows = (obs, rows) if len(group) == n else (columns.take(obs, group), rows.take(group))
            try:
                if not batched:
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
        """:meth:`predict` per row; a row's exception fails it alone."""
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


def _merge(parts: list[Rows]) -> Rows:
    """The rows of several requests, one after another (for inference: no replies through them)."""
    return Rows(
        np.concatenate([r.ids for r in parts]),
        np.concatenate([r.step for r in parts]),
        np.concatenate([r._episodes for r in parts]),
        np.concatenate([r._gen for r in parts]),
    )


def _without(pre: list[tuple[Batch, Rows, np.ndarray]], skip: set[int]) -> list[tuple[Batch, Rows, np.ndarray]]:
    """``pre`` without the rows at positions in ``skip``."""
    out = []
    for o, r, where in pre:
        keep = [j for j, w in enumerate(where.tolist()) if w not in skip]
        if len(keep) == len(where):
            out.append((o, r, where))
        elif keep:
            out.append((columns.take(o, keep), r.take(keep), where[keep]))
    return out


def _float32(actions: Batch) -> Batch:
    """``actions`` as float32 (an array, or each item of a list column), as every model's actions travel."""
    a = actions.get("actions")
    if a is None:
        return actions
    if isinstance(a, list):
        return {**actions, "actions": [np.asarray(x, np.float32) for x in a]}
    return {**actions, "actions": np.asarray(a, np.float32)}
