"""PredictModelServer: the model server to subclass.

Override one of:

- ``predict(obs, row) -> action``: one observation (a dict) of one episode.  Simple, and enough when the server
  sees one episode at a time.
- ``predict_batch(obs, rows) -> actions``: a :data:`~vla_eval.columns.Batch` of observations, a row per
  episode, to a batch of actions.  Use it when the model infers a batch faster than its rows one by one: vector
  benchmarks (hundreds of environments per step) and several shards on one server.  ``rows.ids`` index
  per-episode state kept in arrays.

The default ``predict_batch`` calls ``predict`` per row.  Episodes start and end through ``on_episode_start(rows)``
/ ``on_episode_end(rows, results)``, batched the same way.

With ``chunk_size``, the model returns a chunk ``[T, D]`` per row; the server trims it to ``chunk_size``, returns
its first action and serves the rest from a per-episode buffer before inferring that episode again.  A server that
must see every observation (it keeps a history, or queues actions itself) leaves ``chunk_size`` unset.
"""

from __future__ import annotations

import heapq
import logging
import uuid
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import anyio
import numpy as np
from anyio.abc import TaskGroup
from anyio.to_thread import run_sync as _run_in_thread

from vla_eval import columns
from vla_eval.columns import Batch
from vla_eval.model_servers.base import Episode, ModelServer, Row, Rows
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class _Request:
    obs: Batch
    ids: np.ndarray
    done: anyio.Event = field(default_factory=anyio.Event)
    actions: Batch = field(default_factory=dict)
    errors: dict[int, Exception] = field(default_factory=dict)


class PredictModelServer(ModelServer):
    """Args:
    chunk_size: Actions per inference of an episode (``None``: every observation is inferred and the model's
        output is returned as is).
    max_batch_size: Most rows per ``predict_batch`` call (``None``: no limit).  The default 1 infers episodes one
        at a time; raise it (to ``num_envs`` for a vector benchmark) to batch them.
    max_wait_time: Seconds to wait for more requests before inferring a batch smaller than ``max_batch_size``
        (``0``: infer what has arrived).  Requests from several connections coalesce into one batch.
    max_rows: Most episodes at once (``None``: no limit); 1 for a model whose state is not per episode.
    """

    def __init__(
        self,
        *,
        chunk_size: int | None = None,
        max_batch_size: int | None = 1,
        max_wait_time: float = 0.01,
        max_rows: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.max_rows = max_rows
        self._episodes: list[Episode | None] = []  # by row id
        self._free: list[int] = []  # ended rows' ids, reused smallest first
        self._steps = np.zeros(0, np.int64)
        self._chunked = np.zeros(0, bool)  # by row id: the episode has a chunk size
        self._buffers: dict[int, deque[Any]] = {}  # chunked rows' remaining actions
        self._queue: list[_Request] = []
        self._wake = anyio.Event()
        self._thread_limiter = anyio.CapacityLimiter(1)  # one call into the model at a time

    # ------------------------------------------------------------------
    # Override these
    # ------------------------------------------------------------------

    def predict(self, obs: Observation, row: Row) -> Action:
        """One observation of the episode in ``row``.  Blocking."""
        raise NotImplementedError(f"{type(self).__name__} must override predict() or predict_batch()")

    def predict_batch(self, obs: Batch, rows: Rows) -> Batch:
        """Observations of ``rows`` (row ``i`` of every column for ``rows[i]``) to action columns with a row each.
        Blocking.  Default: :meth:`predict` per row."""
        actions, errors = self._predict_rows(obs, rows)
        if errors:
            raise next(iter(errors.values()))
        return actions

    # ------------------------------------------------------------------
    # Framework side (serve.py)
    # ------------------------------------------------------------------

    async def on_serve_start(self, tg: TaskGroup) -> None:
        self._queue, self._wake = [], anyio.Event()
        self._thread_limiter = anyio.CapacityLimiter(1)
        tg.start_soon(self._dispatch)

    async def start_episodes(self, starts: list[tuple[dict[str, Any], str]]) -> list[int | Exception]:
        out: list[int | Exception] = []
        ids: list[int] = []
        for config, session_id in starts:
            if self.max_rows is not None and len(self._episodes) - len(self._free) >= self.max_rows:
                out.append(RuntimeError(f"{type(self).__name__} runs at most {self.max_rows} episodes at once"))
                continue
            rec = config.get("recording") or {}
            rid = self._alloc()
            self._episodes[rid] = Episode(
                config=config,
                episode_id=rec.get("eid") or str(uuid.uuid4()),
                session_id=rec.get("sid") or session_id,
                eval_id=rec.get("eval_id") or "",
                recording_db_path=rec.get("db_path") or "",
                chunk_size=self.chunk_size,
            )
            self._steps[rid] = 0
            ids.append(rid)
            out.append(rid)
        if not ids:
            return out
        try:
            failed = await self._call(self.on_episode_start, self._rows(ids)) or {}
        except Exception as exc:
            failed = dict.fromkeys(range(len(ids)), exc)
        bad = {ids[i]: exc for i, exc in failed.items()}
        for rid in ids:
            episode = self._episodes[rid]
            self._chunked[rid] = episode is not None and episode.chunk_size is not None
        for rid, exc in bad.items():
            logger.warning("on_episode_start failed for row %d: %s: %s", rid, type(exc).__name__, exc)
            self._release(rid)
        return [bad.get(x, x) if isinstance(x, int) else x for x in out]

    async def end_episodes(self, ids: Sequence[int], results: list[dict[str, Any]]) -> None:
        live = [(i, r) for i, r in zip(ids, results) if 0 <= i < len(self._episodes) and self._episodes[i] is not None]
        if not live:
            return
        try:
            await self._call(self.on_episode_end, self._rows([i for i, _ in live]), [r for _, r in live])
        except Exception:
            logger.exception("on_episode_end failed")
        finally:
            for rid, _ in live:
                self._release(rid)

    async def act(self, obs: Batch, ids: Sequence[int]) -> tuple[Batch, dict[int, Exception]]:
        idx = np.asarray(ids, np.int64)
        if self._chunked[idx].all() and all(self._buffers.get(i) for i in idx.tolist()):  # no inference to wait for
            self._steps[idx] += 1
            return columns.stack([{"actions": self._buffers[i].popleft()} for i in idx.tolist()]), {}
        req = _Request(obs, idx)
        self._queue.append(req)
        self._wake.set()
        await req.done.wait()
        return req.actions, req.errors

    # ------------------------------------------------------------------
    # Rows
    # ------------------------------------------------------------------

    def _alloc(self) -> int:
        if self._free:
            return heapq.heappop(self._free)
        self._episodes.append(None)
        if len(self._episodes) > len(self._steps):
            grow = max(16, len(self._steps))
            self._steps = np.concatenate([self._steps, np.zeros(grow, np.int64)])
            self._chunked = np.concatenate([self._chunked, np.zeros(grow, bool)])
        return len(self._episodes) - 1

    def _release(self, rid: int) -> None:
        self._episodes[rid] = None
        self._buffers.pop(rid, None)
        heapq.heappush(self._free, rid)

    def _rows(self, ids: Sequence[int] | np.ndarray) -> Rows:
        idx = np.asarray(ids, np.int64)
        return Rows(idx, self._steps[idx], self._episodes)

    async def _call(self, fn: Any, *args: Any) -> Any:
        return await _run_in_thread(fn, *args, limiter=self._thread_limiter)

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
                r.errors = {i: stopped for i in range(len(r.ids))}
                r.done.set()
            self._queue = []

    async def _dispatch_loop(self) -> None:
        while True:
            await self._wake.wait()
            self._wake = anyio.Event()
            while self._queue:
                # wait for single-episode requests of other connections; a vector step is a whole step already
                if self.max_wait_time > 0 and not self._full() and all(len(r.ids) == 1 for r in self._queue):
                    with anyio.move_on_after(self.max_wait_time):
                        while not self._full():
                            await self._wake.wait()
                            self._wake = anyio.Event()
                batch = [self._queue.pop(0)]
                rows = len(batch[0].ids)
                limit = self.max_batch_size
                while self._queue and (limit is None or rows + len(self._queue[0].ids) <= limit):
                    rows += len(self._queue[0].ids)
                    batch.append(self._queue.pop(0))
                await self._run(batch)

    def _full(self) -> bool:
        return self.max_batch_size is not None and sum(len(r.ids) for r in self._queue) >= self.max_batch_size

    async def _run(self, batch: list[_Request]) -> None:
        if len(batch) > 1:
            try:
                obs = columns.concat([r.obs for r in batch])
            except Exception:  # requests of different shapes: one at a time
                for r in batch:
                    await self._run([r])
                return
        else:
            obs = batch[0].obs
        try:
            actions, errors = await self._infer(obs, np.concatenate([r.ids for r in batch]))
            start = ok_seen = 0
            for r in batch:
                n = len(r.ids)
                r.errors = {i - start: errors[i] for i in range(start, start + n) if i in errors}
                ok = n - len(r.errors)
                r.actions = actions if len(batch) == 1 else columns.take(actions, range(ok_seen, ok_seen + ok))
                ok_seen += ok
                start += n
        except Exception as exc:
            logger.exception("dispatch failed")
            for r in batch:
                r.actions, r.errors = {}, dict.fromkeys(range(len(r.ids)), exc)
        finally:
            for r in batch:
                r.done.set()

    async def _infer(self, obs: Batch, ids: np.ndarray) -> tuple[Batch, dict[int, Exception]]:
        """Actions of every position that did not fail, in order, and the failures by position."""
        n = len(ids)
        episodes = self._episodes
        chunked = np.flatnonzero(self._chunked[ids]).tolist()
        served: dict[int, Any] = {}
        for i in chunked:
            buffer = self._buffers.get(int(ids[i]))
            if buffer:
                served[i] = buffer.popleft()
        todo = [i for i in range(n) if i not in served] if served else list(range(n))
        errors: dict[int, Exception] = {}
        parts: list[tuple[list[int], Batch]] = []
        batched = type(self).predict_batch is not PredictModelServer.predict_batch
        size = (self.max_batch_size if batched else None) or max(len(todo), 1)  # predict runs row by row anyway
        for start in range(0, len(todo), size):
            group = todo[start : start + size]
            sub = obs if len(group) == n else columns.take(obs, group)
            rows = self._rows(ids[group])
            try:
                if not batched:
                    out, failed = await self._call(self._predict_rows, sub, rows)
                else:
                    out, failed = await self._call(self.predict_batch, sub, rows), {}
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
        if not chunked:
            self._steps[ids[[i for i in range(n) if i not in errors]]] += 1
            return columns.concat([p for _, p in parts]), errors
        # A chunked row returns its chunk's first action and keeps the rest; every row returns "actions" only.
        action: dict[int, Any] = dict(served)
        for positions, out in parts:
            for j, i in enumerate(positions):
                try:
                    a = out["actions"][j]
                    cs = episodes[ids[i]].chunk_size  # type: ignore[union-attr]
                    if cs is not None and np.ndim(a) == 2:
                        a = a[:cs]
                        self._buffers[int(ids[i])] = deque(a[1:])
                        a = a[0]
                    action[i] = a
                except Exception as exc:  # e.g. an empty chunk: this row alone
                    errors[i] = exc
        self._steps[ids[[i for i in range(n) if i not in errors]]] += 1
        return columns.stack([{"actions": action[i]} for i in range(n) if i not in errors]), errors

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


def _float32(actions: Batch) -> Batch:
    """``actions`` as float32 (an array, or each item of a list column), as every model's actions travel."""
    a = actions.get("actions")
    if a is None:
        return actions
    if isinstance(a, list):
        return {**actions, "actions": [np.asarray(x, np.float32) for x in a]}
    return {**actions, "actions": np.asarray(a, np.float32)}
