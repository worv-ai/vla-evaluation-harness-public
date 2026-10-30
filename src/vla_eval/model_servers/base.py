"""Model-server contract: every running episode is a row; inference and the episode lifecycle are batched.

A single-episode benchmark is a batch of one row; a vector benchmark's step, or requests coalesced across
connections, are batches of many.  ``PredictModelServer`` (predict.py) implements the framework side.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from vla_eval.columns import Batch
from vla_eval.specs import DimSpec

if TYPE_CHECKING:
    from anyio.abc import TaskGroup


@dataclass(eq=False)
class Episode:
    """One running episode, as its ``EPISODE_START`` announced it.

    ``session_id`` / ``episode_id`` are the harness's recording ``sid`` / ``eid`` when it records (open a
    :class:`vla_eval.recording.StepRecorder` with them and ``recording_db_path`` to add rows to its episode),
    else generated.  ``chunk_size``: this episode's action chunk length, the server's ``chunk_size`` unless
    ``on_episode_start`` sets it (e.g. per task suite).
    """

    config: dict[str, Any]
    episode_id: str
    session_id: str
    eval_id: str = ""
    recording_db_path: str = ""
    chunk_size: int | None = None


class Row(NamedTuple):
    id: int
    step: int
    is_first: bool
    episode: Episode


class Rows:
    """The rows of one call.

    ``ids[i]``: row ``i``'s id, stable while its episode runs and reused once it ended (below the most
    episodes the server has held at once), so a policy can keep per-episode state in arrays indexed by it.
    ``step[i]``: observations its episode had before this one.  ``episodes[i]``: its :class:`Episode`.
    """

    def __init__(self, ids: np.ndarray, step: np.ndarray, table: Sequence[Episode | None]) -> None:
        self.ids = ids
        self.step = step
        self._table = table

    @property
    def is_first(self) -> np.ndarray:
        return self.step == 0

    @property
    def episodes(self) -> list[Episode]:
        return [self._table[i] for i in self.ids.tolist()]  # type: ignore[misc]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> Row:
        rid = int(self.ids[i])
        episode = self._table[rid]
        assert episode is not None
        return Row(rid, int(self.step[i]), bool(self.step[i] == 0), episode)


class ModelServer(ABC):
    """What ``serve`` drives: episodes start and end in batches, observations come as column batches.

    Subclass :class:`~vla_eval.model_servers.predict.PredictModelServer`; it implements :meth:`start_episodes`,
    :meth:`end_episodes` and :meth:`act` and calls the hooks below.  Load weights and finish warmup in
    ``__init__``: the server accepts connections as soon as it returns.
    """

    async def on_serve_start(self, tg: TaskGroup) -> None:
        """Called by ``serve_async`` before accepting connections; *tg* lives as long as the server.
        Overrides call ``super()`` first."""

    def on_episode_start(self, rows: Rows) -> dict[int, Exception] | None:
        """Episodes start in ``rows`` (each row's ``episode.config`` is its ``EPISODE_START`` payload; reset that
        row's state).  Return the exception of each position whose start failed, or raise to fail them all.
        Blocking; runs on the inference thread."""
        return None

    def on_episode_end(self, rows: Rows, results: list[dict[str, Any]]) -> None:
        """Episodes in ``rows`` ended (``results[i]`` from the harness, ``{}`` when it went away).  Blocking."""

    @abstractmethod
    async def start_episodes(self, starts: list[tuple[dict[str, Any], str]]) -> list[int | Exception]:
        """Start an episode per ``(EPISODE_START payload, default session id)``; its row id or why it failed."""

    @abstractmethod
    async def end_episodes(self, ids: Sequence[int], results: list[dict[str, Any]]) -> None:
        """End these rows' episodes and free the rows."""

    @abstractmethod
    async def act(self, obs: Batch, ids: Sequence[int]) -> tuple[Batch, dict[int, Exception]]:
        """Actions for ``obs`` (row ``i`` of row ``ids[i]``): the columns of the positions that did not fail,
        in order, and the exception of each position that did."""

    def get_action_spec(self) -> dict[str, DimSpec]:
        """Declare the action output format of this model server.

        Returns a ``{component_name: DimSpec}`` dict describing what this
        server produces.  The orchestrator compares this against the
        benchmark's action spec and warns on mismatches.

        Override in every subclass — the default raises ``NotImplementedError``.
        """
        raise NotImplementedError(f"{type(self).__name__} must override get_action_spec()")

    def get_observation_spec(self) -> dict[str, DimSpec]:
        """Declare the observation input format this model server expects.

        Returns a ``{component_name: DimSpec}`` dict describing what this
        server needs from the benchmark.  The orchestrator warns when the
        benchmark doesn't provide a declared component.

        Override in every subclass — the default raises ``NotImplementedError``.
        """
        raise NotImplementedError(f"{type(self).__name__} must override get_observation_spec()")

    def get_observation_params(self) -> dict[str, Any]:
        """Declare observation requirements for this model.

        Returned params are sent in the HELLO response and auto-merged into
        benchmark params by the orchestrator. This lets the model server
        tell the benchmark what observation data it needs (e.g. wrist images,
        proprioceptive state) without requiring manual ``--param`` flags.

        Override in subclasses to auto-detect from model config.
        """
        return {}
