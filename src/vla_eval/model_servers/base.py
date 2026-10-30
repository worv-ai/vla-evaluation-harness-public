"""Model-server contract: every running episode is a row; inference and episode start/end are batched.
``PredictModelServer`` (predict.py) implements the framework side."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import numpy.typing as npt

from vla_eval.specs import DimSpec
from vla_eval.types import Batch, EpisodeResult

if TYPE_CHECKING:
    from anyio.abc import TaskGroup


@dataclass(eq=False)
class Episode:
    """One running episode.  ``session_id``/``episode_id``/``recording_db_path`` open a ``StepRecorder`` on
    its recording; ``on_episode_start`` may override ``chunk_size`` (e.g. per task suite)."""

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
    """The rows of one call.  ``ids``: stable while an episode runs, reused after, below the peak episode
    count, so per-episode state fits in arrays indexed by them.  ``step``: the episode's prior observations."""

    def __init__(
        self, ids: npt.NDArray[np.int64], step: npt.NDArray[np.int64], table: Sequence[Episode | None]
    ) -> None:
        self.ids = ids
        self.step = step
        self._table = table

    @property
    def is_first(self) -> npt.NDArray[np.bool_]:
        return self.step == 0

    @property
    def episodes(self) -> list[Episode]:
        return [self[i].episode for i in range(len(self))]

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self) -> Iterator[Row]:
        return (self[i] for i in range(len(self)))

    def __getitem__(self, i: int) -> Row:
        rid = int(self.ids[i])
        episode = self._table[rid]
        assert episode is not None
        return Row(rid, int(self.step[i]), bool(self.step[i] == 0), episode)


class ModelServer(ABC):
    """What ``serve`` drives.  Subclass ``PredictModelServer``, which implements the abstract methods.
    Load weights and warm up in ``__init__``: connections are accepted as soon as it returns."""

    continuous_inference: bool = False
    """Live mode: ``serve`` infers each connection's latest observation in a loop and pushes its action."""

    async def on_serve_start(self, tg: TaskGroup) -> None:
        """Called by ``serve_async`` before accepting connections; *tg* lives as long as the server.
        Overrides call ``super()`` first."""

    def on_episode_start(self, rows: Rows) -> dict[int, Exception] | None:
        """Reset the state of ``rows`` (``episode.config``: the ``EPISODE_START`` payload).  Blocking.
        Return ``{position: exception}`` for starts that failed, or raise to fail them all."""
        return None

    def on_episode_end(self, rows: Rows, results: list[EpisodeResult]) -> None:
        """Episodes in ``rows`` ended (``results[i]`` from the harness, ``{}`` when it went away).  Blocking."""

    def pick_action(self, actions: np.ndarray, obs_time: float) -> np.ndarray:
        """Continuous inference: the action to send from ``actions`` for an observation received at ``obs_time``."""
        return actions if actions.ndim == 1 else actions[0]

    @abstractmethod
    async def start_episodes(self, starts: list[tuple[dict[str, Any], str]]) -> list[int | Exception]:
        """Start an episode per ``(EPISODE_START payload, default session id)``; its row id or why it failed."""

    @abstractmethod
    async def end_episodes(self, ids: Sequence[int], results: list[EpisodeResult]) -> None:
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
