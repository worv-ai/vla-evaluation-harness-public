"""EpisodeRunner ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vla_eval.benchmarks.base import Benchmark
from vla_eval.recording import EpisodeRecorder
from vla_eval.types import EpisodeResult, Task


class EpisodeRunner(ABC):
    """Abstract base class for episode execution strategies."""

    @abstractmethod
    async def run_episode(
        self,
        benchmark: Benchmark,
        task: Task,
        conn: Any,  # Connection
        *,
        max_steps: int | None = None,
        recorder: EpisodeRecorder | None = None,
    ) -> EpisodeResult:
        """Run a single episode and return the result.

        ``recorder`` (when supplied) is forwarded to
        ``benchmark.start_episode`` so video / step capture works. The
        runner additionally bundles ``{sid, eid, eval_id}`` into the WS
        ``EPISODE_START`` payload so the model server can tag any external
        step pushes (e.g. reflex-train) with the same bucket key.
        """
