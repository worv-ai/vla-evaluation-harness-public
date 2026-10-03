"""Benchmark ABCs: the environment interface for evaluation.

* :class:`BenchmarkCommon` — what every benchmark declares: tasks, render backends,
  action/observation specs, metrics, metadata, the real-time hold, cleanup.
* :class:`Benchmark` — async, universal contract for one environment.  Runners depend
  only on this.  Suitable for both simulation and real-robot environments.
* :class:`StepBenchmark` — sync convenience subclass.  Users implement
  ``reset`` / ``step`` / ``make_obs`` and the class auto-bridges to the
  async parent methods.
* :class:`VectorStepBenchmark` — sync, ``num_envs`` environments stepped together
  (GPU-parallel simulators).  The harness runs up to ``num_envs`` episodes at once.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from vla_eval.specs import DimSpec

import numpy as np

from vla_eval.recording import EpisodeRecorder, NullEpisodeRecorder
from vla_eval.types import Action, EpisodeResult, Observation, Task


def repeat_last_hold(last_action: Action | None, action_dim: int) -> Action:
    """Hold for absolute-position control: repeat the last commanded action, or
    a zero action of ``action_dim`` before the first one.

    Convenience for :meth:`Benchmark.get_hold_action`. Only safe for *absolute*
    targets — for delta/velocity control return a fixed null action instead.
    """
    if last_action is not None:
        return last_action
    return {"actions": np.zeros(action_dim, dtype=np.float32)}


@dataclass
class StepResult:
    """Result of a single environment step."""

    obs: Any
    reward: float
    done: bool
    info: dict[str, Any]


# ---------------------------------------------------------------------------
# Benchmark ABCs
# ---------------------------------------------------------------------------


class BenchmarkCommon(ABC):
    """What every benchmark declares, whatever it steps: its tasks, render backends,
    action/observation specs, aggregated metrics, metadata, real-time hold and cleanup."""

    #: Render backends this benchmark can run on (see ``render: gpu|cpu`` / ``--render``).
    #: Default is GPU-only, so a benchmark that has not been verified on CPU fails fast
    #: at startup instead of crashing mid-run.
    render_backends: ClassVar[frozenset[str]] = frozenset({"gpu"})

    @classmethod
    def configure_render(cls, mode: str) -> dict[str, str]:
        """Set process env for *mode* before any simulator import; return the applied env.

        Called once per run by the orchestrator, between resolving the class and
        constructing it — the renderer is bound at the first simulator import and
        cannot be re-bound afterwards.

        Override together with :attr:`render_backends`.  Most MuJoCo adapters just
        ``return configure_mujoco_render(mode)`` (``vla_eval.render``).
        """
        if mode == "gpu":
            return {}
        raise NotImplementedError(
            f"{cls.__name__} does not implement configure_render() for render: {mode}; "
            "declare the backend in render_backends and set the simulator's env explicitly "
            "(see vla_eval.render)."
        )

    # -- abstract: data ---------------------------------------------------

    @abstractmethod
    def get_tasks(self) -> list[Task]:
        """Return the list of tasks this benchmark provides."""

    # -- optional overrides -----------------------------------------------

    def get_action_spec(self) -> dict[str, DimSpec]:
        """Declare the action input format this benchmark's env consumes.

        Returns a ``{component_name: DimSpec}`` dict.  Use ``accepts`` on
        DimSpec to declare convertible formats (e.g. benchmark converts
        axis-angle to euler internally).

        Override in every subclass — the default raises ``NotImplementedError``.
        """
        raise NotImplementedError(f"{type(self).__name__} must override get_action_spec()")

    def get_observation_spec(self) -> dict[str, DimSpec]:
        """Declare the observation output format this benchmark produces.

        Returns a ``{component_name: DimSpec}`` dict describing what
        ``get_observation()`` / ``make_obs()`` sends to the model server.

        Override in every subclass — the default raises ``NotImplementedError``.
        """
        raise NotImplementedError(f"{type(self).__name__} must override get_observation_spec()")

    def get_metric_keys(self) -> dict[str, str]:
        """Declare which metrics from ``get_result()`` to aggregate.

        Returns ``{field: aggregation}`` where aggregation is one of
        ``"mean"``, ``"sum"``, ``"max"``, ``"min"``.  Fields are stored
        under ``episode["metrics"]`` in the result JSON and aggregated
        into ``TaskResult`` / ``BenchmarkResult`` as ``{agg}_{field}``.

        The default declares ``success`` with ``"mean"`` (= success rate).
        Override to add benchmark-specific metrics.
        """
        return {"success": "mean"}

    def get_metadata(self) -> dict[str, Any]:
        """Return benchmark defaults and metadata. Optional override."""
        return {}

    def get_hold_action(self, last_action: Action | None) -> Action:
        """Return the action to command on a stale real-time tick.

        Real-time runners call this when the model has not produced a fresh
        action this tick (and before the first one, with ``last_action=None``).
        The safe hold is embodiment knowledge with no universal default:

        * absolute-position control → ``return last_action`` (hold the target);
          use :func:`repeat_last_hold`.
        * delta / velocity control → return a fixed null action (stay put);
          repeating a delta would keep moving.

        There is deliberately **no default**: a real-time run whose benchmark
        has not declared a hold fails fast here rather than driving a robot with
        an unsafe reuse. Sync-mode benchmarks never call this, so they need not
        implement it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is running in real-time mode but does not implement "
            "get_hold_action(); declare the embodiment's safe do-nothing action explicitly "
            "(see Benchmark.get_hold_action / repeat_last_hold)."
        )

    def cleanup(self) -> None:
        """Release benchmark resources (environments, renderers, etc.). Optional override."""

    def render(self) -> np.ndarray | None:
        """Render current env state as image. Optional override."""
        return None


class Benchmark(BenchmarkCommon):
    """Universal async benchmark contract.

    Runners call these methods — they never touch sync helpers directly.

    Command methods (mutate state):
        - ``start_episode(task)`` → None (stores env internally).
        - ``apply_action(action)`` → None (actuate only).

    Query methods (read state):
        - ``get_observation()`` → observation dict for the model server.
        - ``is_done()`` → bool.
        - ``get_time()`` → environment time in seconds.

    Data methods:
        - ``get_tasks()`` → list of task dicts.
        - ``get_result()`` → episode result dict.
        - ``get_metadata()`` → benchmark defaults / metadata.
    """

    # -- abstract: commands -----------------------------------------------

    @abstractmethod
    async def start_episode(self, task: Task, recorder: EpisodeRecorder | None = None) -> None:
        """Initialise an episode. ``recorder`` is always non-None (Null when recording is off),
        so subclasses can call ``recorder.record_*`` unconditionally."""

    @abstractmethod
    async def apply_action(self, action: Action) -> None:
        """Execute *action* in the environment (fire-and-forget)."""

    # -- abstract: queries ------------------------------------------------

    @abstractmethod
    async def get_observation(self) -> Observation:
        """Read the current observation from the environment."""

    @abstractmethod
    async def is_done(self) -> bool:
        """Return ``True`` when the episode should end."""

    @abstractmethod
    async def get_time(self) -> float:
        """Return the current environment time (seconds since episode start)."""

    @abstractmethod
    async def get_result(self) -> EpisodeResult:
        """Return the episode result (at least ``{"success": bool}``)."""


# ---------------------------------------------------------------------------
# Step-based convenience subclass
# ---------------------------------------------------------------------------


class StepBenchmark(Benchmark, ABC):
    """Sync step-based benchmark (simulations).

    Subclasses implement:
        - ``reset(task)`` → initial_raw_obs (store env on self)
        - ``step(action)`` → ``StepResult``
        - ``make_obs(raw_obs, task)`` → observation dict for model server
        - ``check_done(step_result)`` → bool  (default: ``step_result.done``)
        - ``get_step_result(step_result)`` → EpisodeResult (abstract)

    The class auto-bridges these to the async parent API via internal state.
    """

    def __init__(self) -> None:
        super().__init__()
        self._last_result: StepResult = StepResult(obs=None, reward=0.0, done=False, info={})
        self._task: Task = {}
        self._t0: float = 0.0

    # -- abstract: user implements ----------------------------------------

    @abstractmethod
    def reset(self, task: Task) -> Any:
        """Reset environment for *task*. Returns initial raw observation (store env on self)."""

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        """Apply action to environment and return :class:`StepResult`."""

    @abstractmethod
    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        """Convert raw observation to the model server's :class:`Observation` format."""

    @abstractmethod
    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        """Extract episode result from the final :class:`StepResult`."""

    def check_done(self, step_result: StepResult) -> bool:
        """Check if episode should terminate. Default: ``step_result.done``."""
        return step_result.done

    # -- async bridge (auto-provided) -------------------------------------
    # NOTE: reset(), step(), make_obs() are sync and may block the event loop
    # when MuJoCo physics or image rendering takes non-trivial time.
    # Offloading them via anyio.to_thread.run_sync() could improve
    # concurrency under high shard counts.  However, the default anyio
    # thread-pool limiter only has 40 tokens — under 50+ concurrent shards
    # a dedicated CapacityLimiter is needed to avoid starvation
    # (see _DECODE_LIMITER in serve.py for an example).

    async def start_episode(self, task: Task, recorder: EpisodeRecorder | None = None) -> None:
        self._t0 = time.monotonic()
        self._task = task
        self._recorder: EpisodeRecorder = recorder or NullEpisodeRecorder()
        raw_obs = self.reset(task)
        self._last_result = StepResult(obs=raw_obs, reward=0.0, done=False, info={})

    async def apply_action(self, action: Action) -> None:
        self._last_result = self.step(action)

    async def get_observation(self) -> Observation:
        return self.make_obs(self._last_result.obs, self._task)

    async def is_done(self) -> bool:
        return self.check_done(self._last_result)

    async def get_time(self) -> float:
        return time.monotonic() - self._t0

    async def get_result(self) -> EpisodeResult:
        return self.get_step_result(self._last_result)


# ---------------------------------------------------------------------------
# Vectorized (several environments stepped together)
# ---------------------------------------------------------------------------


class VectorStepBenchmark(BenchmarkCommon):
    """Sync benchmark over ``num_envs`` environments stepped together.

    For simulators that step many environments per call (MuJoCo Warp / mjlab, ManiSkill3
    on the GPU, Isaac Lab): one process holds them all, and the harness runs up to
    ``num_envs`` episodes at once.  Each running episode has its own model-server
    session (its own connection), and their observations go out concurrently, so a
    batching model server (``PredictModelServer``) infers them in one batch.  Sync mode
    only: the environments step when every running episode has its action.

    Slots are the environment indices ``0 .. num_envs - 1``.  Subclasses implement:

        - ``reset(slots, tasks, recorders)`` → the first raw observation of each slot:
          start the episode ``tasks[i]`` in environment ``slots[i]``, recording through
          ``recorders[i]`` (never None; Null when recording is off).
        - ``step(actions)`` → ``{slot: StepResult}``: advance every environment one step.
          ``actions`` maps each running slot to its action and the result holds those
          slots.  A slot missing from ``actions`` is idle: keep it inert (hold its targets,
          or ignore it); its state does not matter until it is reset.
        - ``make_obs(raw_obs, slot, task)`` → observation dict for the model server.
        - ``get_step_result(slot, step_result)`` → EpisodeResult (at least ``success``).
        - ``check_done(step_result)`` → bool (default: ``step_result.done``).

    ``partial_reset``: ``True`` when ``reset`` may start some slots while others are
    mid-episode (per-environment resets), so a finished slot takes the next episode at
    once.  ``False`` (the default) when the environments start together: the harness
    resets only when no slot is running and runs the episodes in waves of up to
    ``num_envs``; a slot that finishes early idles until its wave ends.

    The harness counts steps and ``max_steps`` per episode, and measures each episode's
    ``elapsed_sec`` as wall time from its reset.
    """

    partial_reset: ClassVar[bool] = False

    def __init__(self, num_envs: int = 1) -> None:
        super().__init__()
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self.num_envs = num_envs

    @abstractmethod
    def reset(self, slots: list[int], tasks: list[Task], recorders: list[EpisodeRecorder]) -> list[Any]:
        """Start ``tasks[i]`` in environment ``slots[i]``; return the first raw observation of each."""

    @abstractmethod
    def step(self, actions: dict[int, Action]) -> dict[int, StepResult]:
        """Advance every environment one step; return a result for each slot in ``actions``."""

    @abstractmethod
    def make_obs(self, raw_obs: Any, slot: int, task: Task) -> Observation:
        """Convert slot ``slot``'s raw observation to the model server's :class:`Observation` format."""

    @abstractmethod
    def get_step_result(self, slot: int, step_result: StepResult) -> EpisodeResult:
        """Extract the episode result of slot ``slot`` from its final :class:`StepResult`."""

    def check_done(self, step_result: StepResult) -> bool:
        """Check if a slot's episode should terminate. Default: ``step_result.done``."""
        return step_result.done
