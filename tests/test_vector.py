"""VectorStepBenchmark: several environments per process, one model-server session per running episode."""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Any, ClassVar
from unittest.mock import patch

import numpy as np
import pytest
import websockets.exceptions

from vla_eval.benchmarks.base import BenchmarkCommon, StepResult, VectorStepBenchmark
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.orchestrator import Orchestrator, UnhealthyShardError
from vla_eval.recording import RecordingStore

from tests.conftest import start_server, stop_server


class StubVectorBenchmark(VectorStepBenchmark):
    """Task ``i``'s episode lasts ``i + 2`` steps, plus ``episode_idx % (jitter + 1)``.  Slot ``s`` observes
    ``value = s + 1``; the echo server sends it back as the action, so ``misrouted`` lists every action that
    reached the wrong slot."""

    last: ClassVar[StubVectorBenchmark]  # the instance the orchestrator built

    def __init__(self, num_envs: int = 3, num_tasks: int = 2, jitter: int = 0, **kwargs: Any) -> None:
        super().__init__(num_envs)
        self.num_tasks = num_tasks
        self.jitter = jitter
        self.t = [0] * num_envs
        self.length = [0] * num_envs
        self.running: set[int] = set()
        self.resets: list[tuple[list[int], list[int]]] = []  # (slots reset, slots running at the time)
        self.steps: list[list[int]] = []
        self.misrouted: list[tuple[int, float]] = []
        StubVectorBenchmark.last = self

    def get_tasks(self) -> list[dict[str, Any]]:
        return [{"name": f"task_{i}", "length": i + 2} for i in range(self.num_tasks)]

    def reset(self, slots, tasks, recorders):
        self.resets.append((list(slots), sorted(self.running)))
        for slot, task in zip(slots, tasks):
            self.t[slot], self.length[slot] = 0, task["length"] + task["episode_idx"] % (self.jitter + 1)
            self.running.add(slot)
        return list(slots)

    def step(self, actions):
        self.steps.append(sorted(actions))
        out = {}
        for slot, action in actions.items():
            if float(action["actions"][0]) != slot + 1:
                self.misrouted.append((slot, float(action["actions"][0])))
            self.t[slot] += 1
            done = self.t[slot] >= self.length[slot]
            if done:
                self.running.discard(slot)
            out[slot] = StepResult(obs=slot, reward=float(done), done=done, info={})
        return out

    def make_obs(self, raw_obs, slot, task):
        return {"value": float(slot + 1), "task_description": task["name"]}

    def get_step_result(self, slot, step_result):
        return {"success": step_result.done}

    def get_metadata(self) -> dict[str, Any]:
        return {"max_steps": 50}


class PartialResetStub(StubVectorBenchmark):
    partial_reset = True


class NeverDoneStub(StubVectorBenchmark):
    def step(self, actions):
        return {slot: StepResult(obs=slot, reward=0.0, done=False, info={}) for slot in actions}


class BrokenResetStub(StubVectorBenchmark):
    def reset(self, slots, tasks, recorders):
        raise RuntimeError("Failed to find a supported physical device")


class SlotTwoModelErrorStub(StubVectorBenchmark):
    """Slot 1 sends a value the error server rejects."""

    def make_obs(self, raw_obs, slot, task):
        return {"value": -1.0 if slot == 1 else float(slot + 1), "task_description": task["name"]}


class SessionLogServer(PredictModelServer):
    """Echoes ``value`` and logs which session sent it; rejects negative values."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple[str, float]] = []

    def predict(self, obs: dict[str, Any], ctx: SessionContext) -> dict[str, Any]:
        if obs["value"] < 0:
            raise ValueError("bad observation")
        self.seen.append((ctx.session_id, obs["value"]))
        return {"actions": obs["value"] * np.ones(7, dtype=np.float32)}


class BatchLogServer(PredictModelServer):
    def __init__(self) -> None:
        super().__init__(max_batch_size=8, max_wait_time=0.2)
        self.sizes: list[int] = []

    def predict_batch(self, obs_batch, ctx_batch):
        self.sizes.append(len(obs_batch))
        return [{"actions": o["value"] * np.ones(7, dtype=np.float32)} for o in obs_batch]


def _config(url: str, tmp_path, name: str = "vec", episodes: int = 4, **params: Any) -> dict[str, Any]:
    entry = {
        "benchmark": "tests.test_vector:StubVectorBenchmark",
        "name": name,
        "episodes_per_task": episodes,
        "params": {"num_envs": 3, "num_tasks": 2, **params},
    }
    return {"server": {"url": url}, "output_dir": str(tmp_path), "benchmarks": [entry]}


async def _run(cls: type, config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    with patch("vla_eval.orchestrator.resolve_import_string", return_value=cls):
        (result,) = await Orchestrator(config, **kwargs).run()
    return result


def _episodes(result: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [(t["task"], ep) for t in result["tasks"] for ep in t["episodes"]]


@pytest.fixture
async def log_server(free_port: int):
    server = SessionLogServer()
    task = await start_server(server, free_port)
    yield f"ws://127.0.0.1:{free_port}", server
    await stop_server(task)


@pytest.mark.anyio
async def test_waves_run_every_item_and_reset_only_when_idle(echo_server, tmp_path):
    result = await _run(StubVectorBenchmark, _config(echo_server, tmp_path), no_save=True)
    bench = StubVectorBenchmark.last
    episodes = _episodes(result)
    assert sorted((t, ep["episode_id"]) for t, ep in episodes) == [
        (f"task_{t}", e) for t in range(2) for e in range(4)
    ]
    assert result["mean_success"] == pytest.approx(1.0) and "partial" not in result
    assert {t: {ep["steps"] for tt, ep in episodes if tt == t} for t in ("task_0", "task_1")} == {
        "task_0": {2},
        "task_1": {3},
    }
    assert [len(slots) for slots, _ in bench.resets] == [3, 3, 2]
    assert all(running == [] for _, running in bench.resets)
    assert max(len(s) for s in bench.steps) == 3 and bench.misrouted == []


@pytest.mark.anyio
async def test_partial_reset_refills_a_slot_as_soon_as_it_frees(echo_server, tmp_path):
    config = _config(echo_server, tmp_path, jitter=3)
    await _run(StubVectorBenchmark, config, no_save=True)
    waves = StubVectorBenchmark.last
    result = await _run(PartialResetStub, config, no_save=True)
    bench = StubVectorBenchmark.last
    assert len(_episodes(result)) == 8 and result["mean_success"] == pytest.approx(1.0)
    assert any(running for _, running in bench.resets)  # a slot started while others ran
    assert bench.misrouted == []
    assert len(bench.steps) < len(waves.steps)  # 15 env steps in waves of three, fewer when slots refill


@pytest.mark.anyio
@pytest.mark.parametrize("no_save", [True, False])
async def test_each_running_episode_has_its_own_session(log_server, tmp_path, no_save):
    url, server = log_server
    await _run(StubVectorBenchmark, _config(url, tmp_path, episodes=6), no_save=no_save, eval_id=f"ev-s{no_save}")
    by_session: dict[str, set[float]] = {}
    for sid, value in server.seen:
        by_session.setdefault(sid, set()).add(value)
    assert len(by_session) == 3 and all(len(v) == 1 for v in by_session.values())
    if not no_save:  # the recording sid is the server's session id, one per slot
        with sqlite3.connect(tmp_path / f"recording-ev-s{no_save}.sqlite") as db:
            sids = {row[0] for row in db.execute("SELECT sid FROM episode_results")}
        assert sids == set(by_session)


@pytest.mark.anyio
async def test_concurrent_slots_reach_the_server_as_one_batch(free_port, tmp_path):
    server = BatchLogServer()
    task = await start_server(server, free_port)
    try:
        result = await _run(StubVectorBenchmark, _config(f"ws://127.0.0.1:{free_port}", tmp_path), no_save=True)
    finally:
        await stop_server(task)
    assert result["mean_success"] == pytest.approx(1.0)
    assert max(server.sizes) == 3


@pytest.mark.anyio
async def test_max_steps_ends_each_episode(echo_server, tmp_path):
    config = _config(echo_server, tmp_path, episodes=2)
    config["benchmarks"][0]["max_steps"] = 4
    result = await _run(NeverDoneStub, config, no_save=True)
    episodes = _episodes(result)
    assert len(episodes) == 4 and {ep["steps"] for _, ep in episodes} == {4}
    assert result["mean_success"] == pytest.approx(0.0)


@pytest.mark.anyio
async def test_a_model_error_fails_only_its_episode(log_server, tmp_path):
    url, _ = log_server
    result = await _run(SlotTwoModelErrorStub, _config(url, tmp_path), no_save=True)
    episodes = _episodes(result)
    failed = [ep for _, ep in episodes if ep.get("failure_reason")]
    assert len(episodes) == 8 and len(failed) == 3  # slot 1 runs one episode per wave
    assert {ep["failure_reason"] for ep in failed} == {"model_act"}
    assert all(ep["metrics"]["success"] for _, ep in episodes if not ep.get("failure_reason"))


@pytest.mark.anyio
async def test_a_reset_error_fails_the_episodes_it_started(echo_server, tmp_path):
    result = await _run(BrokenResetStub, _config(echo_server, tmp_path), no_save=True)
    episodes = _episodes(result)
    assert len(episodes) == 8 and {ep["failure_reason"] for _, ep in episodes} == {"env_start"}
    assert "partial" not in result


@pytest.mark.anyio
async def test_live_mode_is_rejected(echo_server, tmp_path):
    config = _config(echo_server, tmp_path)
    config["benchmarks"][0]["mode"] = "live"
    with pytest.raises(ValueError, match="sync mode only"):
        await _run(StubVectorBenchmark, config, no_save=True)


@pytest.mark.anyio
async def test_shards_share_the_queue_without_double_claims(echo_server, tmp_path):
    config = _config(echo_server, tmp_path, episodes=5)
    with patch("vla_eval.orchestrator.resolve_import_string", return_value=StubVectorBenchmark):
        runs = [Orchestrator(config, shard_id=i, num_shards=2, eval_id="ev-vq", no_save=False) for i in range(2)]
        results = await asyncio.gather(*(orch.run() for orch in runs))
    keys = [(t, ep["episode_id"]) for (r,) in results for t, ep in _episodes(r)]
    assert sorted(keys) == [(f"task_{t}", e) for t in range(2) for e in range(5)]
    store = RecordingStore(tmp_path / "recording-ev-vq.sqlite")
    assert store.queue_progress("ev-vq-vec") == (10, 10)
    store.close()


@pytest.mark.anyio
async def test_unhealthy_shard_requeues_failed_and_in_flight_items(echo_server, tmp_path):
    from vla_eval.results.export import export_db

    config = _config(echo_server, tmp_path, name="u", episodes=3, num_envs=2)
    with pytest.raises(UnhealthyShardError):
        await _run(BrokenResetStub, config, shard_id=0, num_shards=2, eval_id="ev-vu", requeue_unhealthy=True)
    healthy = await _run(StubVectorBenchmark, config, shard_id=1, num_shards=2, eval_id="ev-vu")
    assert len(_episodes(healthy)) == 6
    (aggregate,) = export_db(tmp_path / "recording-ev-vu.sqlite", tmp_path)
    assert aggregate["num_episodes_total"] == 6 and aggregate["num_errors"] == 0


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("act_error", "first_reason"),
    [(websockets.exceptions.ConnectionClosed(None, None), "connection_closed"), (TimeoutError(), "timeout")],
)
@pytest.mark.parametrize(
    "reconnect_error",
    [ConnectionError("unreachable after retries"), TimeoutError("HELLO"), RuntimeError("Expected HELLO reply")],
)
async def test_a_failed_reconnect_ends_every_running_episode_and_returns_partial(
    tmp_path, act_error, first_reason, reconnect_error
):
    calls = 0

    class FlakyConnection:
        def __init__(self, url, **kwargs):
            self.server_info: dict[str, Any] = {}

        async def connect(self, **kwargs):
            pass

        async def close(self):
            pass

        async def start_episode(self, cfg):
            pass

        async def end_episode(self, result):
            pass

        async def act(self, obs):
            nonlocal calls
            calls += 1
            if calls == 8:  # first step of the second wave
                raise act_error
            return {"actions": obs["value"] * np.ones(7, dtype=np.float32)}

        async def reconnect(self):
            raise reconnect_error

    config = _config("ws://fake:9999", tmp_path, name="a")
    with patch("vla_eval.orchestrator.Connection", FlakyConnection):
        result = await _run(StubVectorBenchmark, config, shard_id=0, num_shards=1, eval_id="ev-a")
    episodes = _episodes(result)
    assert result["partial"] is True
    store = RecordingStore(tmp_path / "recording-ev-a.sqlite")
    assert store.queue_progress("ev-a-a") == (3, 8)  # the aborted wave stays for a rerun
    store.close()
    # wave 1 finished (3); wave 2's three episodes ended with the server: the one that failed, two unreachable
    assert len(episodes) == 6
    reasons = sorted(ep.get("failure_reason") or "" for _, ep in episodes)
    assert reasons == sorted(["", "", "", first_reason, "server_unreachable", "server_unreachable"])


class StepTimeoutOnceStub(StubVectorBenchmark):
    def step(self, actions):
        if not self.steps:
            self.steps.append(sorted(actions))
            raise TimeoutError("simulator stalled")
        return super().step(actions)


@pytest.mark.anyio
async def test_an_environment_timeout_fails_its_wave_and_the_run_goes_on(echo_server, tmp_path):
    result = await _run(StepTimeoutOnceStub, _config(echo_server, tmp_path), no_save=True)
    episodes = _episodes(result)
    assert len(episodes) == 8 and "partial" not in result
    assert sorted(ep.get("failure_reason") or "" for _, ep in episodes) == [""] * 5 + ["timeout"] * 3


@pytest.mark.anyio
async def test_a_failed_recording_leaves_its_item_reclaimable(echo_server, tmp_path):
    from vla_eval.recording import EpisodeRecorder, RecordingError

    config = _config(echo_server, tmp_path, name="rf", episodes=2)
    with patch.object(EpisodeRecorder, "close", side_effect=RecordingError("disk full")):
        with pytest.raises(RecordingError):
            await _run(StubVectorBenchmark, config, shard_id=0, num_shards=1, eval_id="ev-rf")
    store = RecordingStore(tmp_path / "recording-ev-rf.sqlite")
    assert store.queue_progress("ev-rf-rf") == (0, 4)
    store.close()


def test_claim_skips_items_in_flight(tmp_path):
    store = RecordingStore(tmp_path / "q.sqlite")
    store.seed_queue("e", [0, 0, 1, 1])
    first = store.claim("e", 0, None, 1)
    assert first is not None
    assert store.claim("e", 0, None, 1) == first  # a rerun picks its unfinished item up again
    second = store.claim("e", 0, 0, 1, exclude={first})
    assert second is not None and second != first
    store.close()


def test_vector_benchmarks_validate_as_benchmarks():
    assert issubclass(StubVectorBenchmark, BenchmarkCommon)
    with pytest.raises(ValueError):
        StubVectorBenchmark(num_envs=0)
