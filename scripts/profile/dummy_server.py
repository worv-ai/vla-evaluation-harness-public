"""Instant model server for benchmark-side profiling: every step costs ~0 s of inference.

    python scripts/profile/dummy_server.py --action-dim 7 [--hold-key joint_state] [--chunk-size 1]
        [--latency-ms 0] [--obs-params '{"send_wrist_image": true}'] [--port 8000]

Returns zeros, or the observation's ``--hold-key`` vector (absolute-position benchmarks such as RoboTwin stay put).
Logs the cumulative observation rate every ``--log-every`` seconds so a run's steps/s can be read from the log.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.serve import serve


class DummyServer(PredictModelServer):
    def __init__(self, a: argparse.Namespace) -> None:
        super().__init__(chunk_size=a.chunk_size, max_batch_size=a.max_batch_size, max_wait_time=0.005)
        self.a = a
        self.obs_params = json.loads(a.obs_params)
        self.steps = self.batches = 0
        self.t0: float | None = None
        self.t_log = 0.0

    def get_observation_params(self) -> dict[str, Any]:
        return self.obs_params

    def _action(self, obs: dict[str, Any]) -> np.ndarray:
        act = np.zeros(self.a.action_dim, np.float32)
        if self.a.hold_key and self.a.hold_key in obs:
            held = np.asarray(obs[self.a.hold_key], np.float32).reshape(-1)[: self.a.action_dim]
            act[: held.size] = held
        return act if not self.chunk_size or self.chunk_size <= 1 else np.tile(act, (self.chunk_size, 1))

    def predict_batch(self, obs_batch: list[dict[str, Any]], ctx_batch: list[Any]) -> list[dict[str, Any]]:
        now = time.monotonic()
        if self.t0 is None:
            self.t0, self.t_log = now, now
        if self.a.latency_ms:
            time.sleep(self.a.latency_ms / 1000)
        self.steps += len(obs_batch)
        self.batches += 1
        if now - self.t_log >= self.a.log_every:
            self.t_log = now
            print(
                f"profile steps={self.steps} batches={self.batches} elapsed={now - self.t0:.1f}s "
                f"rate={self.steps / (now - self.t0):.2f}/s",
                flush=True,
            )
        return [{"actions": self._action(obs)} for obs in obs_batch]

    def predict(self, obs: dict[str, Any], ctx: Any) -> dict[str, Any]:
        return self.predict_batch([obs], [ctx])[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--action-dim", type=int, required=True)
    p.add_argument("--hold-key", default=None, help="observation key whose vector is echoed back as the action")
    p.add_argument("--chunk-size", type=int, default=1)
    p.add_argument("--max-batch-size", type=int, default=64)
    p.add_argument("--latency-ms", type=float, default=0.0, help="sleep per batched forward, to mimic a model")
    p.add_argument("--obs-params", default="{}", help="JSON returned by get_observation_params()")
    p.add_argument("--log-every", type=float, default=30.0)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    serve(DummyServer(a), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
