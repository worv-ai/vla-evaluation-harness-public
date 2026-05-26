"""Tests for the ``report_to`` tracker integration.

Covers the dispatch / availability / robustness pieces directly, and runs an
end-to-end wandb integration test in offline mode (``WANDB_MODE=offline``) when
the ``wandb`` package is installed. The offline test asserts the on-disk run
artifact exists and that the harness-injected ``eval_id`` flowed through —
the same convergence handle the orchestrator and ``vla-eval merge`` rely on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from vla_eval.tracking import (
    INTEGRATION_TO_TRACKER,
    Tracker,
    _scalar_metrics,
    _scalar_summary,
    call_each,
    get_reporting_trackers,
    is_trackio_available,
    is_wandb_available,
)


# ---------- get_reporting_trackers dispatch ----------


def test_get_reporting_trackers_none_or_empty_returns_empty() -> None:
    assert get_reporting_trackers(None) == []
    assert get_reporting_trackers("none") == []
    assert get_reporting_trackers([]) == []


def test_get_reporting_trackers_unknown_raises_with_supported_list() -> None:
    with pytest.raises(ValueError, match="not a supported tracker"):
        get_reporting_trackers("not_a_real_backend")
    with pytest.raises(ValueError, match="not a supported tracker"):
        get_reporting_trackers(["wandb", "not_a_real_backend"])


def test_get_reporting_trackers_all_picks_only_installed_backends() -> None:
    # Force both availability checks to False — "all" should give an empty list,
    # not error, even when no backend is installed.
    with mock.patch("vla_eval.tracking._IS_AVAILABLE", {"wandb": lambda: False, "trackio": lambda: False}):
        assert get_reporting_trackers("all") == []


# ---------- availability helpers ----------


def test_is_wandb_available_false_when_spec_missing() -> None:
    with mock.patch("importlib.util.find_spec", return_value=None):
        assert is_wandb_available() is False


def test_is_trackio_available_false_when_spec_missing() -> None:
    with mock.patch("importlib.util.find_spec", return_value=None):
        assert is_trackio_available() is False


# ---------- base Tracker is a complete no-op ----------


def test_base_tracker_hooks_dont_raise() -> None:
    t = Tracker()
    t.on_eval_begin("eval-id", {"some": "config"})
    t.on_benchmark_begin("bench", {})
    t.on_episode_end("bench", "task", {"metrics": {"success": True}}, "success")
    t.on_benchmark_end("bench", {"mean_success": 0.5})
    t.on_eval_end([])
    t.close()


# ---------- call_each robustness ----------


class _RecordingTracker(Tracker):
    """Captures every hook call as a tuple for assertions."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def on_eval_begin(self, eval_id, config):
        self.calls.append(("on_eval_begin", (eval_id, config)))

    def on_benchmark_begin(self, bench_name, bench_config):
        self.calls.append(("on_benchmark_begin", (bench_name, bench_config)))

    def on_episode_end(self, bench_name, task_name, ep_dict, status):
        self.calls.append(("on_episode_end", (bench_name, task_name, ep_dict, status)))

    def on_benchmark_end(self, bench_name, result):
        self.calls.append(("on_benchmark_end", (bench_name, result)))

    def on_eval_end(self, all_results):
        self.calls.append(("on_eval_end", (all_results,)))

    def close(self):
        self.calls.append(("close", ()))


class _BrokenTracker(Tracker):
    """Raises on every hook — should not abort other trackers in the list."""

    def on_eval_begin(self, *a, **kw):
        raise RuntimeError("backend exploded")

    def on_episode_end(self, *a, **kw):
        raise RuntimeError("backend exploded")


def test_call_each_isolates_per_tracker_errors() -> None:
    good = _RecordingTracker()
    bad = _BrokenTracker()
    # bad is first so good must still fire even when bad raised
    call_each([bad, good], "on_eval_begin", "eid", {"x": 1})
    assert good.calls == [("on_eval_begin", ("eid", {"x": 1}))]


def test_call_each_handles_unknown_hook_per_tracker() -> None:
    good = _RecordingTracker()
    # Calling a hook that doesn't exist on the base — getattr raises AttributeError,
    # which call_each must swallow per-tracker.
    call_each([good], "nonexistent_hook")
    assert good.calls == []  # the missing hook didn't raise out of call_each


# ---------- scalar helpers ----------


def test_scalar_metrics_keeps_numerics_and_coerces_bool() -> None:
    got = _scalar_metrics({"success": True, "steps": 42, "ratio": 0.5, "label": "ok", "obj": object()})
    assert got == {"success": 1.0, "steps": 42.0, "ratio": 0.5}


def test_scalar_summary_drops_bools_and_strings() -> None:
    got = _scalar_summary({"mean_success": 0.75, "num_episodes": 4, "partial": True, "benchmark": "foo", "tasks": []})
    assert got == {"mean_success": 0.75, "num_episodes": 4}


# ---------- INTEGRATION_TO_TRACKER table ----------


def test_dispatch_table_keys_match_availability_checks() -> None:
    # Anyone adding a backend must wire both the class AND the availability check;
    # this guards the README claim that ``report_to: all`` picks installed backends.
    from vla_eval.tracking import _IS_AVAILABLE

    assert set(INTEGRATION_TO_TRACKER) == set(_IS_AVAILABLE)


# ---------- WandbTracker end-to-end in offline mode ----------


@pytest.mark.skipif(not is_wandb_available(), reason="wandb not installed")
def test_wandb_tracker_offline_e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive the full lifecycle against wandb's offline mode.

    Asserts that:
    - The full sequence of hooks runs without raising.
    - ``on_eval_begin`` creates an offline run directory keyed by ``eval_id``
      (proves the harness-injected id flowed through to wandb.init).
    - The summary keys set by ``on_benchmark_end`` are visible on the live
      ``wandb.run.summary`` dict before ``finish``. (In offline mode wandb does
      not write a JSON-shaped summary file; the binary ``.wandb`` record is the
      source of truth, so we inspect the live object instead.)
    """
    from vla_eval.tracking import WandbTracker

    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_PROJECT", "vla-eval-test")
    monkeypatch.setenv("WANDB_SILENT", "true")  # don't auth or sync

    eval_id = "test-eval-1234"

    t = WandbTracker()
    t.on_eval_begin(eval_id, {"output_dir": str(tmp_path), "benchmarks": []})
    t.on_episode_end(
        "robosuite",
        "lift",
        {"metrics": {"success": True}, "steps": 17, "elapsed_sec": 0.8},
        "success",
    )
    t.on_episode_end(
        "robosuite",
        "lift",
        {"metrics": {"success": False}, "steps": 30, "elapsed_sec": 1.5},
        "fail",
    )
    t.on_benchmark_end("robosuite", {"benchmark": "robosuite", "mean_success": 0.5, "num_episodes": 2})

    # Inspect the live summary before finish() drops `wandb.run`.
    assert t._wandb.run is not None
    summary_snapshot = dict(t._wandb.run.summary)

    t.on_eval_end([])
    t.close()

    # Run directory carries the eval_id we injected (proves the convergence handle works).
    run_dirs = list(tmp_path.glob("wandb/offline-run-*"))
    assert len(run_dirs) == 1, f"expected one offline run dir, found {run_dirs}"
    assert eval_id in run_dirs[0].name, f"eval_id not in run dir name: {run_dirs[0]}"

    assert summary_snapshot.get("robosuite/mean_success") == 0.5
    assert summary_snapshot.get("robosuite/num_episodes") == 2
