"""RoboTwin tasks whose check_success reads attributes that only play_once sets get them after setup_demo."""

import sys
import types
from types import SimpleNamespace

import pytest

from vla_eval.benchmarks.robotwin.benchmark import _restore_play_once_attributes


@pytest.fixture
def fake_envs_utils(monkeypatch):
    utils = types.ModuleType("envs.utils")
    utils.__dict__.update(ArmTag=lambda side: f"tag:{side}", get_face_prod=lambda q, local_axis, target_axis: q[0])
    monkeypatch.setitem(sys.modules, "envs", types.ModuleType("envs"))
    monkeypatch.setitem(sys.modules, "envs.utils", utils)


def _task(name, **attrs):
    return type(name, (), {})() if not attrs else type(name, (), attrs)()


def _pose(p=(0.0, 0.0, 0.0), q=(1.0, 0.0, 0.0, 0.0)):
    return SimpleNamespace(get_pose=lambda: SimpleNamespace(p=p, q=q))


@pytest.mark.parametrize("face, side", [(0.5, "left"), (-0.5, "right")])
def test_open_laptop(fake_envs_utils, face, side):
    env = _task("open_laptop", laptop=_pose(q=(face, 0, 0, 0)))
    _restore_play_once_attributes(env)
    assert env.arm_tag == f"tag:{side}"


@pytest.mark.parametrize("x, side", [(0.1, "right"), (-0.1, "left")])
def test_place_object_scale(fake_envs_utils, x, side):
    env = _task("place_object_scale", object=_pose(p=(x, 0.0, 0.8)))
    _restore_play_once_attributes(env)
    assert env.arm_tag == f"tag:{side}"
    assert not hasattr(env, "origin_z")


def test_put_object_cabinet(fake_envs_utils):
    env = _task("put_object_cabinet", object=_pose(p=(-0.2, 0.0, 0.75)))
    _restore_play_once_attributes(env)
    assert env.arm_tag == "tag:left"
    assert env.origin_z == 0.75


def test_other_tasks_untouched():
    env = _task("beat_block_hammer")
    _restore_play_once_attributes(env)  # needs no RoboTwin import
    assert not hasattr(env, "arm_tag")
