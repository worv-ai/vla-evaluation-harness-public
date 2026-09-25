"""RoboCasa adapter logic that runs without the image: proprio state and asset-path rewriting."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from vla_eval.benchmarks.robocasa import benchmark as rc

RAW_OBS = {
    "robot0_agentview_left_image": np.zeros((4, 4, 3), np.uint8),
    "robot0_base_to_eef_pos": np.array([0.2, 0.0, 0.6]),
    "robot0_base_to_eef_quat": np.array([-0.99, -0.04, -0.12, 0.01]),
    "robot0_gripper_qpos": np.array([0.02, -0.02]),
    "robot0_base_pos": np.array([1.0, 2.0, 0.7]),
}


def test_state_is_sent_only_when_asked():
    assert "states" not in rc.RoboCasaBenchmark().make_obs(RAW_OBS, {"name": "OpenDrawer"})
    bench = rc.RoboCasaBenchmark(send_state=True)
    states = bench.make_obs(RAW_OBS, {"name": "OpenDrawer"})["states"]
    np.testing.assert_allclose(states, [0.2, 0.0, 0.6, -0.99, -0.04, -0.12, 0.01, 0.02, -0.02], rtol=1e-6)
    assert states.dtype == np.float32 and bench.get_observation_spec()["state"].dims == 9


def test_relative_asset_paths_become_absolute():
    root = ET.fromstring(
        '<mujoco><asset><mesh file="meshes/a.obj"/><texture file="/abs/t.png"/></asset><body><geom/></body></mujoco>'
    )
    rc._absolutize_asset_paths(root, "/assets/objects/fish_5")
    assert [e.get("file") for e in root.iter() if e.get("file")] == [
        "/assets/objects/fish_5/meshes/a.obj",
        "/abs/t.png",
    ]
