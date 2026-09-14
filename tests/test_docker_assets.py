"""Asset bundles must detect corruption and preserve simulator symlinks."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("assets", Path(__file__).resolve().parents[1] / "docker/assets.py")
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


def bundle(tmp_path):
    asset = tmp_path / "assets/mesh.bin"
    asset.parent.mkdir()
    asset.write_bytes(b"mesh")
    (asset.parent / "alias").symlink_to("/assets/mesh.bin")
    manifest = {
        "version": 1,
        "paths": ["/assets"],
        "files": [
            {"path": "/assets/mesh.bin", "sha256": hashlib.sha256(b"mesh").hexdigest(), "bytes": 4},
            {"path": "/assets/alias", "symlink": "/assets/mesh.bin"},
        ],
    }
    output = tmp_path / assets.MANIFEST
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps(manifest))
    return asset


def test_verify_accepts_absolute_simulator_symlink(tmp_path):
    bundle(tmp_path)
    assert assets.verify(tmp_path)["paths"] == ["/assets"]


def test_verify_rejects_modified_asset(tmp_path):
    asset = bundle(tmp_path)
    asset.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        assets.verify(tmp_path)


def test_verify_rejects_file_replaced_by_symlink(tmp_path):
    asset = bundle(tmp_path)
    asset.unlink()
    asset.symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="Expected an asset file"):
        assets.verify(tmp_path)


def test_paths_cannot_escape_bundle(tmp_path):
    (tmp_path / "assets").symlink_to("/tmp", target_is_directory=True)
    with pytest.raises(ValueError, match="escapes bundle"):
        assets.local_path(tmp_path, "/assets/mesh")
    with pytest.raises(ValueError, match="Invalid asset path"):
        assets.local_path(tmp_path, "/assets/../../etc/passwd")


def test_cpu_pruning_preserves_runtime_resources(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "prune_cpu", Path(__file__).resolve().parents[1] / "docker/prune_cpu.py"
    )
    pruner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pruner)
    prefix = tmp_path / "env"
    site = prefix / "lib/python3.8/site-packages"
    retained = [
        site / "torch/bin/torch_shm_manager",
        site / "open3d/cpu/pybind.so",
        site / "numpy/_core/tests/_natype.py",
        tmp_path / "app/libero-pro/notebooks/custom_assets/mesh.xml",
    ]
    deleted = [
        site / "torch/bin/test_api",
        site / "open3d/cuda/pybind.so",
        tmp_path / "app/libero-pro/notebooks/demo.ipynb",
    ]
    for path in retained + deleted:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"payload")
    pruner.prune(prefix, tmp_path / "app", tmp_path / "pruned.json")
    assert all(path.exists() for path in retained)
    assert all(not path.exists() for path in deleted)


def test_initial_state_conversion_preserves_float_bits(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    import numpy as np

    spec = importlib.util.spec_from_file_location(
        "optimize_libero", Path(__file__).resolve().parents[1] / "docker/optimize_libero.py"
    )
    optimizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(optimizer)
    original = np.array([[0.0, -0.0, np.nan], [1.25, 1e-200, -3.5]], dtype=np.float64)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=lambda *a, **kw: original))
    source = tmp_path / "state.pruned_init"
    source.write_bytes(b"serialized state")
    record = optimizer.convert_state(source)
    replay = np.load(record["target"], allow_pickle=False)
    assert replay.dtype == original.dtype
    assert replay.tobytes() == original.tobytes()
    assert record["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_runtime_rejects_bundle_from_another_revision(tmp_path, monkeypatch):
    bundle(tmp_path)
    calls = []

    def fake_docker(*args, **kwargs):
        calls.append(args)
        return json.dumps({"version": 1, "paths": ["/assets"], "files": []})

    monkeypatch.setattr(assets, "docker", fake_docker)
    with pytest.raises(ValueError, match="manifests differ"):
        assets.run("runtime:version", tmp_path, ["test"])
    assert len(calls) == 1  # Only read the manifest; evaluation never starts.
