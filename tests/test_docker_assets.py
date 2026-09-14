"""Asset bundles must detect corruption and preserve simulator symlinks."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from vla_eval import assets


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
    assert spec is not None and spec.loader is not None
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
    assert spec is not None and spec.loader is not None
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


def test_automatic_mounts_extract_once_and_recheck_content(tmp_path, monkeypatch):
    calls = []
    manifest = {}

    def fake_extract(image, destination, **kwargs):
        calls.append(image)
        bundle(destination)
        manifest.update(json.loads((destination / assets.MANIFEST).read_text()))

    def fake_docker(*args, **kwargs):
        if args[:2] == ("image", "inspect"):
            return "sha256:1234"
        return json.dumps(manifest)

    monkeypatch.setattr(assets, "extract", fake_extract)
    monkeypatch.setattr(assets, "docker", fake_docker)
    configuration = {"image": "data:tag", "directory": str(tmp_path)}
    mounts = assets.prepare_mounts("runtime:tag", configuration, ensure_local=lambda _: None)
    assert mounts == [str(tmp_path / "sha256-1234/assets") + ":/assets"]
    assert assets.prepare_mounts("runtime:tag", configuration, ensure_local=lambda _: None) == mounts
    assert calls == ["sha256:1234"]
    (tmp_path / "sha256-1234/assets/mesh.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        assets.prepare_mounts("runtime:tag", configuration, ensure_local=lambda _: None)


def test_failed_extraction_does_not_publish_cache(tmp_path, monkeypatch):
    def fail(image, destination, **kwargs):
        (destination / "partial").write_text("incomplete")
        raise RuntimeError("interrupted")

    monkeypatch.setattr(assets, "extract", fail)
    monkeypatch.setattr(assets, "docker", lambda *args, **kwargs: "sha256:5678")
    with pytest.raises(RuntimeError, match="interrupted"):
        assets.prepare_mounts("runtime", {"image": "assets", "directory": str(tmp_path)}, ensure_local=lambda _: None)
    assert not (tmp_path / "sha256-5678").exists()
    assert not list(tmp_path.glob(".extract-*"))


@pytest.mark.parametrize("value", [[], {}, {"image": "data"}, {"image": "data", "directory": 42}])
def test_invalid_asset_configuration_is_rejected(value):
    from vla_eval.config import DockerConfig

    with pytest.raises(ValueError, match="docker.assets"):
        DockerConfig.from_dict({"image": "runtime", "assets": value})
