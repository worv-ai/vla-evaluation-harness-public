"""Runtime assembly must preserve code and links without duplicating data."""

import importlib.util
import json
import re
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "docker"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "docker" / (name + ".py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_excludes_top_level_and_nested_assets(tmp_path, monkeypatch):
    module = load_module("assemble", monkeypatch)
    source, destination = tmp_path / "source", tmp_path / "runtime"
    for name in ("assets/mesh.bin", "app/simulator/assets/texture.bin", "app/simulator/code.py"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"payload")
    (source / "app/code-link").symlink_to("simulator/code.py")
    (source / "usr/local/share/vla-build").mkdir(parents=True)
    spec = {"assets": ["/assets", "/app/simulator/assets"], "runtime_paths": ["/assets", "/app"]}
    module.assemble(spec, source, destination)
    assert not list((destination / "assets").iterdir())
    assert not list((destination / "app/simulator/assets").iterdir())
    assert (destination / "app/code-link").read_bytes() == b"payload"
    assert (source / "assets/mesh.bin").exists()
    manifest = json.loads((destination / "usr/local/share/vla-build/assets.json").read_text())
    assert {f["path"] for f in manifest["files"]} == {"/assets/mesh.bin", "/app/simulator/assets/texture.bin"}
    verifier = load_module("verify_assets", monkeypatch)
    lock = {"example": {"sha256": verifier.tree_digest(manifest)}}
    verifier.verify_lock("example", manifest, lock)
    manifest["files"][0]["sha256"] = "changed"
    with pytest.raises(ValueError, match="Asset content changed"):
        verifier.verify_lock("example", manifest, lock)


def test_generated_recipes_are_current_and_shell_parses():
    subprocess.run([sys.executable, str(ROOT / "docker/render.py"), "--check"], check=True)
    for recipe in (ROOT / "docker/recipes").glob("*.Dockerfile"):
        text = recipe.read_text().replace("\\\n", " ")
        for line in text.splitlines():
            if line.startswith("RUN "):
                subprocess.run(
                    ["bash", "-n"],
                    input=re.sub(r"^RUN (?:--\S+ +)*", "", line),
                    text=True,
                    check=True,
                    capture_output=True,
                )


def test_every_profile_lock_has_artifact_hashes_and_cpu_excludes_cuda():
    catalog = json.loads((ROOT / "docker/images.json").read_text())
    for name, spec in catalog["benchmarks"].items():
        if "runtime_paths" not in spec:
            continue
        for profile in ["gpu", "cpu"] if spec["cpu"] else ["gpu"]:
            lock = ROOT / "docker/locks" / (name + "-" + profile + ".txt")
            logical_lines = lock.read_text().replace("\\\n", " ").splitlines()
            assert logical_lines, name
            for line in logical_lines:
                if not line or line.startswith("#"):
                    continue
                assert "--hash=sha256:" in line, (name, line)
                if profile == "cpu":
                    assert not line.startswith(("nvidia-", "jax-cuda", "triton==")), (name, line)
