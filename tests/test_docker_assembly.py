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


def test_every_profile_lock_pin_has_hashes_and_cpu_excludes_cuda(monkeypatch):
    catalog = load_module("catalog", monkeypatch)
    hashes = {line.split()[0]: line for line in (ROOT / "docker/locks/hashes.txt").read_text().splitlines()}
    used = set()
    for name, spec in catalog.BENCHMARKS.items():
        if "runtime_paths" not in spec:
            continue
        for profile in ["gpu", "cpu"] if spec["cpu"] else ["gpu"]:
            pins = catalog.lock_file(name, profile).read_text().split()
            assert pins, name
            for pin in pins:
                package, version = pin.split("==")
                key = catalog.normalize(package) + "==" + version
                assert "--hash=sha256:" in hashes.get(key, ""), (name, pin)
                used.add(key)
                if profile == "cpu":
                    assert not pin.startswith(("nvidia-", "jax-cuda", "triton==")), (name, pin)
    assert set(hashes) == used, "hashes.txt has unused pins; run docker/lock.py"


def test_harness_changes_do_not_invalidate_locked_dependency_install(monkeypatch):
    renderer = load_module("render", monkeypatch)
    for name, spec in renderer.BENCHMARKS.items():
        if "runtime_paths" not in spec:
            continue
        text = renderer.render(name, spec)
        locked = text.index("RUN python /usr/local/lib/vla/install_locked.py")
        assert text.count("COPY src/ src/") == 1
        assert locked < text.index("COPY pyproject.toml README.md ./") < text.index("COPY src/ src/")
        assert "RUN uv pip install --no-cache-dir --no-deps -e ." in text
        if spec["prune"]["libero_numpy"]:
            assert text.index("COPY src/ src/") < text.index("python /usr/local/lib/vla/optimize_libero.py")
    robotwin = renderer.render("robotwin", renderer.BENCHMARKS["robotwin"])
    locked = robotwin.index("RUN python /usr/local/lib/vla/install_locked.py")
    assert locked < robotwin.index("urdf_loader.py") and locked < robotwin.index("planner.py")


def test_locks_include_direct_harness_dependencies():
    import tomllib

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    names = set()
    for requirement in metadata["project"]["dependencies"]:
        match = re.match(r"[\w.-]+", requirement)
        assert match is not None
        names.add(re.sub(r"[-_.]+", "-", match[0].lower()))
    for lock in (ROOT / "docker/locks").glob("*.in"):
        installed = {line.split("==")[0] for line in lock.read_text().splitlines()}
        # Python >=3.11 does not require the typing backport.
        assert names - {"typing-extensions"} <= installed, lock.name
