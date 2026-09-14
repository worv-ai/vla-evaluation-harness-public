"""Exercise build routing without invoking Docker or downloading packages."""

import importlib.util
import os
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "docker/build.sh"
spec = importlib.util.spec_from_file_location("export_runtime", ROOT / "docker/export_runtime.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def plan(*args):
    result = subprocess.run(
        ["bash", str(BUILD), *args, "--dry-run"],
        cwd="/tmp",
        text=True,
        capture_output=True,
        env={**os.environ, "HARNESS_VERSION": "1.2.3"},
    )
    assert result.returncode == 0, result.stderr
    return [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("docker build ")]


def image_names(commands):
    return [command[command.index("-t") + 1].split("/")[-1] for command in commands]


def test_cpu_benchmark_uses_small_runtime_and_requested_tag():
    commands = plan("libero", "--tag", "slim")
    assert image_names(commands) == ["base:slim", "base-cpu:slim", "libero:slim"]
    assert any(arg.startswith("CUDA_IMAGE=ubuntu:22.04@sha256:") for arg in commands[1])
    assert "RUNTIME_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base-cpu:slim" in commands[2]


def test_derived_build_orders_dependencies():
    commands = plan("simpler_xvla", "--tag", "slim")
    assert image_names(commands) == ["base:slim", "base-cpu:slim", "simpler:slim", "simpler-xvla:slim"]
    assert "BUILD_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base:slim" in commands[-1]


def test_digest_overrides_do_not_rebuild_bases():
    commands = plan("libero", "--base-image", "builder@sha256:abc", "--runtime-image", "runtime@sha256:def")
    assert image_names(commands) == ["libero:latest"]
    assert "BASE_IMAGE=builder@sha256:abc" in commands[0]
    assert "RUNTIME_IMAGE=runtime@sha256:def" in commands[0]


def test_robodojo_retains_its_external_parent_and_license_gate():
    assert plan("robodojo") == []
    commands = plan("robodojo", "--accept-license", "robodojo")
    assert image_names(commands) == ["robodojo:latest"]
    assert "BASE_IMAGE=robodojo:cuda12.8" in commands[0]
    assert "ACCEPT_NVIDIA_EULA=YES" in commands[0]


def test_build_all_covers_every_recipe_once_and_skips_gated_images():
    names = image_names(plan())
    expected = {p.name.removeprefix("Dockerfile.").replace("_", "-") for p in (ROOT / "docker").glob("Dockerfile.*")}
    expected |= {"base-runtime", "base-cpu"}
    expected -= {"rlbench", "behavior1k", "robodojo"}
    assert {name.split(":")[0] for name in names} == expected
    assert len(names) == len(set(names))


@pytest.mark.parametrize("args", [("unknown",), ("--tag",), ("--accept-license", "unknown"), ("libero", "simpler")])
def test_invalid_requests_fail_before_building(args):
    result = subprocess.run(["bash", str(BUILD), *args, "--dry-run"], text=True, capture_output=True)
    assert result.returncode != 0
    assert "docker build " not in result.stdout


def test_build_arguments_preserve_spaces_and_shell_metacharacters():
    value = "SOURCE=some value; $(touch /tmp/should-not-exist)"
    commands = plan("base", "--build-arg", value)
    assert value in commands[0]


def test_export_preserves_editable_sources_assets_and_symlinks(tmp_path):
    source = tmp_path / "app"
    source.mkdir()
    asset = source / "asset.bin"
    asset.write_bytes(b"asset-data")
    (source / "relative").symlink_to("asset.bin")
    (source / "absolute").symlink_to(str(asset))
    (source / "module.py").write_text("VALUE = 42\n")
    destination = tmp_path / "runtime-root"
    exporter.export_runtime([str(source)], destination)
    copied = destination / source.relative_to("/")
    assert not source.exists()
    assert (copied / "relative").read_bytes() == b"asset-data"
    assert os.readlink(copied / "absolute") == str(asset)
    assert (copied / "module.py").read_text() == "VALUE = 42\n"


def test_export_does_not_silently_drop_required_asset_paths(tmp_path):
    with pytest.raises(FileNotFoundError):
        exporter.export_runtime([str(tmp_path / "missing-assets")], tmp_path / "runtime-root")


@pytest.mark.parametrize("path", ["/", "relative/path", "/opt/../etc"])
def test_export_rejects_ambiguous_paths(tmp_path, path):
    with pytest.raises(ValueError):
        exporter.export_runtime([path], tmp_path / "runtime-root")


def test_runtime_source_paths_include_temporary_editable_checkouts():
    for name, paths in {
        "rlbench": ["/tmp/PyRep", "/tmp/RLBench", "/opt/coppeliasim"],
        "robocerebra": ["/tmp/RoboCerebra"],
        "molmospaces": ["/assets", "/cache/molmo-spaces-resources"],
    }.items():
        recipe = (ROOT / "docker" / f"Dockerfile.{name}").read_text()
        export = next(
            line for line in recipe.splitlines() if "RUN python /usr/local/lib/vla/export_runtime.py" in line
        )
        assert all(path in shlex.split(export) for path in paths)


def test_default_configs_exist_in_build_context():
    import json

    for recipe in (ROOT / "docker").glob("Dockerfile.*"):
        for line in recipe.read_text().splitlines():
            if line.startswith("CMD "):
                command = json.loads(line[4:])
                if "--config" in command:
                    path = command[command.index("--config") + 1]
                    relative = "configs/" + path.split("/configs/", 1)[1]
                    assert (ROOT / relative).is_file(), (recipe.name, path)
