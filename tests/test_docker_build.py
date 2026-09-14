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
    assert image_names(commands) == ["base:slim", "base-render:slim", "libero:slim-gpu"]
    assert "runtime" in commands[1]
    assert "RUNTIME_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base-render:slim" in commands[2]


def test_derived_build_orders_dependencies():
    commands = plan("simpler_xvla", "--tag", "slim")
    assert image_names(commands) == ["base:slim", "base-render:slim", "simpler:slim-gpu", "simpler-xvla:slim-gpu"]
    assert "BUILD_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base:slim" in commands[-1]


def test_digest_overrides_do_not_rebuild_bases():
    commands = plan("libero", "--base-image", "builder@sha256:abc", "--runtime-image", "runtime@sha256:def")
    assert image_names(commands) == ["libero:latest-gpu"]
    assert "BASE_IMAGE=builder@sha256:abc" in commands[0]
    assert "RUNTIME_IMAGE=runtime@sha256:def" in commands[0]


def test_robodojo_retains_its_external_parent_and_license_gate():
    assert plan("robodojo") == []
    commands = plan("robodojo", "--accept-license", "robodojo")
    assert image_names(commands) == ["robodojo:latest-gpu"]
    assert "BASE_IMAGE=robodojo:cuda12.8" in commands[0]
    assert "ACCEPT_NVIDIA_EULA=YES" in commands[0]


def test_build_all_covers_every_recipe_once_and_skips_gated_images():
    names = image_names(plan())
    expected = {p.name.removeprefix("Dockerfile.").replace("_", "-") for p in (ROOT / "docker").glob("Dockerfile.*")}
    expected |= {"base-runtime", "base-render", "base-cuda"}
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


def test_cpu_profile_uses_separate_runtime_and_tag():
    commands = plan("libero", "--profile", "all", "--tag", "split")
    assert image_names(commands) == [
        "base:split",
        "base-render:split",
        "libero:split-gpu",
        "base-cpu:split",
        "libero:split-cpu",
    ]
    assert "runtime-cpu" in commands[-2]
    assert "IMAGE_PROFILE=cpu" in commands[-1]
    assert "RUNTIME_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base-cpu:split" in commands[-1]


def test_gpu_only_benchmark_rejects_cpu_before_build():
    result = subprocess.run(
        ["bash", str(BUILD), "robotwin", "--profile", "cpu", "--dry-run"], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "requires a GPU" in result.stderr
    assert "docker build " not in result.stdout


def test_cpu_all_builds_only_supported_images():
    commands = plan("--profile", "cpu")
    names = image_names(commands)
    assert "kinetix:latest-cpu" in names
    assert "kinetix:latest-gpu" in names
    assert not any("robotwin" in name or "simpler" in name for name in names)


def test_cpu_entrypoint_forces_render_and_preserves_arguments(tmp_path):
    executable = tmp_path / "capture"
    executable.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    executable.chmod(0o755)
    result = subprocess.run(
        ["sh", str(ROOT / "docker/image_entrypoint.sh"), str(executable), "run", "--config", "a b.yaml"],
        capture_output=True,
        text=True,
        env={**os.environ, "VLA_IMAGE_PROFILE": "cpu"},
    )
    assert result.returncode == 0
    assert result.stdout.splitlines() == ["run", "--config", "a b.yaml", "--render", "cpu"]


def test_push_routes_cpu_gpu_tags_and_gpu_compatibility_alias(tmp_path):
    fake = tmp_path / "docker"
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$DOCKER_CALLS"\n')
    fake.chmod(0o755)
    calls = tmp_path / "calls"
    subprocess.run(
        ["bash", str(ROOT / "docker/push.sh"), "libero", "--tag", "split", "--profile", "all", "--no-latest"],
        env={**os.environ, "PATH": str(tmp_path) + ":" + os.environ["PATH"], "DOCKER_CALLS": str(calls)},
        check=True,
        capture_output=True,
        text=True,
    )
    pushes = [line.split()[-1] for line in calls.read_text().splitlines() if line.startswith("push ")]
    assert [ref.rsplit(":", 1)[-1] for ref in pushes] == ["split-gpu", "split", "split-cpu"]


def test_cpu_wheels_keep_public_versions_and_remove_accelerator_packages(monkeypatch):
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location("prepare_cpu", ROOT / "docker/prepare_cpu.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    versions = {
        "torch": "2.9.1+cu128",
        "torchvision": "0.24.1+cu128",
        "jax": "0.7.2",
        "jax-cuda12-plugin": "0.7.2",
        "nvidia-cublas-cu12": "12.8.4.1",
        "triton": "3.5.1",
    }
    monkeypatch.setattr(
        prepare.metadata,
        "distributions",
        lambda: [SimpleNamespace(metadata={"Name": name}, version=version) for name, version in versions.items()],
    )
    calls = []
    monkeypatch.setattr(prepare.subprocess, "run", lambda args, **kwargs: calls.append(args))
    prepare.main()
    assert "torch==2.9.1+cpu" in calls[0]
    assert "torchvision==0.24.1+cpu" in calls[0]
    assert "--no-deps" in calls[0]
    assert calls[0][calls[0].index("--torch-backend") + 1] == "cpu"
    assert set(calls[1][5:]) == {"jax-cuda12-plugin", "nvidia-cublas-cu12", "triton"}


def test_rlbench_keeps_tini_outside_the_profile_wrapper():
    import json

    recipe = (ROOT / "docker/Dockerfile.rlbench").read_text()
    entrypoint = json.loads(next(line[11:] for line in recipe.splitlines() if line.startswith("ENTRYPOINT ")))
    assert entrypoint == ["tini", "-g", "--", "/usr/local/bin/image-entrypoint", "/rlbench_entrypoint.sh"]


def test_cpu_can_derive_from_an_immutable_gpu_artifact():
    commands = plan("libero", "--profile", "cpu", "--gpu-image", "libero@sha256:abc")
    assert image_names(commands) == ["base-cpu:latest", "libero:latest-cpu"]
    assert "GPU_IMAGE=libero@sha256:abc" in commands[-1]


def test_cpu_recipe_preserves_rlbench_native_runtime_and_editable_sources():
    result = subprocess.run(
        ["bash", str(ROOT / "docker/generate_cpu_dockerfile.sh"), str(ROOT / "docker/Dockerfile.rlbench")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "FROM ${GPU_IMAGE} AS gpu" in result.stdout
    assert "FROM gpu AS builder" in result.stdout
    assert "export_runtime.py --inventory-only" in result.stdout
    assert "FROM ${RUNTIME_IMAGE} AS runtime" in result.stdout
    assert "xvfb libfontconfig1 tini libdbus-1-3" in result.stdout
    assert "COPY --from=gpu /tmp/PyRep /tmp/PyRep" in result.stdout
    assert "COPY --from=gpu /opt/coppeliasim /opt/coppeliasim" in result.stdout
    assert "ARG IMAGE_PROFILE=cpu" in result.stdout


def test_custom_cpu_runtime_is_not_used_to_build_its_gpu_source():
    commands = plan("libero", "--profile", "cpu", "--runtime-image", "cpu-runtime@sha256:abc")
    assert image_names(commands) == ["base:latest", "base-render:latest", "libero:latest-gpu", "libero:latest-cpu"]
    assert "RUNTIME_IMAGE=cpu-runtime@sha256:abc" in commands[-1]
    assert "RUNTIME_IMAGE=ghcr.io/allenai/vla-evaluation-harness/base-render:latest" in commands[-2]
