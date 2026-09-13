"""Charliecloud runtime: run the benchmark image without a daemon or root.

Charliecloud (https://hpc.github.io/charliecloud/) unpacks the same OCI image
Docker would pull into a plain directory and runs it inside an unprivileged
user namespace. Nothing is installed system-wide, so it works on Slurm nodes
where ``docker`` is unavailable. Requirements:

* ``ch-image``/``ch-run``/``ch-convert``/``ch-fromhost`` >= 0.45.1 on ``PATH``
  (``pixi global install charliecloud`` or conda-forge; 0.45 rejects Ubuntu images).
* Unprivileged user namespaces enabled: ``unshare -Ur true`` must succeed.
* For GPU rendering, ``nvidia-container-cli`` on the host (``ch-fromhost --nvidia``
  copies the driver's user-space libraries into the image directory once).

Differences from Docker that callers should know:

* The image runs as the calling uid; results are never root-owned (``docker.user`` is
  ignored) and the host network is shared (no ``--network host`` needed).
* ``docker.cpus`` pinning is not applied (no cgroups); ``docker.gpus`` maps to
  ``CUDA_VISIBLE_DEVICES``.
* The image directory lives under ``vla_eval.dirs.home()/charliecloud`` and is reused
  until removed; pass ``--yes`` to pull a missing one without a prompt.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from vla_eval import dirs
from vla_eval.cli._console import stderr_console as _stderr_console
from vla_eval.config import DockerConfig

logger = logging.getLogger(__name__)

TOOLS = ("ch-image", "ch-run", "ch-convert", "ch-fromhost")
NVIDIA_MARKER = ".vla-eval-nvidia-injected"


def find_tools() -> dict[str, str]:
    """Absolute paths of the Charliecloud tools, or exit with an install hint."""
    found = {t: shutil.which(t) for t in TOOLS}
    missing = [t for t, path in found.items() if path is None]
    if missing:
        _stderr_console().print(
            f"[red]ERROR: Charliecloud tools not found on PATH: {', '.join(missing)}.[/red]\n"
            "  Install with: pixi global install charliecloud   (or conda-forge; needs >= 0.45.1)\n"
            "  See docs/runtimes.md."
        )
        sys.exit(1)
    return {t: str(path) for t, path in found.items()}


def image_dir_for(image: str, root: Path | None = None) -> Path:
    """Unpacked image directory: ``<root>/<image with '/'→'%' and ':'→'+'>`` (ch-image's own mangling)."""
    root = root or dirs.home() / "charliecloud"
    return root / image.replace("/", "%").replace(":", "+")


def ensure_image_dir(image: str, *, auto_yes: bool, gpu: bool, tools: dict[str, str] | None = None) -> Path:
    """Pull + unpack *image* on first use, inject the NVIDIA driver once when *gpu*; return the directory.

    Concurrent shards on a cold cache serialise on a per-image lock, and the export lands in a
    temporary directory that is renamed into place only once complete.
    """
    import fcntl

    tools = tools or find_tools()
    img_dir = image_dir_for(image)
    img_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{img_dir}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (img_dir / "ch" / "metadata.json").is_file():
            con = _stderr_console()
            con.print(f"\n[yellow]⚠  Charliecloud image directory for '{image}' not found at {img_dir}.[/yellow]")
            con.print("   Pulling and unpacking may take several minutes and use several GB of disk.\n")
            if not auto_yes:
                if not sys.stdin.isatty():
                    con.print(
                        "[red]ERROR: Cannot confirm in non-interactive mode. Use --yes to skip confirmation.[/red]"
                    )
                    sys.exit(1)
                if input("Proceed with ch-image pull? [y/N] ").strip().lower() not in ("y", "yes"):
                    con.print("Aborted.")
                    sys.exit(0)
            if subprocess.call([tools["ch-image"], "pull", image]) != 0:
                con.print(f"[red]ERROR: ch-image pull failed for {image}.[/red]")
                sys.exit(1)
            # ch-run refuses directories inside ch-image's storage; export a plain copy.
            tmp_dir = Path(f"{img_dir}.tmp-{os.getpid()}")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if subprocess.call([tools["ch-convert"], "-i", "ch-image", "-o", "dir", image, str(tmp_dir)]) != 0:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                con.print(f"[red]ERROR: ch-convert failed for {image}.[/red]")
                sys.exit(1)
            shutil.rmtree(img_dir, ignore_errors=True)  # a partial export from an earlier failure
            os.rename(tmp_dir, img_dir)
        if gpu and not (img_dir / NVIDIA_MARKER).exists():
            if shutil.which("nvidia-container-cli") is None:
                _stderr_console().print(
                    "[red]ERROR: GPU rendering under Charliecloud needs nvidia-container-cli on the host "
                    "(ch-fromhost --nvidia). Install libnvidia-container, or run with --render cpu.[/red]"
                )
                sys.exit(1)
            if subprocess.call([tools["ch-fromhost"], "--nvidia", str(img_dir)]) != 0:
                _stderr_console().print("[red]ERROR: ch-fromhost --nvidia failed.[/red]")
                sys.exit(1)
            (img_dir / NVIDIA_MARKER).touch()
    return img_dir


def read_metadata(img_dir: Path) -> dict[str, Any]:
    """Entrypoint/cmd/cwd that ch-image saved from the OCI config."""
    with open(img_dir / "ch" / "metadata.json") as f:
        return json.load(f)


def _bind(spec: str) -> list[str]:
    """Docker ``-v host:container[:ro]`` → ``ch-run -b host:container`` (bind mounts are always rw)."""
    parts = spec.split(":")
    if len(parts) == 3 and parts[2] in ("ro", "rw"):
        spec = f"{parts[0]}:{parts[1]}"
    return ["-b", spec]


def build_ch_run_cmd(
    img_dir: Path,
    *,
    ch_run: str,
    results_dir: str,
    config_path: str,
    env: dict[str, str],
    volumes: list[str],
    dev_mount: list[str] | None,
    inner_args: list[str],
) -> list[str]:
    """Assemble the ``ch-run`` command line. Pure; unit-tested without Charliecloud installed."""
    from vla_eval.cli._docker import CONTAINER_CONFIG, CONTAINER_RESULTS

    meta = read_metadata(img_dir)
    entrypoint = list(meta.get("entrypoint") or [])
    # --write-fake: tmpfs overlay so mount points (/workspace/results) can be created in a read-only image.
    # --unset-env=* then --set-env: start from the image's environment, as Docker does, not the host's.
    cmd = [ch_run, "--write-fake", "--unset-env=*", "--set-env"]
    if (img_dir / "root").is_dir():
        cmd.append("--set-env=HOME=/root")  # images configure tools under /root (e.g. ~/.libero)
    for key, value in env.items():
        cmd.append(f"--set-env={key}={value}")
    cmd.extend(["--cd", meta.get("cwd") or "/workspace"])
    cmd.extend(_bind(f"{results_dir}:{CONTAINER_RESULTS}"))
    cmd.extend(_bind(f"{config_path}:{CONTAINER_CONFIG}"))
    if dev_mount:
        cmd.extend(_bind(dev_mount[1]))
    for vol in volumes:
        cmd.extend(_bind(vol))
    cmd.extend([str(img_dir), "--", *entrypoint, *inner_args])
    return cmd


def _host_visible_gpus() -> str | None:
    """The scheduler's device mask (Slurm sets CUDA_VISIBLE_DEVICES per job), or None."""
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        if key in os.environ:
            return os.environ[key]
    return None


def _gpu_env(gpus: str | None, shard_id: int | None, num_shards: int | None) -> dict[str, str]:
    """Device visibility for the container (no device flags: the driver is in the image directory).

    ``--unset-env=*`` drops the host mask, so an ``all``/unset spec re-applies it explicitly
    instead of exposing every GPU on the node; shards round-robin within that mask.
    """
    from vla_eval.docker_resources import gpu_visibility_env, is_no_gpu_spec, parse_gpus

    if is_no_gpu_spec(gpus):
        return {"CUDA_VISIBLE_DEVICES": ""}
    want_all = gpus is None or gpus.strip().lower() == "all"
    host_mask = _host_visible_gpus()
    if num_shards is None:
        return gpu_visibility_env(host_mask if want_all else gpus)
    assert shard_id is not None
    if want_all and host_mask is not None:
        pool = [g.strip() for g in host_mask.split(",") if g.strip()]
    else:
        pool = parse_gpus(gpus)
    if not pool:
        return {"CUDA_VISIBLE_DEVICES": ""}
    env = gpu_visibility_env(pool[shard_id % len(pool)])
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})  # same reasoning as shard_docker_flags
    return env


def run_via_charliecloud(
    config: dict[str, Any],
    *,
    auto_yes: bool = False,
    dev: bool = False,
    shard_id: int | None = None,
    num_shards: int | None = None,
    accept_license: list[str] | None = None,
    eval_id: str | None = None,
    no_save: bool = False,
) -> int:
    """Execute the evaluation under ``ch-run``. Returns the exit code."""
    from vla_eval.cli._docker import dev_src_mount_flags, exec_child, inner_run_args, prepare_container_config
    from vla_eval.docker_resources import is_no_gpu_spec

    tools = find_tools()
    docker_cfg = DockerConfig.from_dict(config.get("docker"))
    if docker_cfg.image is None:
        _stderr_console().print("[red]ERROR: 'docker.image' must be set in config[/red]")
        sys.exit(1)
    if docker_cfg.user:
        logger.info("docker.user=%r ignored: Charliecloud always runs as the calling user", docker_cfg.user)
    if docker_cfg.cpus:
        logger.info("docker.cpus=%r ignored: Charliecloud does not pin CPU sets", docker_cfg.cpus)

    gpu = not is_no_gpu_spec(docker_cfg.gpus) and str(config.get("render", "gpu")).lower() != "cpu"
    img_dir = ensure_image_dir(docker_cfg.image, auto_yes=auto_yes, gpu=gpu, tools=tools)

    dev_mount: list[str] | None = None
    if dev:
        try:
            dev_mount = dev_src_mount_flags()
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        logger.info("Dev mode: mounting %s -> /workspace/src", dev_mount[1].split(":", 1)[0])

    results_dir, config_path = prepare_container_config(config)
    env: dict[str, str] = {"VLA_EVAL_HOST_OUTPUT_DIR": results_dir}
    if os.environ.get("VLA_EVAL_WATCHDOG_TIMEOUT_S"):
        env["VLA_EVAL_WATCHDOG_TIMEOUT_S"] = os.environ["VLA_EVAL_WATCHDOG_TIMEOUT_S"]
    for env_str in docker_cfg.env:
        key, _, value = env_str.partition("=")
        env[key] = value if _ else os.environ.get(key, "")
    if accept_license:
        env["VLA_EVAL_ACCEPTED_LICENSES"] = ",".join(accept_license)
    env.update(_gpu_env(docker_cfg.gpus, shard_id, num_shards))

    cmd = build_ch_run_cmd(
        img_dir,
        ch_run=tools["ch-run"],
        results_dir=results_dir,
        config_path=config_path,
        env=env,
        volumes=docker_cfg.volumes,
        dev_mount=dev_mount,
        inner_args=inner_run_args(shard_id=shard_id, num_shards=num_shards, eval_id=eval_id, no_save=no_save),
    )
    logger.info("Running via Charliecloud: %s", " ".join(cmd))

    def _stop(proc: Any) -> None:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    try:
        return exec_child(cmd, _stop)
    finally:
        Path(config_path).unlink(missing_ok=True)
