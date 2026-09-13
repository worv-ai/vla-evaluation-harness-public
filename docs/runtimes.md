# Container runtimes

The benchmark side of an evaluation runs from a pinned OCI image. Two runtimes can execute it;
both consume the same image and the same eval YAML, and both are selected the same way:

```
--runtime <name>   >   $VLA_EVAL_RUNTIME   >   docker.runtime: <name> in the YAML   >   docker
```

| | Docker (default) | Charliecloud |
|---|---|---|
| Needs | Docker daemon, membership in the `docker` group, NVIDIA Container Toolkit for GPUs | Nothing system-wide: the four `ch-*` tools on `PATH` and a kernel that allows unprivileged user namespaces |
| Install | https://docs.docker.com/get-docker/ | `pixi global install charliecloud` (conda-forge, **>= 0.45.1**) |
| OS | Linux, macOS, Windows (via a Linux VM) | Linux only |
| Runs as | image-default user (root) unless `docker.user` is set | the calling user; results are never root-owned |
| Network | `--network host` so `localhost` reaches the model server | host network, always |
| GPU | `--gpus` device flags | `ch-fromhost --nvidia` copies the host driver's user-space libraries into the unpacked image once; needs `nvidia-container-cli` on the host. `docker.gpus` maps to `CUDA_VISIBLE_DEVICES`; an unset/`all` spec inherits the job's own mask (Slurm's `CUDA_VISIBLE_DEVICES`) rather than every GPU on the node |
| CPU pinning (`docker.cpus`) | `--cpuset-cpus` | not applied (no cgroups) |
| Image storage | Docker's image store | `~/.cache/vla-eval/charliecloud/<image>` (override the root with `VLA_EVAL_HOME`); delete a directory to force a fresh pull |

## When to use Charliecloud

Slurm compute nodes rarely run a Docker daemon, and where they do, results written by a root
container are a recurring headache. Charliecloud is the path for that case:

```bash
pixi global install charliecloud          # or: conda install -c conda-forge charliecloud
unshare -Ur true                          # must succeed: unprivileged user namespaces are enabled
vla-eval run --runtime charliecloud --yes -c configs/benchmarks/libero/smoke_test.yaml
```

The first run pulls the image with `ch-image`, exports it to a plain directory with
`ch-convert`, and, unless `render: cpu` or `docker.gpus: none`, injects the NVIDIA driver with
`ch-fromhost --nvidia`. Later runs reuse the directory; concurrent shards on a cold cache serialise on a per-image lock and the export is renamed into place only when complete. The image's own environment is used
(`--unset-env='*' --set-env`), so host variables such as a conda activation do not leak in, and
`HOME` is set to `/root` when the image has one so per-user tool configs baked at build time
(LIBERO's `~/.libero`) are found.

Verified on an H100 node with the LIBERO image: GPU (EGL) and CPU (osmesa) renders both
produce the same frames as the Docker run. See the note on `unshare` below if `ch-run` fails
at startup.

## Unprivileged user namespaces

Charliecloud (and the non-setuid conda-forge build of Apptainer) creates the container with
an unprivileged user namespace: the process maps its own uid to `root` *inside* the namespace,
which lets it set up mounts without any privilege on the host. Some sites disable this because
it exposes kernel code paths that were historically root-only. Check with:

```bash
unshare -Ur true && echo ok
```

If that fails, look at `/proc/sys/user/max_user_namespaces` (must be non-zero),
`kernel.unprivileged_userns_clone` (Debian/Ubuntu, must be 1) and, on Ubuntu 24.04,
`kernel.apparmor_restrict_unprivileged_userns`. A site that keeps these off can only run
containers through a setuid Apptainer installed by its administrators, which vla-eval does not
drive.

## What Charliecloud does not cover

- `vla-eval test --benchmark` (the smoke runner) still launches Docker directly.
- `scripts/run_sharded.sh` works with either runtime, but `docker.cpus` partitioning is a no-op
  under Charliecloud.
- macOS and Windows: use Docker.
