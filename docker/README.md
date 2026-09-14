# Docker Images

Each benchmark has an isolated runtime. Build `VERSION-gpu` for NVIDIA rendering
or compute, and `VERSION-cpu` for software rendering without an attached GPU.
The unsuffixed tag remains a compatibility alias for the GPU image.

## Image hierarchy

| Image | Target | Contents |
| --- | --- | --- |
| `base` | `builder` | Ubuntu, compilers, Pixi and uv; build use only |
| `base-cuda` | `builder`, CUDA parent | Build tools plus system CUDA for BEHAVIOR/RoboTwin |
| `base-runtime` | `runtime`, CUDA parent | CUDA 12.1 and NVIDIA rendering libraries |
| `base-render` | `runtime` | NVIDIA EGL/Vulkan support without system CUDA |
| `base-cpu` | `runtime-cpu` | Mesa software rendering, GPU visibility disabled |

`docker/images.sh` lists the CPU-capable benchmarks: LIBERO variants,
RoboCerebra, CALVIN, RLBench, DuoBench, RoboCasa/365, Kinetix, RoboMME,
MolmoSpaces and VLABench. GPU-only benchmarks reject an explicit CPU build;
`--profile all` builds both variants where supported. GPU image does not imply
CUDA PyTorch: simulators such as LIBERO use GPU rendering with CPU PyTorch.

CPU builds derive from a GPU artifact. They replace accelerator PyTorch wheels
with the same public version's CPU wheels and remove CUDA libraries, Triton and
JAX CUDA plugins. The final image copies only the selected environment, source
and asset paths into a fresh CPU runtime; the GPU parent's layers are not retained.
Unchanged asset layers can be shared by both images. Simulator installation and
asset downloads run only once when building both profiles.

`generate_cpu_dockerfile.sh` combines a small conversion stage with the benchmark's
existing runtime section, preserving native libraries, editable paths and startup
wrappers without maintaining duplicate recipes. CPU entrypoints force
`--render cpu` for harness `run` and `test`. Direct Python invocations should call
`configure_render("cpu")` before importing simulator rendering modules.

Python 3.10/3.11 environments normally use uv-managed, patch-pinned Python and
`/opt/venv`. Legacy Python 3.8, BEHAVIOR and DuoBench use the native environments
in `docker/pixi/pixi.lock`, installed with `pixi install --locked`. Their Python
and native package artifacts are locked by URL and checksum. No Miniforge base
environment is installed. Pixi and uv remain in builders; only the selected
environment at `/opt/pixi/.pixi/envs/PROFILE` enters the runtime. PyPI simulator
packages are still installed by uv after native environment creation.

Exceptions:

- RoboTwin retains nvcc, C++ tools and Python build dependencies for CuRobo JIT.
- CALVIN retains Git and its minimal repositories for runtime version queries.
- RLBench retains CoppeliaSim, Xvfb and tini. Its wrapper executes the harness
  directly, preserving signal delivery without a Conda subprocess.
- Simpler variants inherit the slim Simpler runtime; X-VLA temporarily mounts the
  build-stage `patch` binary using BuildKit.
- RoboDojo retains the externally built, licensed Isaac Sim base unchanged. Its
  harness uses a separate uv venv with access to the upstream environment. This
  repository retains that external base; removing its CUDA toolchain needs separate
  upstream validation. pip installs the harness inside the uv-created venv so
  existing simulator packages are reused instead of shadowed by new versions.

Editable sources under `/tmp` are runtime dependencies in RLBench and
RoboCerebra. MolmoSpaces assets link into `/cache/molmo-spaces-resources`. The
exporter preserves these locations, permissions and symlinks. It removes only
pip/uv installer caches; it does not delete arbitrary `tests`, headers, static
libraries, simulator caches or datasets. Large benchmark assets remain embedded
so existing configs do not acquire new volume-mount requirements. MolmoSpaces
retains its upstream lazy scene/object downloads; importing the package at build
time does not make every benchmark episode available offline.

## Build and push

BuildKit and Linux x86-64 are required. Run on a machine where Docker use is
permitted and enough disk is available for the intermediate builder cache.

```bash
# Inspect every build command without Docker, downloads or package resolution.
docker/build.sh --dry-run

# Build the GPU artifact, then its CPU counterpart.
docker/build.sh libero --tag slim --profile all

# Derive a CPU image from an existing immutable GPU artifact without rebuilding it.
docker/build.sh libero --tag slim --profile cpu \
  --gpu-image ghcr.io/allenai/vla-evaluation-harness/libero@sha256:GPU_DIGEST

# Build all unrestricted images; restricted images require explicit opt-in.
docker/build.sh --tag slim --profile all
docker/build.sh behavior1k --tag slim --accept-license behavior1k

# Upstream RoboDojo is built separately; it is never based on the common builder.
docker/build.sh robodojo --base-image robodojo:cuda12.8 --accept-license robodojo

# Reuse an immutable builder/runtime pair from a previous release.
docker/build.sh libero --tag release \
  --base-image ghcr.io/allenai/vla-evaluation-harness/base@sha256:BUILDER_DIGEST \
  --runtime-image ghcr.io/allenai/vla-evaluation-harness/base-render@sha256:RUNTIME_DIGEST

# Reuse exact ManiSkill2 assets if the upstream download server is unavailable.
docker/build.sh maniskill2 --build-arg \
  MANISKILL_ASSET_IMAGE=ghcr.io/allenai/vla-evaluation-harness/maniskill2@sha256:710ebca79942e8c580af36cc71f7dacc340ba535f2be4efb17141d526ef4794a

# Override a pinned source intentionally.
docker/build.sh libero --build-arg LIBERO_REF=COMMIT_SHA

docker/push.sh --tag release --profile all libero
```

For a host-launched CPU evaluation, select both the software renderer and the
CPU image in the evaluation config:

```yaml
render: cpu
docker:
  image: ghcr.io/allenai/vla-evaluation-harness/libero:slim-cpu
```

The script works from any directory and builds only required bases. `--base-image`
overrides the builder, except for RoboDojo where it selects the external parent.
`--runtime-image` overrides the final shared runtime. A CPU build creates its GPU
source first unless `--gpu-image` supplies an existing artifact built with the
same recipe and runtime paths; pre-refactor images may use incompatible prefixes. `--build-arg` is forwarded
to every build in that invocation; build individual base targets separately when
an override should apply to only one base. Direct `docker build` users must supply
matching `BASE_IMAGE` and `RUNTIME_IMAGE` tags/digests themselves.

## Reproducibility

The Ubuntu/CUDA parent images and Pixi/uv binary images are pinned by registry digest. Source revisions
use immutable commit IDs; the LIBERO-Plus and RoboCerebra asset
downloads also select a dataset commit. Existing simulator version pins and NumPy
compatibility overrides are retained. RoboMME explicitly uses the upstream
PyTorch 2.9.1 / torchvision 0.24.1 pins with official CUDA 12.8 wheels; its previous
CUDA 12.1 attempt and fallback could not satisfy those pins. See the
[official PyTorch wheel matrix](https://pytorch.org/get-started/previous-versions/).
VLABench asset download failures now fail the build instead of producing a
silently incomplete image.

Each common-base benchmark stores an audit inventory in
`/usr/local/share/vla-build/`: Python version, installed pip packages, source
checkout revisions, runtime paths, runtime dpkg versions, and a Conda explicit
package list when applicable. These describe what was installed; **they are not a
complete replay lockfile**. In particular, pip freeze does not lock artifact
hashes, editable sources or build dependencies.

The existing `UV_EXCLUDE_NEWER=2026-07-07` cutoff bounds PyPI resolution.
Pixi locks the native Conda packages, including Python, CMake and urdfdom, without
combining incompatible simulator requirements into one solve. It does not lock
apt repositories, PyPI artifacts or upstream asset downloaders. Fully repeatable
source rebuilds still need benchmark-specific PyPI locks, apt snapshots and asset
checksum manifests. Preserve published images by digest to replay exact artifacts.

To update native packages intentionally, edit `docker/pixi/pixi.toml`, run
`pixi lock --manifest-path docker/pixi/pixi.toml` with Pixi 0.80.0, review the
lockfile and rebuild every affected benchmark. See the
[Pixi container guidance](https://pixi.prefix.dev/latest/deployment/container/).

The [uv Docker guidance](https://docs.astral.sh/uv/guides/integration/docker/)
and [Docker multi-stage build documentation](https://docs.docker.com/build/building/multi-stage/)
describe the environment-copy approach used here.

## Validation and size measurement

See [CPU/GPU and Pixi validation](validation-profiles.md) and the
[initial slimming measurements](validation.md).

Host tests exercise dependency ordering, license gates, custom base routing,
argument quoting and asset/source export without launching Docker:

```bash
uv run pytest tests/test_docker_build.py tests/test_docker_resources.py \
  tests/test_docker_pull_mirror.py tests/test_charliecloud.py
```

Before publishing, build every image on an approved worker and compare against
the previous release. Do not infer a measured size reduction from the Dockerfiles.
Each common benchmark runs `vla-eval --help` during its final-stage build; this
checks the copied interpreter and harness, not simulator compatibility.

For each benchmark, run its existing smoke config with the same seed, model,
assets and GPU driver as the baseline. Check simulator imports, reset, one action,
nonempty rendered frames, success metrics and clean shutdown. In particular test
RLBench GLX/Xvfb, DuoBench native linking, CALVIN Git calls, both Simpler patches,
RoboTwin JIT and MolmoSpaces symlinks. Review changed Python package versions
between the inventories; legacy metadata conflicts mean a blanket `pip check`
is not a sufficient compatibility criterion.

```bash
# On the approved Docker worker, record both image IDs and uncompressed sizes.
docker image inspect ghcr.io/allenai/vla-evaluation-harness/libero:BASELINE \
  --format '{{.Id}} {{.Size}}'
docker image inspect ghcr.io/allenai/vla-evaluation-harness/libero:slim \
  --format '{{.Id}} {{.Size}}'
docker history ghcr.io/allenai/vla-evaluation-harness/libero:slim
```

Image size, registry compressed transfer size and shared local disk use are
different quantities. Builder caches can remain large even when final images are
smaller; do not prune a shared daemon's cache to measure this change.

## Adding a benchmark

1. Choose the rendering or CUDA runtime, create an isolated build environment, and list
   all runtime source/asset paths in the exporter call.
2. Add native runtime packages explicitly and preserve required environment vars.
3. Add the name and CPU capability to `images.sh`, plus any build license gate. Extend the Docker-free tests and validate a simulator smoke run.

## Minimal CPU runtimes and separate assets

LIBERO CPU profiles convert initial-state files to NumPy during the build, verify
shape, dtype and exact array bytes, and remove PyTorch from the runtime. This
covers LIBERO, Pro, Plus and Mem. `libero-numpy-states.json` records original and
converted file hashes and loader patches. Custom PyTorch initial-state files must
be converted by rebuilding the CPU image. GPU profiles retain the original loader.

CPU exports remove Open3D's bundled CUDA implementation, PyTorch native test
executables and C++ headers, selected numerical-library tests, checkout docs, and notebook documents (preserving notebook asset directories).
`/usr/local/share/vla-build/cpu-pruned.json` records deleted paths and bytes.
These are evaluation runtimes; compiling new PyTorch C++ extensions requires a
builder image.

For the smallest runtime artifact, split simulator data from an existing profile:

```bash
docker/package_runtime.sh vlabench \
  ghcr.io/allenai/vla-evaluation-harness/vlabench:VERSION-cpu \
  ghcr.io/allenai/vla-evaluation-harness/vlabench:VERSION-cpu
uv run python docker/assets.py extract \
  ghcr.io/allenai/vla-evaluation-harness/vlabench:VERSION-cpu-assets ./vla-assets
uv run python docker/assets.py run \
  ghcr.io/allenai/vla-evaluation-harness/vlabench:VERSION-cpu-runtime ./vla-assets \
  test --config /workspace/configs/benchmarks/vlabench/eval.yaml
```

Packaging produces `-runtime` and `-assets` images without dependency resolution
or network access in build steps. The runtime preserves the source image’s environment and entrypoint. The asset image contains data only; extraction does not execute it. The
helper verifies every file's SHA-256 and symlink target, checks the runtime's
manifest, and mounts data at its original paths. Extraction also prints volume
entries for the harness's `docker.volumes` configuration. CPU/GPU profiles can
share an extracted bundle when their asset manifests match. Mounts are writable
because some simulators generate caches; run `verify` again to detect changes.

Keep both artifact digests for replay. Splitting assets reduces runtime download
size and allows data reuse; it does **not** eliminate the data's storage cost.
MolmoSpaces still downloads task-specific resources lazily. The unsplit profile
remains available for self-contained distribution. Packaging currently covers
the 14 CPU-capable benchmark families and their corresponding GPU profiles.
