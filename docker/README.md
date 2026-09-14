# Docker Images

Each benchmark has an isolated runtime. Model inference usually runs outside the
simulator container; ManiSkill, Kinetix, CuRobo and Isaac Sim still require CUDA.

## Image hierarchy

`Dockerfile.base` builds three shared images from two targets:

| Image | Target | Contents |
| --- | --- | --- |
| `base` | `builder` | CUDA runtime, compilers, headers, Miniforge and uv; build use only |
| `base-runtime` | `runtime` | CUDA 12.1 and EGL/Vulkan/X11 shared libraries |
| `base-cpu` | `runtime`, Ubuntu parent | EGL/Vulkan/X11 libraries without the CUDA compute runtime |

Benchmark Dockerfiles install into a `builder` stage, then export the final
Python environment, editable source trees, configs and assets into a fresh runtime
stage. This drops the Conda base environment, compiler layers and old copies of
packages replaced during installation. Removing files in a later `RUN` alone
would leave their bytes in earlier image layers.

The small `base-cpu` runtime has no system CUDA libraries but retains NVIDIA
injection and EGL/Vulkan GPU rendering. LIBERO variants, RoboCerebra, CALVIN,
RLBench, RoboCasa, RoboCasa365 and VLABench select CPU PyTorch wheels. Simpler
and DuoBench also use this base. ManiSkill2, MIKASA, RoboMME, MolmoSpaces and
Kinetix retain their CUDA-enabled PyTorch/JAX wheels, which provide their own
CUDA libraries; their final images do not duplicate the system CUDA runtime.
BEHAVIOR-1K and RoboTwin retain the original system CUDA runtime.

Python 3.10/3.11 environments use uv-managed, patch-pinned Python and `/opt/venv`.
Both the interpreter under `/opt/python` and the venv are copied at the same
absolute paths. Python 3.8 environments retain Conda Python. BEHAVIOR-1K and
DuoBench also retain their Conda environments for the existing libffi/urdfdom ABI.
The Conda package manager is absent from their runtime; `PATH` and `CONDA_PREFIX`
select the copied environment directly.

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

# Build dependencies once, then the requested benchmark.
docker/build.sh libero --tag slim

# Build all unrestricted images; restricted images require explicit opt-in.
docker/build.sh --tag slim
docker/build.sh behavior1k --tag slim --accept-license behavior1k

# Upstream RoboDojo is built separately; it is never based on the common builder.
docker/build.sh robodojo --base-image robodojo:cuda12.8 --accept-license robodojo

# Reuse an immutable builder/runtime pair from a previous release.
docker/build.sh libero --tag release \
  --base-image ghcr.io/allenai/vla-evaluation-harness/base@sha256:BUILDER_DIGEST \
  --runtime-image ghcr.io/allenai/vla-evaluation-harness/base-cpu@sha256:RUNTIME_DIGEST

# Reuse exact ManiSkill2 assets if the upstream download server is unavailable.
docker/build.sh maniskill2 --build-arg \
  MANISKILL_ASSET_IMAGE=ghcr.io/allenai/vla-evaluation-harness/maniskill2@sha256:710ebca79942e8c580af36cc71f7dacc340ba535f2be4efb17141d526ef4794a

# Override a pinned source intentionally.
docker/build.sh libero --build-arg LIBERO_REF=COMMIT_SHA

docker/push.sh --tag release libero
```

The script works from any directory and builds only required bases. `--base-image`
overrides the builder, except for RoboDojo where it selects the external parent.
`--runtime-image` overrides the final shared runtime. `--build-arg` is forwarded
to every build in that invocation; build individual base targets separately when
an override should apply to only one base. Direct `docker build` users must supply
matching `BASE_IMAGE` and `RUNTIME_IMAGE` tags/digests themselves.

## Reproducibility

The Ubuntu/CUDA parent images are pinned by registry digest, and the versioned
Miniforge installer is checked against its published SHA-256. Source revisions
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

The existing `UV_EXCLUDE_NEWER=2026-07-07` cutoff bounds PyPI resolution, but does
not freeze Conda repodata, apt repositories or all upstream asset downloaders.
Some benchmarks intentionally install NumPy versions outside the harness's
metadata constraints. A single combined `uv lock`/Pixi solve would fail or change
those environments. Pixi is therefore not introduced in this refactor: copying
existing native environments removes Conda from runtime without also changing
their solver and ABI. Benchmark-specific validated locks, package mirrors and
asset checksum manifests remain necessary for fully repeatable source rebuilds.
Preserve and run published images by digest to replay the exact built artifact.

The [uv Docker guidance](https://docs.astral.sh/uv/guides/integration/docker/)
and [Docker multi-stage build documentation](https://docs.docker.com/build/building/multi-stage/)
describe the environment-copy approach used here.

## Validation and size measurement

See [measured results and runtime checks](validation.md) for this refactor.

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

1. Choose the CPU or CUDA runtime, create an isolated build environment, and list
   all runtime source/asset paths in the exporter call.
2. Add native runtime packages explicitly and preserve required environment vars.
3. Add the name to `build.sh` and `push.sh`, including any license gate and runtime
   selection. Extend the Docker-free tests and validate a simulator smoke run.
