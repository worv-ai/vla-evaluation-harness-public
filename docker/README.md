# Docker images

Build on Linux x86-64 with BuildKit. Each benchmark has an isolated environment.
`VERSION-cpu` uses software rendering; `VERSION-gpu` enables NVIDIA rendering or
compute. The unsuffixed tag aliases the full GPU image. GPU rendering does not
necessarily need CUDA PyTorch.

## Architecture

`images.json` owns benchmark capabilities, source revisions, runtime paths and
asset paths. Edit it and `recipes/*.Dockerfile`, then run:

```bash
uv run python docker/render.py
uv run python docker/render.py --check
```

The generated Dockerfiles use these stages:

```
sources ── pinned repositories, shared across profiles
   ↓
builder ── Pixi/uv environment, build tools, profile-specific packages
   ↓
assembled ── checksum assets; copy only runtime files to /runtime-root
   ├── assets  (scratch, data + manifest)
   └── runtime (OS runtime + environment + code, no externalized data)
          ↓
        full (+ asset layers)
```

CPU builds directly select CPU dependencies and a CPU runtime base. They do not
build a GPU image first. Sources and native environment setup can share BuildKit
cache between profiles. Downloads that require simulator Python APIs still run
inside the benchmark builder. Runtime assembly excludes declared asset roots
before copying; it never copies the entire root filesystem. Full and split
images share runtime layers, and identical asset layers can be shared across
profiles.

| Base | Purpose |
| --- | --- |
| `base` | Ubuntu builder with compilers, Pixi and uv |
| `base-cuda` | Builder with system CUDA for BEHAVIOR/RoboTwin |
| `base-render` | NVIDIA EGL/Vulkan runtime without system CUDA |
| `base-runtime` | CUDA 12.1 runtime with NVIDIA rendering |
| `base-cpu` | Mesa software rendering, GPU visibility disabled |

Python 3.8, BEHAVIOR and DuoBench use checksum-locked native Pixi environments.
Other images use patch-pinned uv Python installations. No Miniforge base
environment, Pixi executable or uv installer is copied into ordinary runtimes.
CPU pruning removes development payloads and accelerator packages; supported
LIBERO variants convert initial states to NumPy and remove PyTorch entirely.

Exceptions are deliberate: RoboTwin retains tools for CuRobo JIT; CALVIN retains
Git for runtime version queries; RLBench retains CoppeliaSim, Xvfb and tini.
Simpler patch variants inherit their parent runtime. RoboDojo retains its
externally built licensed Isaac Sim base. RoboDojo supports the full layout only; `--layout all` retains that full image. RLBench, BEHAVIOR and RoboDojo require license
acceptance and are excluded from registry publishing.

## Build and publish

```bash
# CPU only: no GPU image prerequisite.
docker/build.sh libero --tag release --profile cpu --layout all

# Independent CPU/GPU profiles where supported.
docker/build.sh libero --tag release --profile all --layout split

# Inspect commands without Docker or network access.
docker/build.sh libero --profile all --layout all --dry-run

# Explicit license acceptance for local builds.
docker/build.sh rlbench --accept-license rlbench --profile cpu

# Override the external RoboDojo parent.
docker/build.sh robodojo --base-image robodojo:cuda12.8 --accept-license robodojo

# Reuse immutable builder/runtime bases.
docker/build.sh libero --profile cpu \
  --base-image ghcr.io/allenai/vla-evaluation-harness/base@sha256:BUILDER_DIGEST \
  --runtime-image ghcr.io/allenai/vla-evaluation-harness/base-cpu@sha256:RUNTIME_DIGEST

# Publish the same layout produced by build.sh; restricted images are refused.
docker/push.sh libero --tag release --profile cpu --layout all
```

`--layout full` (default) produces `VERSION-PROFILE`. `split` produces
`VERSION-PROFILE-runtime` and `VERSION-PROFILE-assets`; `all` produces all three.
Full images retain existing config compatibility. Do not sum image sizes to
estimate disk use: Docker shares layers.

ManiSkill2 supports an immutable asset fallback when its upstream server is down:

```bash
docker/build.sh maniskill2 --build-arg \
  MANISKILL_ASSET_IMAGE=ghcr.io/allenai/vla-evaluation-harness/maniskill2@sha256:710ebca79942e8c580af36cc71f7dacc340ba535f2be4efb17141d526ef4794a
```

## Run with separate assets

```yaml
render: cpu
docker:
  gpus: none
  image: ghcr.io/allenai/vla-evaluation-harness/libero:release-cpu-runtime
  assets:
    image: ghcr.io/allenai/vla-evaluation-harness/libero:release-cpu-assets
    directory: /absolute/path/to/asset-cache
```

The harness resolves the asset image ID, extracts it atomically under a file
lock, verifies file hashes and symlink targets, compares the runtime manifest,
and mounts original simulator paths. Later runs reuse that extraction and verify
it again. Docker uses `directory`; Charliecloud uses its image cache and checks
the same manifest. Extra `docker.volumes` are applied afterwards. Asset mounts
are writable for upstream simulators; changed locked files fail the next check.
MolmoSpaces retains lazy downloads, so its initial bundle does not guarantee
that every scene runs offline.

Manual inspection remains available with `uv run python -m vla_eval.assets
extract|verify|run` (see `--help`). Asset images contain no shell or entrypoint
and are never executed during extraction.

## Reproducibility and updates

- OS, Pixi, uv and CUDA parents are pinned by digest.
- Ubuntu packages use the dated snapshot in `Dockerfile.base`, inherited by
  later apt installs. NVIDIA's separate package repository is not snapshotted.
- `pixi/pixi.lock` fixes native packages by URL and checksum.
- `locks/NAME-PROFILE.in` records reviewed exact registry package versions;
  corresponding `.txt` files pin artifact hashes. The builder reinstalls these
  with hash verification before export. Editable/VCS packages retain recipe
  source revisions. Intermediate upstream build dependency resolution and
  proprietary installers are not fully hermetic.
- `locks/assets.json` fixes each explicit asset tree's content digest. Changed
  data fails the build rather than silently changing evaluation inputs.
- `/usr/local/share/vla-build/` records sources, native/apt/package inventories,
  applied runtime lock, snapshot and asset manifest.

To update packages, review `.in` versions and regenerate hashes with
`uv run python docker/lock.py [NAME-PROFILE ...]`. Update asset locks only after
reviewing the new tree and validating a simulator reset/action/render. Use
`uv run python docker/lock_assets.py NAME LOCAL_BUILDER_IMAGE` to inspect it,
then add `--write` to record it. Build that local image with `--target builder`
when the old asset lock intentionally needs replacing. Changing
a source revision can require both package and asset lock updates. A matching
inventory alone is not a functional smoke test.

Current multi-stage results are in [validation-stages.md](validation-stages.md).
Historical size and smoke results are in `validation-profiles.md` and
`validation-minimal.json`; they describe their recorded image IDs, not every
subsequent source revision.
