# CPU/GPU profiles and Pixi validation

This follows the slimming checkpoint `cbb57ad`; its release comparisons remain in
[validation.md](validation.md). Local validation tags use `pixi-validation-gpu`
and `pixi-validation-cpu`. No registry publication is part of this validation.

## Reproducibility scope

Pixi 0.80.0 installs four native environments with `--locked`. Comparing locked
artifact URLs against the checkpoint's Conda inventories gives exact matches for
`py38`, `behavior` and `duobench`. CALVIN preserves all remaining artifact URLs
while dropping the orphaned `icu-78.3` package. `pixi lock --check` passes.

CPU conversion starts from the GPU artifact, preserves its native environment,
simulator sources and assets, replaces PyTorch wheels at the same public version,
and removes accelerator dependencies. Asset layers are copied separately, but
actual cross-profile layer sharing is not assumed in size comparisons. `/usr/local/share/vla-build` records
the native lock, installed packages, source revisions and GPU source reference.

Native Conda artifacts are locked; apt, all PyPI artifacts and all upstream asset
downloaders are not. Use published image digests for exact artifact replay.

Docker routing, export, mirror and Charliecloud tests: **96 passed**. Shell syntax,
Python lint and whitespace checks pass.

## Runtime checks

CPU checks use `docker run --runtime=runc` without GPU attachment. They assert
that no CUDA package family, CUDA-enabled PyTorch, system CUDA tree, or exposed
NVIDIA device remains. Pixi, Conda and uv are absent from these final runtimes.
GPU checks attach one NVIDIA A100 and exercise the configured renderer.

Simulator checks reset one task, apply one explicit action, require non-flat
rendered observations, and cleanly shut down. These are compatibility checks,
not complete benchmark-score evaluations. CPU and GPU rasterizers may produce
different pixels.

CALVIN uses one evaluation sequence generated before EGL starts, avoiding the
upstream multiprocessing/fork interaction. Its GPU frame SHA-256 matches the
previous release: `b6a39bad353b53d48502436bc07ad53b9bde09ffc4815e45e11ca33334405ed6`.
Kinetix CPU and GPU frames also match in the tested pixel-observation task.
RLBench checks its actual tini/profile/Xvfb entrypoint chain, including automatic
`--render cpu` injection.

MolmoSpaces retains upstream lazy scene/object downloads. BEHAVIOR requires
licensed assets and compatible RTX hardware for full scene validation; RoboDojo
retains its external licensed base.

## Minimal runtime follow-up

The initial CPU split still embedded all simulator assets. The follow-up converts LIBERO initial states losslessly to NumPy and removes
PyTorch from its CPU evaluation runtime. Other CPU exports remove Open3D's bundled CUDA binaries, native PyTorch tests/headers, selected library
tests, checkout documentation and notebook documents. NumPy test resources and
LIBERO-Pro's `notebooks/custom_assets` are retained because upstream imports and
simulator XML loading use them at runtime. Every deletion is recorded in
`cpu-pruned.json`.

All 14 CPU benchmark families pass reset/action/render after applying the pruning
script to their built CPU environments. The additional packaging step creates
standalone runtime and data images from an existing artifact, without installing
packages. It preserves environment, entrypoint and command, and records SHA-256
checksums plus symlink targets for the externalized data. The asset helper rejects
modified files and mismatched runtime/bundle manifests.

Actual split-image smoke checks passed for LIBERO, VLABench and RLBench, including
asset extraction, checksum verification, reset/action/render and a second checksum
verification. RLBench exercises its actual tini/Xvfb entrypoint. These checks also
cover native Pixi and uv environments, and additional runtime OS libraries.

Measured bytes and image IDs are in [validation-profile-sizes.json](validation-profile-sizes.json).
Sizes are uncompressed Docker image sizes; summing them double-counts shared layers.
Asset separation reduces the executable artifact, not the total data requirement.
The 14-profile pruning checks use disposable containers; complete minimal artifact
builds were performed for the three representative families above.

LIBERO CPU initial states are converted with a per-file shape, dtype and byte
comparison. All four LIBERO variants passed reset/action/render without PyTorch;
the base LIBERO frame hash matches the pre-conversion CPU image exactly. The base
image converts 130 state files and removes approximately 668 MB of PyTorch files.
`libero-numpy-states.json` records both serialization hashes and loader patch hashes.

## Measured CPU artifacts

| Benchmark | Initial CPU image (GB) | Minimal runtime (GB) | Separate data image (GB) |
|---|---:|---:|---:|
| libero | 3.08 | 1.34 | 1.04 |
| vlabench | 14.63 | 1.69 | 12.07 |
| rlbench | 1.88 | 1.77 | 0.09 |

The initial CPU images already had CUDA wheels removed. LIBERO's self-contained
CPU image additionally falls from 3.08 GB to 2.38 GB before separating its data.
VLABench deletes about 0.89 GB of development/CUDA payload; approximately 12.07 GB
of its reduction is data externalization. File-level deletion totals are recorded
in [validation-minimal.json](validation-minimal.json).

The complete profile matrix was built: 22 GPU and 14 CPU images. Actual simulator
reset/action/render checks passed for 20 GPU and 14 CPU profiles. BEHAVIOR passed
OmniGibson/Isaac Sim imports and a CUDA matrix calculation; RoboDojo passed CLI
startup. Full licensed BEHAVIOR/RoboDojo scene runs were not validated.

VLABench's intermediate full-profile artifacts were retired after measurement to
recover local disk space; its minimal runtime and data images remain available.
The size inventory marks retired references. Cleanup targeted only this task's
images and identified build-cache records; release images were retained.
