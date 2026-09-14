# Docker refactor validation

Validated on 2026-09-13 with Docker 28.5.1, Linux x86-64 and NVIDIA A100 GPUs.
The source worktree is based on `5808b58ebfac4fc0019a0bed0c4d2736325aa273`.
Local candidate tags use `slim-validation`; no images were published.

## Scope

Sizes below are Docker's uncompressed image sizes, in decimal GB. They are not
registry download sizes or incremental shared-disk usage. Baselines are existing
local release images, not fresh rebuilds: `0.5.0`, except RLBench/RoboDojo `latest`
and BEHAVIOR-1K `0.4.0`. Image IDs and exact byte counts are recorded in
`validation-sizes.json`. The build-only `base` intentionally keeps the toolchain;
its new runtime targets are 2.54 GB (CUDA) and 0.38 GB (without system CUDA).

| Image | Baseline GB | Candidate GB | Reduction |
| --- | ---: | ---: | ---: |
| `base` | 3.30 | 3.30 | -0.1% |
| `base-cpu` | — | 0.38 | — |
| `base-runtime` | — | 2.54 | — |
| `behavior1k` | 23.26 | 21.89 | 5.9% |
| `calvin` | 9.53 | 4.59 | 51.8% |
| `duobench` | 5.45 | 2.49 | 54.3% |
| `kinetix` | 10.13 | 6.80 | 32.9% |
| `libero` | 5.99 | 3.08 | 48.5% |
| `libero-mem` | 11.21 | 3.28 | 70.7% |
| `libero-plus` | 14.84 | 11.79 | 20.5% |
| `libero-pro` | 6.23 | 3.32 | 46.7% |
| `maniskill2` | 9.78 | 6.68 | 31.7% |
| `mikasa-robo` | 10.79 | 7.36 | 31.7% |
| `molmospaces` | 29.26 | 21.24 | 27.4% |
| `rlbench` | 4.80 | 1.88 | 60.8% |
| `robocasa` | 21.37 | 11.19 | 47.6% |
| `robocasa365` | 35.65 | 27.50 | 22.8% |
| `robocerebra` | 6.36 | 3.45 | 45.7% |
| `robodojo` | 36.31 | 36.31 | 0.0% |
| `robomme` | 16.98 | 8.86 | 47.8% |
| `robotwin` | 28.62 | 28.27 | 1.2% |
| `simpler` | 4.88 | 1.65 | 66.2% |
| `simpler-groot` | 4.88 | 1.65 | 66.1% |
| `simpler-xvla` | 4.88 | 1.65 | 66.1% |
| `vlabench` | 17.72 | 14.63 | 17.5% |

The sum of the 22 benchmark image sizes fell from 318.93 GB to
229.59 GB (28.0%). Shared layers are counted once per image;
this sum is not the daemon disk footprint. RoboDojo is effectively unchanged;
RoboTwin retains its CUDA/JIT toolchain and large assets and improves only 1.2%.

## Runtime checks

The final build stage checks `vla-eval --help`. This alone does not establish
simulator compatibility. Separate GPU runs reset a real environment, apply one
explicit action and check that rendered observations contain non-flat pixels.
These checks do not measure policy scores or cover every task/asset variation.

- LIBERO: baseline and candidate seed-7 frames matched byte-for-byte on repeat
  runs; one earlier run differed at the least significant pixel bit.
- Simpler: both camera hashes matched the baseline after reset/action, including
  the runtime without system CUDA. Both policy-specific image patches were tested.
- RoboCasa: both camera hashes matched when using the same harness source and
  explicitly seeding Python/NumPy after importing RoboCasa. Uncontrolled global
  RNG initialization changes the selected scene.
- LIBERO-PRO, LIBERO-Mem, LIBERO-Plus, RoboCerebra and RoboCasa365: reset, action
  and camera observations passed.
- RLBench: tested through its actual tini/Xvfb wrapper; rendering and CoppeliaSim
  shutdown passed.
- VLABench: reset, one action and a 480×480 camera observation passed.
- ManiSkill2 and MIKASA-Robo: reset, one action and camera observations passed
  without system CUDA.
- Kinetix: JAX pixel observations and one action passed without system CUDA.
  JAX GPU matrix multiplication also passed; ManiSkill2/MIKASA passed the
  equivalent PyTorch CUDA check.
- DuoBench: reset, Cartesian action and all three camera observations passed.
- RoboMME: default lavapipe rendering, reset, one action and both camera
  observations passed. CUDA PyTorch matrix multiplication was checked separately.
- RoboTwin: built CuRobo CUDA extensions, reset, one joint action and all three
  camera observations passed.
- MolmoSpaces: CUDA matrix multiplication passed. Both baseline and candidate
  failed an offline reset on the same lazily downloaded scene archive
  (`procthor-objaverse-val_val_1093.tar.zst`); common-asset preinstallation does
  not cover every episode. This network dependency predates the refactor.
  With downloads enabled, reset, one action and both camera observations passed.
- BEHAVIOR-1K: OmniGibson, Isaac Sim and CUDA PyTorch imports passed. Full scene
  execution requires its licensed dataset and suitable RTX hardware; it was not
  established on this A100 host.
- RoboDojo: the external upstream base is preserved. Its harness CLI is checked;
  full licensed Isaac Sim benchmark execution is not established here.

CALVIN's EGL device helper is compiled during the image build, because otherwise
its first reset invokes a compiler absent from the runtime. Reset, one action and
200×200 rendering passed when one evaluation sequence was generated before EGL
initialization; the frame hash matched the baseline with the same harness source.
A direct full-config smoke stalled in the upstream sequence
ProcessPoolExecutor after EGL initialization; the 1,000-sequence evaluation was
not validated. The smoke workaround changes initialization order only.

The original UCSD ManiSkill2 asset server timed out during both Docker download
and a separate HTTP check. The measured candidate reuses `/workspace/data` from
the baseline image via `MANISKILL_ASSET_IMAGE`, pinned to digest
`sha256:710ebca79942e8c580af36cc71f7dacc340ba535f2be4efb17141d526ef4794a`.
Default source builds still depend on that upstream server; this explicit option
preserves the existing assets rather than switching dataset versions.

RoboTwin's three downloaded archive IDs match the baseline's Hugging Face
metadata. The baseline dataset revision is
`9dc9299c163db059931898a9f0852098a61155a1`; the candidate downloaded revision
`785feb15aa4a4f532395ad2b1d2be5f28cb561ad`, with identical selected archive IDs:

| Archive | SHA-256 (Hugging Face LFS object ID) |
| --- | --- |
| `background_texture.zip` | `54ede0fb5b783e0faa2bc98720d3affd6ca3bb9280b225b48c1aafaf31473070` |
| `embodiments.zip` | `6b87d7d55e106d8ff25917e0538eb1e177fc549280e8a742a8cec3cb9f953fc6` |
| `objects.zip` | `6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27` |

## Host checks

79 tests passed across `test_docker_build.py`, `test_docker_resources.py`,
`test_docker_pull_mirror.py` and `test_charliecloud.py`. Ruff checks passed for the
new Python helpers/tests. All Dockerfile shell `RUN` blocks and build/push/wrapper
scripts passed Bash syntax checks.

## Repeating a simulator check

Run from the repository root on an approved Docker/GPU worker. Select the config
and action dimension appropriate to the simulator; this example uses LIBERO.
The helper uses the first configured task and episode zero without a model server.

```bash
docker run --rm --gpus device=0 --cpus 4 --memory 8g --network none \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -v "$PWD/docker/smoke_benchmark.py:/tmp/smoke.py:ro" \
  --entrypoint python \
  ghcr.io/allenai/vla-evaluation-harness/libero:slim-validation \
  /tmp/smoke.py /workspace/configs/benchmarks/libero/spatial.yaml \
  --action '[0,0,0,0,0,0,-1]'
```

RLBench requires the Xvfb wrapper; do not bypass it with a Python entrypoint.
Kinetix's default symbolic config has no images; use `observation_type: pixels`
for this rendering check. GPU drivers can introduce pixel-level differences.
