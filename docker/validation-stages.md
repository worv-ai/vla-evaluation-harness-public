# Multi-stage validation (2026-09-14)

Measured local Linux/amd64 image sizes (decimal GB). Immutable IDs and frame
hashes are recorded in [validation-stages.json](validation-stages.json).

| Image | Full | Runtime | Assets |
| --- | ---: | ---: | ---: |
| LIBERO CPU | 2.383 | 1.344 | 1.039 |
| LIBERO GPU | 3.077 | 2.039 | 1.039 |
| Kinetix CPU | 1.572 | 1.572 | empty manifest |
| Kinetix GPU | — | 6.798 | — |

- LIBERO CPU full and automatically mounted split runtime passed reset, one
  action and rendering with identical frame hashes. PyTorch is absent.
- LIBERO GPU passed the same simulation smoke using NVIDIA EGL.
- LIBERO CPU/GPU asset tags resolve to the same image ID.
- Kinetix CPU and GPU passed reset, action and pixel rendering with identical
  frame hashes. JAX reported `cuda:0` in the GPU runtime and CPU devices in the
  CPU runtime. CPU images have no accelerator packages, Pixi or uv executables.
- Full suite: 596 passed, 2 skipped. Ruff, formatting, ty, generated-file checks,
  all recipe shell syntax, and all 33 hashed profile locks passed. Tests cover
  top-level asset exclusion, cache corruption, failed extraction cleanup and
  Charliecloud asset mounting.

Only LIBERO/Kinetix were rebuilt and exercised in this pass. Other recipes were
rendered and statically checked; their earlier functional results remain in
`validation-profiles.md`. Build-tool and proprietary installer resolution is
not fully hermetic despite final registry artifact hashes and asset tree locks.

The first Kinetix GPU export was interrupted when the shared disk filled.
After removing exact disposable cache chains from this work, the runtime build
and GPU smoke completed. Tagged images and unrelated caches were preserved;
final free space was approximately 56 GB. No images were published.

## Shared harness installation and reporting (2026-09-16)

The shared generator now installs hash-locked dependencies before copying the
harness. Harness installation uses `--no-deps`, avoiding another dependency
resolution and preserving simulator version exceptions. All 19 standard recipes
use this order; the upstream install plus final hash verification still both run
on cold builds.

- 600 tests passed, 2 skipped; Ruff, formatting, ty and generated-file checks passed.
- An incremental build on the recorded LIBERO CPU image installed the current
  harness without adding PyTorch or changing NumPy 1.24.4. Reset/action/render
  passed through the new `docker/report.py` command.
- The same report ran RLBench CPU reset/action/render through its existing
  tini/Xvfb entrypoint. Smoke failures also propagate as nonzero report exits.
- Full benchmark images were not rebuilt for this follow-up: only about 4 GB
  remained on the shared disk. The incremental image includes uv for installation
  testing and is not a release-size measurement. No additional size reduction is
  claimed; the change improves reuse of dependency layers on code/config edits.
