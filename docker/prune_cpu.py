"""Remove development payloads from CPU evaluation environments, recording every deletion."""

import json
import shutil
import sys
from pathlib import Path


def prune(prefix=None, checkouts=Path("/app"), manifest=Path("/usr/local/share/vla-build/cpu-pruned.json")):
    prefix = Path(prefix or sys.prefix)
    sites = sorted(prefix.glob("lib/python*/site-packages"))
    candidates = []
    for site in sites:
        # Open3D's regular wheel bundles a second, CUDA-enabled native implementation.
        candidates += [site / "open3d/cuda", site / "torch/test", site / "torch/include"]
        for pattern in ("*Test", "test_*", "protoc*", "tutorial_*"):
            candidates.extend((site / "torch/bin").glob(pattern))
        # NumPy 2.x imports _core.tests from numpy.testing; preserve those resources.
        for package in ("scipy", "pandas", "matplotlib", "h5py", "sklearn", "sympy"):
            candidates.extend((site / package).rglob("tests"))
    for checkout in checkouts.glob("*"):
        for name in ("docs", ".github"):
            candidates.append(checkout / name)
        # LIBERO-Pro loads meshes from notebooks/custom_assets at runtime.
        candidates.extend((checkout / "notebooks").rglob("*.ipynb"))
    removed = []
    for path in sorted(set(candidates)):
        if not path.exists() or path.is_symlink():
            continue
        size = (
            sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())
            if path.is_dir()
            else path.stat().st_size
        )
        removed.append({"path": str(path), "bytes": size})
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    previous = json.loads(manifest.read_text()) if manifest.exists() else []
    manifest.write_text(json.dumps(previous + removed, indent=2) + "\n")
    print("Removed CPU development/CUDA payload: {} bytes".format(sum(row["bytes"] for row in removed)))


if __name__ == "__main__":
    prune()
