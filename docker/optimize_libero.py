"""Convert LIBERO initial states losslessly so evaluation does not need PyTorch."""

import hashlib
import importlib.metadata as metadata
import json
import os
import re
import sys
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert_state(path):
    import numpy as np
    import torch

    value = torch.load(str(path), map_location="cpu", weights_only=False)
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.kind not in "fiu":
        raise ValueError("Unsupported initial-state type: " + str(path))
    target = Path(str(path) + ".npy")
    np.save(str(target), array, allow_pickle=False)
    replay = np.load(str(target), allow_pickle=False)
    if replay.dtype != array.dtype or replay.shape != array.shape or replay.tobytes() != array.tobytes():
        raise ValueError("Initial-state conversion changed values: " + str(path))
    return {
        "source": str(path),
        "source_sha256": sha256(path),
        "target": str(target),
        "target_sha256": sha256(target),
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }


def remove_wheel(name):
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return 0
    paths = [Path(os.path.abspath(dist.locate_file(f))) for f in dist.files or []]
    prefix = Path(sys.prefix).resolve()
    for path in paths:
        path.relative_to(prefix)
    removed = 0
    for path in paths:
        if path.is_file() or path.is_symlink():
            removed += path.lstat().st_size
            path.unlink()
    parents = {parent for path in paths for parent in path.parents if parent != prefix and prefix in parent.parents}
    for parent in sorted(parents, key=lambda p: len(p.parts), reverse=True):
        try:
            parent.rmdir()
        except OSError:
            pass
    return removed


def optimize():
    modules = sorted(Path("/app").glob("*/libero/libero/benchmark/__init__.py"))
    if not modules or all("NUMPY_INIT_STATES = True" in p.read_text() for p in modules):
        return
    patches = []
    states = []
    for module in modules:
        source = module.read_text()
        calls = source.count("torch.load(init_states_path)")
        if not calls or source.count("torch.") != calls or source.count("import torch\n") != 1:
            raise ValueError("Unsupported LIBERO loader; retain PyTorch until it is reviewed: " + str(module))
        files = sorted(module.parents[3].rglob("*.pruned_init"))
        if not files:
            raise ValueError("No LIBERO initial states found: " + str(module))
        states.extend(convert_state(path) for path in files)
        patched = source.replace("import torch\n", "import numpy as np\nNUMPY_INIT_STATES = True\n")
        patched = patched.replace(
            "torch.load(init_states_path)", 'np.load(init_states_path + ".npy", allow_pickle=False)'
        )
        patches.append({"path": str(module), "before_sha256": sha256(module)})
        module.write_text(patched)
        patches[-1]["after_sha256"] = sha256(module)
    adapter = Path("/workspace/src/vla_eval/benchmarks/libero/benchmark.py")
    source = adapter.read_text()
    if "NUMPY_INIT_STATES" not in source:
        # Older built harness artifacts apply an unconditional torch.load patch.
        pattern = r"        # LIBERO init states use torch.save.*?        from libero.libero import benchmark\n"
        patched, count = re.subn(pattern, "        from libero.libero import benchmark\n", source, flags=re.S)
        if count != 1:
            raise ValueError("Unrecognized harness LIBERO loader")
        patches.append({"path": str(adapter), "before_sha256": sha256(adapter)})
        adapter.write_text(patched)
        patches[-1]["after_sha256"] = sha256(adapter)
    for state in states:
        Path(state["source"]).unlink()
    wheels = {}
    for name in ("torchvision", "torchaudio", "torch"):
        try:
            wheels[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    removed = sum(remove_wheel(name) for name in ("torchvision", "torchaudio", "torch"))
    freeze = Path("/usr/local/share/vla-build/requirements.freeze.txt")
    if freeze.exists():
        freeze.write_text(
            "".join(
                line
                for line in freeze.read_text().splitlines(keepends=True)
                if not re.match(r"^(torch|torchvision|torchaudio)(==| @)", line)
            )
        )
    manifest = {"states": states, "patches": patches, "removed_wheels": wheels, "removed_wheel_bytes": removed}
    Path("/usr/local/share/vla-build/libero-numpy-states.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Converted {} initial-state files; removed {} PyTorch bytes".format(len(states), removed))


if __name__ == "__main__":
    optimize()
