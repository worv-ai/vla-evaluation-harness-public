"""Install reviewed exact artifacts after upstream build steps, before runtime export."""

import argparse
import importlib.metadata as metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path("/usr/local/lib/vla")


def normalize(name):
    return re.sub(r"[-_.]+", "-", name.lower())


def install(name, profile):
    lock = ROOT / "locks" / (name + "-" + profile + ".in")
    lock = lock if lock.exists() else ROOT / "locks" / (name + ".in")
    pins = lock.read_text().split()
    hashes = {line.split()[0]: line for line in (ROOT / "locks/hashes.txt").read_text().splitlines()}
    requirements = []
    for pin in pins:
        package, version = pin.split("==")
        key = normalize(package) + "==" + version
        if key not in hashes:
            raise KeyError("No hashes for " + pin + "; run docker/lock.py")
        requirements.append(hashes[key])
    backend = (
        "cpu"
        if profile == "cpu"
        else json.loads((ROOT / "images.json").read_text())["benchmarks"][name]["torch_backend"]
    )
    # All registry packages are reinstalled with hash verification. Editable/VCS
    # simulator packages stay at the source revisions recorded by their recipes.
    environment = {k: v for k, v in os.environ.items() if k not in ("UV_TORCH_BACKEND", "UV_EXCLUDE_NEWER")}
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as file:
        file.write("\n".join(requirements) + "\n")
        file.flush()
        subprocess.run(
            ["uv", "--no-config", "pip", "install", "--python", sys.executable, "--no-cache", "--no-deps"]
            + ["--reinstall", "--require-hashes", "--index-strategy", "unsafe-first-match"]
            + ["--index", "https://pypi.org/simple", "--default-index", "https://download.pytorch.org/whl/" + backend]
            + ["-r", file.name],
            check=True,
            env=environment,
        )
    expected = dict(pin.split("==") for pin in pins)
    installed = {normalize(d.metadata["Name"]): d.version for d in metadata.distributions()}
    mismatches = {n: (v, installed.get(normalize(n))) for n, v in expected.items() if installed.get(normalize(n)) != v}
    if mismatches:
        raise RuntimeError("Runtime does not match lock: " + repr(mismatches))
    output = Path("/usr/local/share/vla-build")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(lock, output / "runtime-requirements.lock")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark")
    parser.add_argument("profile", choices=("cpu", "gpu"))
    args = parser.parse_args()
    install(args.benchmark, args.profile)
