"""Install reviewed exact artifacts after upstream build steps, before runtime export."""

import argparse
import importlib.metadata as metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def install(name, profile):
    root = Path("/usr/local/lib/vla/locks")
    lock = root / (name + "-" + profile + ".txt")
    if not lock.exists():
        raise FileNotFoundError("Missing hashed runtime lock: " + str(lock))
    # All registry packages are reinstalled with hash verification. Editable/VCS
    # simulator packages stay at the source revisions recorded by their recipes.
    environment = {k: v for k, v in os.environ.items() if k not in ("UV_TORCH_BACKEND", "UV_EXCLUDE_NEWER")}
    subprocess.run(
        [
            "uv",
            "--no-config",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-cache",
            "--no-deps",
            "--reinstall",
            "--require-hashes",
            "--index-strategy",
            "unsafe-first-match",
            "--index",
            "https://pypi.org/simple",
            "--default-index",
            "https://download.pytorch.org/whl/"
            + (
                "cpu"
                if profile == "cpu"
                else json.loads(Path("/usr/local/lib/vla/images.json").read_text())["benchmarks"][name][
                    "torch_backend"
                ]
            ),
            "-r",
            str(lock),
        ],
        check=True,
        env=environment,
    )
    expected = dict(re.findall(r"^([\w.-]+)==([^\s\\]+)", lock.read_text(), re.M))
    installed = {re.sub(r"[-_.]+", "-", d.metadata["Name"].lower()): d.version for d in metadata.distributions()}
    mismatches = {n: (v, installed.get(n)) for n, v in expected.items() if installed.get(n) != v}
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
