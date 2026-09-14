"""Regenerate hashed runtime package locks from reviewed exact-version inputs."""

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

from catalog import BENCHMARKS, ROOT


def compile_lock(path):
    name, profile = path.stem.rsplit("-", 1)
    spec = BENCHMARKS[name]
    recipe = (ROOT / "recipes" / (name + ".Dockerfile")).read_text()
    import re

    version = re.search(r"ARG PYTHON_VERSION=([0-9.]+)", recipe)
    python = version[1] if version else {"behavior1k": "3.10", "duobench": "3.11"}.get(name, "3.8")
    backend = "cpu" if profile == "cpu" else spec["torch_backend"]
    command = [
        "uv",
        "--no-config",
        "pip",
        "compile",
        "--no-deps",
        "--generate-hashes",
        "--no-header",
        "--no-annotate",
        "--python-version",
        python,
        "--python-platform",
        "x86_64-manylinux_2_35",
        "--index-strategy",
        "unsafe-first-match",
        "--index",
        "https://pypi.org/simple",
        "--default-index",
        "https://download.pytorch.org/whl/" + backend,
        str(path),
        "-o",
        str(path.with_suffix(".txt")),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(path.name + ": " + result.stderr[-2000:])
    return path.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*")
    args = parser.parse_args()
    paths = sorted((ROOT / "locks").glob("*.in"))
    if args.names:
        unknown = set(args.names) - {p.stem for p in paths}
        if unknown:
            parser.error("Unknown profile locks: " + ", ".join(sorted(unknown)))
        paths = [p for p in paths if p.stem in args.names]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name in pool.map(compile_lock, paths):
            print(name, flush=True)


if __name__ == "__main__":
    main()
