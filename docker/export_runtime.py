"""Assemble a runtime tree without retaining intermediate Docker layers."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

OPTIONAL_PATHS = {"/app", "/root"}


def export_runtime(paths, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for value in paths:
        source = Path(value)
        if not source.is_absolute() or ".." in source.parts or source == Path("/"):
            raise ValueError("Runtime paths must be absolute, non-root paths")
        if not source.exists():
            if str(source) in OPTIONAL_PATHS:
                continue
            raise FileNotFoundError(source)
        target = destination / source.relative_to("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        # Move, not copy: large asset trees must not need a second scratch copy.
        # Symlinks keep their original targets, which have identical runtime paths.
        shutil.move(str(source), str(target))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    manifest = Path("/usr/local/share/vla-build")
    manifest.mkdir(parents=True, exist_ok=True)
    (manifest / "python.txt").write_text(sys.version + "\n")
    frozen = subprocess.check_output(["uv", "pip", "freeze", "--python", sys.executable], text=True)
    (manifest / "requirements.freeze.txt").write_text(frozen)
    if Path(sys.prefix, "conda-meta").is_dir():
        explicit = subprocess.check_output(
            ["/opt/conda/bin/conda", "list", "--explicit", "--prefix", sys.prefix], text=True
        )
        (manifest / "conda-explicit.txt").write_text(explicit)
    (manifest / "runtime-paths.json").write_text(json.dumps(args.paths, indent=2) + "\n")
    # Only installer caches are disposable. Simulator/HF caches can back asset symlinks.
    for value in ("/root/.cache/pip", "/root/.cache/uv"):
        shutil.rmtree(value, ignore_errors=True)
    export_runtime([*args.paths, str(manifest)], "/runtime-root")


if __name__ == "__main__":
    main()
