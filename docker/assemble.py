"""Assemble runtime files without copying asset payloads into an intermediate tree."""

import argparse
import json
import os
import shutil
from pathlib import Path

from split_assets import inventory


def assemble(spec, source, destination):
    assets = [Path(path) for path in spec["assets"] if (source / Path(path).relative_to("/")).exists()]
    manifest = inventory([source / p.relative_to("/") for p in assets])
    for entry in manifest["files"]:
        entry["path"] = "/" + str(Path(entry["path"]).relative_to(source))
    manifest["paths"] = [str(p) for p in assets]
    # Hashes bind data to runtime independently of CPU/GPU package versions.
    for value in [*spec["runtime_paths"], "/usr/local/share/vla-build"]:
        relative = Path(value).relative_to("/")
        original, target = source / relative, destination / relative
        if Path(value) in assets:
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not original.exists():
            if value in ("/app", "/root"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            raise FileNotFoundError(original)
        target.parent.mkdir(parents=True, exist_ok=True)

        def ignore(directory, names):
            return [name for name in names if Path("/") / (Path(directory) / name).relative_to(source) in assets]

        if original.is_dir() and not original.is_symlink():
            shutil.copytree(original, target, symlinks=True, ignore=ignore, dirs_exist_ok=True)
        elif original.is_symlink():
            target.symlink_to(os.readlink(str(original)))
        else:
            shutil.copy2(original, target)
    for asset in assets:
        (destination / asset.relative_to("/")).mkdir(parents=True, exist_ok=True)
    output = destination / "usr/local/share/vla-build/assets.json"
    output.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark")
    parser.add_argument("--source", type=Path, default=Path("/"))
    parser.add_argument("--destination", type=Path, default=Path("/runtime-root"))
    args = parser.parse_args()
    catalog = json.loads(Path("/usr/local/lib/vla/images.json").read_text())
    spec = catalog["benchmarks"][args.benchmark]
    # Optional asset directories must still exist for static COPY instructions.
    for value in spec["assets"]:
        (args.source / Path(value).relative_to("/")).mkdir(parents=True, exist_ok=True)
    assemble(spec, args.source, args.destination)
    from verify_assets import verify_lock

    manifest = json.loads((args.destination / "usr/local/share/vla-build/assets.json").read_text())
    lock = json.loads(Path("/usr/local/lib/vla/locks/assets.json").read_text())
    verify_lock(args.benchmark, manifest, lock)


if __name__ == "__main__":
    main()
