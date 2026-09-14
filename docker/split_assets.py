"""Inventory simulator data separately from executable runtime files."""

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

# These directories contain simulator data, not Python modules. Keep original paths
# so XML includes and upstream absolute symlinks continue to resolve after mounting.
PATTERNS = [
    "/assets",
    "/cache/molmo-spaces-resources",
    "/app/*/libero/libero/assets",
    "/app/*/notebooks/custom_assets",
    "/app/*/robosuite/models/assets",
    "/app/*/robocasa/models/assets",
    "/app/VLABench/VLABench/assets",
    "/app/calvin/calvin_env/data",
    "/app/calvin/calvin_env/tacto/meshes",
    "/opt/rcs/assets",
    "/opt/duobench/assets",
    "/tmp/RLBench/rlbench/assets",
    "/app/ManiSkill/mani_skill/assets",
    "/app/robomme_benchmark/assets",
]


def asset_paths():
    patterns = PATTERNS + [sys.prefix + "/lib/python*/site-packages/robosuite/models/assets"]
    paths = sorted(
        {Path(p) for pattern in patterns for p in glob.glob(pattern) if Path(p).is_dir() and not Path(p).is_symlink()}
    )
    return [p for p in paths if not any(parent in paths for parent in p.parents)]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(paths):
    files = []
    for root in paths:
        for directory, dirs, names in os.walk(root, followlinks=False):
            for name in sorted(dirs + names):
                path = Path(directory, name)
                if path.is_symlink():
                    files.append({"path": str(path), "symlink": os.readlink(path)})
                elif path.is_file():
                    files.append({"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size})
    return {"version": 1, "paths": [str(p) for p in paths], "files": sorted(files, key=lambda x: x["path"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--source-id")
    args = parser.parse_args()
    paths = asset_paths()
    if args.list:
        print("\n".join(str(p) for p in paths))
        return
    manifest = inventory(paths)
    manifest["source_image_id"] = args.source_id
    output = Path("/usr/local/share/vla-build/assets.json")
    output.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    for path in paths:
        shutil.rmtree(path)
        path.mkdir(parents=True)
    print(
        "Externalized {} asset bytes across {} directories".format(
            sum(f.get("bytes", 0) for f in manifest["files"]), len(paths)
        )
    )


if __name__ == "__main__":
    main()
