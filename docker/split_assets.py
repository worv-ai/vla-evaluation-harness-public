"""Hash explicit simulator data paths, preserving symlink targets."""

import hashlib
import os
from pathlib import Path


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
