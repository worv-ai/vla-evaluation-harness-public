"""Check a data manifest against the reviewed per-benchmark tree digest."""

import hashlib
import json
from pathlib import Path


def tree_digest(manifest):
    payload = {key: manifest[key] for key in ("version", "paths", "files")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_lock(name, manifest, lock):
    expected = lock.get(name)
    if expected is None:
        raise ValueError("Missing asset lock for " + name + "; review and lock the data before building")
    actual = tree_digest(manifest)
    if actual != expected["sha256"]:
        raise ValueError("Asset content changed for " + name + ": expected " + expected["sha256"] + ", got " + actual)


def main():
    import sys
    from split_assets import inventory

    name = sys.argv[1]
    catalog = json.loads(Path("/tmp/images.json").read_text())
    paths = [Path(p) for p in catalog["benchmarks"][name]["assets"]]
    manifest = inventory(paths)
    print(
        json.dumps(
            {
                "sha256": tree_digest(manifest),
                "paths": manifest["paths"],
                "files": len(manifest["files"]),
                "bytes": sum(f.get("bytes", 0) for f in manifest["files"]),
            }
        )
    )


if __name__ == "__main__":
    main()
