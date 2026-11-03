"""Fetch an image's layers into ch-image's download cache with resumable HTTP range requests, then pull.

    python scripts/profile/fetch_layers.py ghcr.io/worv-ai/vla-evaluation-harness-public/vlabench:0.7.0

ch-image downloads each layer in one streamed request and cannot resume; ghcr.io (or a proxy on the way) cuts
multi-GB layers at ~6 GB on every attempt here. This resumes with ``curl -C -`` per layer, verifies the digest,
places the finished ``<sha256>.tar.gz`` in ``$CH_IMAGE_STORAGE/dlcache`` and runs ``ch-image pull``, which then
finds every layer cached.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    ]
)


def get(url: str, token: str | None = None, accept: str | None = None) -> bytes:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if accept:
        req.add_header("Accept", accept)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main() -> None:
    image = sys.argv[1]
    registry, _, rest = image.partition("/")
    repo, _, tag = rest.rpartition(":")
    token = json.loads(get(f"https://{registry}/token?scope=repository:{repo}:pull"))["token"]
    manifest = json.loads(get(f"https://{registry}/v2/{repo}/manifests/{tag}", token, ACCEPT))
    if "manifests" in manifest:  # multi-arch index: take linux/amd64
        digest = next(
            m["digest"] for m in manifest["manifests"] if m.get("platform", {}).get("architecture") == "amd64"
        )
        manifest = json.loads(get(f"https://{registry}/v2/{repo}/manifests/{digest}", token, ACCEPT))
    layers = [layer["digest"].split(":", 1)[1] for layer in manifest["layers"]]
    sizes = {layer["digest"].split(":", 1)[1]: layer["size"] for layer in manifest["layers"]}
    storage = (
        os.environ.get("CH_IMAGE_STORAGE")
        or subprocess.check_output(["ch-image", "gestalt", "storage-path"], text=True).strip()
    )
    dlcache = Path(storage) / "dlcache"
    dlcache.mkdir(parents=True, exist_ok=True)
    for i, sha in enumerate(layers, 1):
        final = dlcache / f"{sha}.tar.gz"
        if final.exists() and final.stat().st_size == sizes[sha]:
            print(f"layer {i}/{len(layers)} {sha[:7]}: cached", flush=True)
            continue
        part = dlcache / f"part_{sha}.tar.gz"
        for attempt in range(1, 41):
            token = json.loads(get(f"https://{registry}/token?scope=repository:{repo}:pull"))["token"]
            rc = subprocess.call(
                [
                    "curl",
                    "-sSL",
                    "-H",
                    f"Authorization: Bearer {token}",
                    "-C",
                    "-",
                    "-o",
                    str(part),
                    f"https://{registry}/v2/{repo}/blobs/sha256:{sha}",
                ]
            )
            have = part.stat().st_size if part.exists() else 0
            print(f"layer {i}/{len(layers)} {sha[:7]}: attempt {attempt} rc={rc} {have}/{sizes[sha]}", flush=True)
            if have >= sizes[sha]:
                break
        h = hashlib.sha256()
        with open(part, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != sha:
            print(f"layer {sha[:7]}: digest mismatch, removing", flush=True)
            part.unlink()
            sys.exit(1)
        part.rename(final)
    import time

    for attempt in range(30):  # the storage lock is held by any concurrent ch-image; wait it out
        if subprocess.call(["ch-image", "pull", image]) == 0:
            print("PULL_OK", flush=True)
            return
        time.sleep(60)
    print("PULL_FAILED", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
