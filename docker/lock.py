"""Refresh locks/hashes.txt: artifact hashes for every pin in locks/*.in, CPython 3.11 x86-64 Linux only."""

import argparse
import json
import re
import urllib.error
import urllib.request
from urllib.parse import unquote
from concurrent.futures import ThreadPoolExecutor

from catalog import BENCHMARKS, ROOT, lock_file, normalize
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import parse_wheel_filename

PLATFORMS = [f"manylinux_2_{minor}_x86_64" for minor in range(35, 4, -1)]
PLATFORMS += ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64"]
TAGS = {*cpython_tags((3, 11), platforms=PLATFORMS), *compatible_tags((3, 11), "cp311", PLATFORMS)}
HASHES = ROOT / "locks/hashes.txt"


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read().decode()


def pypi(name, version):
    try:
        data = json.loads(fetch(f"https://pypi.org/pypi/{name}/{version}/json"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return []
        raise
    return [(f["filename"], f["digests"]["sha256"]) for f in data["urls"]]


def pytorch(name, version, backend):
    page = fetch(f"https://download.pytorch.org/whl/{backend}/{name}/")
    files = [(unquote(f), d) for f, d in re.findall(r'href="[^"]*/([^/"#]+)#sha256=([0-9a-f]{64})"', page)]
    return [(file, digest) for file, digest in files if file.startswith(name.replace("-", "_") + "-" + version + "-")]


def select(files):
    """Wheels installable on the image, or the sdist when none is."""
    wheels = [d for f, d in files if f.endswith(".whl") and TAGS & set(parse_wheel_filename(f)[3])]
    return wheels or [d for f, d in files if f.endswith((".tar.gz", ".zip"))]


def resolve(pin, backends):
    name, version = pin.split("==")
    # Same order as install_locked.py: PyPI first, then the profile's PyTorch index.
    hashes = select(pypi(name, version))
    for backend in sorted(backends):
        hashes = hashes or select(pytorch(name, version, backend))
    if not hashes:
        raise RuntimeError("No CPython 3.11 x86-64 artifact for " + pin)
    return pin + " " + " ".join("--hash=sha256:" + h for h in sorted(hashes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Re-fetch pins that already have hashes")
    args = parser.parse_args()
    backends = {}
    for name, spec in BENCHMARKS.items():
        if "runtime_paths" not in spec:
            continue
        for profile in ("gpu", "cpu") if spec["cpu"] else ("gpu",):
            backend = "cpu" if profile == "cpu" else spec["torch_backend"]
            for line in lock_file(name, profile).read_text().split():
                backends.setdefault(normalize(line.split("==")[0]) + "==" + line.split("==")[1], set()).add(backend)
    known = {}
    if HASHES.exists() and not args.refresh:
        known = {line.split()[0]: line for line in HASHES.read_text().splitlines()}
    todo = sorted(set(backends) - set(known))
    with ThreadPoolExecutor(max_workers=8) as pool:
        for pin, line in zip(todo, pool.map(lambda p: resolve(p, backends[p]), todo)):
            known[pin] = line
            print(pin, flush=True)
    HASHES.write_text("".join(known[pin] + "\n" for pin in sorted(backends)))


if __name__ == "__main__":
    main()
