"""Extract, verify and mount a versioned simulator asset image. Run with uv run."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

MANIFEST = Path("usr/local/share/vla-build/assets.json")


def docker(*args, capture=False):
    return subprocess.run(
        ["docker", *map(str, args)], check=True, text=True, stdout=subprocess.PIPE if capture else None
    ).stdout


def local_path(root, absolute):
    path = Path(absolute)
    if not path.is_absolute() or ".." in path.parts or path == Path("/"):
        raise ValueError("Invalid asset path: " + absolute)
    candidate = root / path.relative_to("/")
    # An asset itself can be an absolute symlink into another mounted asset tree.
    # Its parents must not escape the bundle through a symlink.
    try:
        candidate.parent.resolve().relative_to(root.resolve())
    except ValueError:
        raise ValueError("Asset parent escapes bundle: " + absolute) from None
    return candidate


def verify(root):
    manifest = json.loads((root / MANIFEST).read_text())
    for entry in manifest["files"]:
        path = local_path(root, entry["path"])
        if "symlink" in entry:
            if not path.is_symlink() or os.readlink(path) != entry["symlink"]:
                raise ValueError("Asset symlink mismatch: " + entry["path"])
        else:
            if path.is_symlink():
                raise ValueError("Expected an asset file: " + entry["path"])
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != entry["sha256"]:
                raise ValueError("Asset checksum mismatch: " + entry["path"])
    return manifest


def extract(image, destination, *, quiet=False):
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("Extraction destination must be empty")
    image_id = docker("image", "inspect", image, "--format", "{{.Id}}", capture=True).strip()
    container = docker(
        "create", "--label", "vla-eval.asset-extraction=true", image_id, "/not-executed", capture=True
    ).strip()
    try:
        docker("cp", container + ":/.", str(destination))
    finally:
        docker("rm", container)
    manifest = verify(destination)
    (destination / "asset-image-id.txt").write_text(image_id + "\n")
    if not quiet:
        print("Verified {} files from {}".format(len(manifest["files"]), image_id))
        print("Mounts for docker.volumes:")
        for path in manifest["paths"]:
            print("  - " + json.dumps(str(local_path(destination, path)) + ":" + path))


def run(image, root, arguments):
    manifest = verify(root)
    # A bundle from another simulator revision must not silently change evaluation.
    raw = docker(
        "run",
        "--rm",
        "--pull=never",
        "--runtime=runc",
        "--network",
        "none",
        "--entrypoint",
        "cat",
        image,
        "/" + str(MANIFEST),
        capture=True,
    )
    if any(json.loads(raw)[key] != manifest[key] for key in ("version", "paths", "files")):
        raise ValueError("Runtime and asset manifests differ")
    profile = docker("image", "inspect", image, "--format", "{{json .Config.Env}}", capture=True)
    gpu = "VLA_IMAGE_PROFILE=gpu" in json.loads(profile)
    command = ["run", "--rm", "--pull=never", *(["--gpus", "all"] if gpu else ["--runtime=runc"])]
    for path in manifest["paths"]:
        command += ["--mount", "type=bind,src={},dst={}".format(local_path(root, path), path)]
    # Docker options precede the image; simulator arguments follow it.
    command += [image, *arguments]
    docker(*command)


def prepare_mounts(image, configuration, *, ensure_local):
    """Resolve one immutable bundle under a lock and validate it against the runtime."""
    import fcntl
    import tempfile
    import shutil

    asset_image = configuration.get("image")
    directory = configuration.get("directory")
    if not asset_image or not directory:
        raise ValueError("docker.assets requires image and directory")
    ensure_local(asset_image)
    image_id = docker("image", "inspect", asset_image, "--format", "{{.Id}}", capture=True).strip()
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / image_id.replace(":", "-")
    with (root / (destination.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.exists():
            temporary = Path(tempfile.mkdtemp(prefix=".extract-", dir=root))
            try:
                extract(image_id, temporary, quiet=True)
                temporary.rename(destination)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        manifest = verify(destination)
        raw = docker(
            "run",
            "--rm",
            "--pull=never",
            "--runtime=runc",
            "--network",
            "none",
            "--entrypoint",
            "cat",
            image,
            "/" + str(MANIFEST),
            capture=True,
        )
        expected = json.loads(raw)
        if any(expected[key] != manifest[key] for key in ("version", "paths", "files")):
            raise ValueError("Runtime and asset manifests differ")
    return [str(local_path(destination, path)) + ":" + path for path in manifest["paths"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("image")
    ex.add_argument("directory", type=Path)
    ve = sub.add_parser("verify")
    ve.add_argument("directory", type=Path)
    ru = sub.add_parser("run")
    ru.add_argument("image")
    ru.add_argument("directory", type=Path)
    ru.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.directory = args.directory.resolve()
    if args.command == "extract":
        extract(args.image, args.directory)
    elif args.command == "verify":
        verify(args.directory)
        print("Asset checksums verified")
    else:
        run(args.image, args.directory, args.arguments)


if __name__ == "__main__":
    main()
