"""Compare two local images; optionally run the existing benchmark smoke test."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def tree_bytes(root):
    total = 0
    for directory, _, files in os.walk(root):
        for name in files:
            path = Path(directory, name)
            if not path.is_symlink():
                try:
                    total += path.stat().st_size
                except FileNotFoundError:
                    pass
    return total


def inspect_contents():
    directories = {
        str(p): tree_bytes(p)
        for p in Path("/").iterdir()
        if p.is_dir() and not p.is_symlink() and p.name not in ("proc", "sys", "dev")
    }
    packages = []
    for dist in importlib.metadata.distributions():
        size = 0
        for entry in set(dist.files or []):
            path = Path(str(dist.locate_file(entry)))
            if path.is_file() and not path.is_symlink():
                size += path.stat().st_size
        packages.append({"name": dist.metadata["Name"], "bytes": size})
    return {
        "directories": sorted(directories.items(), key=lambda x: x[1], reverse=True)[:10],
        "packages": sorted(packages, key=lambda x: x["bytes"], reverse=True)[:10],
    }


def inspect_image(reference):
    result = subprocess.check_output(["docker", "image", "inspect", reference], text=True)
    data = json.loads(result)[0]
    return {"reference": reference, "id": data["Id"], "bytes": data["Size"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--output", type=Path, help="Write measurements and smoke result as JSON")
    parser.add_argument("--config", help="Smoke config path inside the image")
    parser.add_argument("--action", help="JSON action vector; required with --config")
    parser.add_argument("--render", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--gpus", default="device=0", help="Docker GPU selection for GPU smoke only")
    args = parser.parse_args()
    if bool(args.config) != bool(args.action):
        parser.error("--config and --action must be supplied together")
    if args.action:
        json.loads(args.action)
    before, after = inspect_image(args.before), inspect_image(args.after)
    report = {
        "before": before,
        "after": after,
        "delta_bytes": after["bytes"] - before["bytes"],
        "smoke": {"status": "not run"},
    }
    root = Path(__file__).resolve().parent
    common = ["docker", "run", "--rm", "--pull=never", "--network", "none", "--cpus", "2", "--memory", "8g"]
    raw = subprocess.check_output(
        common
        + [
            "--runtime=runc",
            "--entrypoint",
            "python",
            "--mount",
            "type=bind,src=" + str(root / "report.py") + ",dst=/tmp/vla-report.py,readonly",
            after["id"],
            "/tmp/vla-report.py",
            "--inside",
        ],
        text=True,
    )
    report["contents"] = json.loads(raw)
    code = 0
    if args.config:
        # Preserve native Xvfb/tini wrappers while replacing the CLI with the smoke runner.
        wrapper = (
            "import runpy, sys\nsys.argv = "
            + repr(["smoke", args.config, "--action", args.action, "--render", args.render])
            + '\nrunpy.run_path("/tmp/vla-smoke.py", run_name="__main__")'
        )
        flags = ["--runtime=runc"] if args.render == "cpu" else ["--gpus", args.gpus]
        import tempfile

        with tempfile.TemporaryDirectory(prefix="vla-smoke-") as temporary:
            shim = Path(temporary, "vla-eval")
            shim.write_text("#!/usr/bin/env python\n" + wrapper + "\n")
            shim.chmod(0o755)
            # Metadata supplies the original PATH; no environment activation changes.
            metadata = json.loads(subprocess.check_output(["docker", "image", "inspect", after["id"]], text=True))[0]
            path = next(
                (v[5:] for v in metadata["Config"].get("Env", []) if v.startswith("PATH=")),
                "/usr/local/bin:/usr/bin:/bin",
            )
            result = subprocess.run(
                common
                + flags
                + [
                    "-e",
                    "PATH=/tmp/vla-smoke-bin:" + path,
                    "-e",
                    "OMP_NUM_THREADS=1",
                    "-e",
                    "OPENBLAS_NUM_THREADS=1",
                    "-e",
                    "XLA_PYTHON_CLIENT_PREALLOCATE=false",
                    "--mount",
                    "type=bind,src=" + str(root / "smoke_benchmark.py") + ",dst=/tmp/vla-smoke.py,readonly",
                    "--mount",
                    "type=bind,src=" + str(shim) + ",dst=/tmp/vla-smoke-bin/vla-eval,readonly",
                    after["id"],
                ],
                capture_output=True,
                text=True,
            )
        code = result.returncode
        report["smoke"] = {
            "status": "passed" if code == 0 else "failed",
            "exit_code": code,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    print("Image sizes (uncompressed GB; shared layers are counted in each image):")
    print(
        "{:.3f} -> {:.3f} ({:+.3f} GB)".format(
            before["bytes"] / 1e9, after["bytes"] / 1e9, report["delta_bytes"] / 1e9
        )
    )
    print("Largest directories (recursive; includes assets):")
    for name, size in report["contents"]["directories"]:
        print("  {:.1f} MB {}".format(size / 1e6, name))
    print("Largest installed packages (metadata-listed files; excludes editable source trees):")
    for entry in report["contents"]["packages"]:
        print("  {:.1f} MB {}".format(entry["bytes"] / 1e6, entry["name"]))
    print("Smoke: " + report["smoke"]["status"])
    if code:
        print(report["smoke"]["stderr"], file=sys.stderr)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return code


if __name__ == "__main__":
    if sys.argv[1:] == ["--inside"]:
        print(json.dumps(inspect_contents()))
    else:
        sys.exit(main())
