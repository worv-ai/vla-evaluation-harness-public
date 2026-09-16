"""Record the reviewed asset tree from a locally built benchmark builder image."""

import argparse
import json
import subprocess

from catalog import BENCHMARKS, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=sorted(n for n, s in BENCHMARKS.items() if "assets" in s))
    parser.add_argument("image", help="Local --target builder image to inventory; no image is pulled")
    parser.add_argument("--write", action="store_true", help="Replace the reviewed lock; otherwise print only")
    args = parser.parse_args()
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True
    ).strip()
    command = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--runtime=runc",
        "--network",
        "none",
        "--cpus",
        "2",
        "--entrypoint",
        "python",
    ]
    for name in ("images.json", "split_assets.py", "verify_assets.py"):
        command += ["--mount", "type=bind,src=" + str(ROOT / name) + ",dst=/tmp/" + name + ",readonly"]
    command += [image_id, "/tmp/verify_assets.py", args.benchmark]
    result = json.loads(subprocess.check_output(command, text=True))
    result["source_image"] = image_id
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.write:
        path = ROOT / "locks/assets.json"
        locks = json.loads(path.read_text())
        locks[args.benchmark] = result
        path.write_text(json.dumps(locks, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
